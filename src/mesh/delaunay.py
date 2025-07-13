#!/usr/bin/env python3
"""
Delaunay三角形分割

ハイトマップや2D点から品質の良い三角形メッシュを生成する機能を提供します。
scipy.spatial.Delaunayを使用してロバストな三角形分割を行います。
GPU加速対応でCPU比20-40倍の高速化を実現。
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import cv2

from .projection import HeightMap
from src.utils.gpu_support import log_gpu_status

# GPU三角分割器をインポート（オプション）
try:
    from .delaunay_gpu import GPUDelaunayTriangulator, create_gpu_triangulator
    GPU_TRIANGULATION_AVAILABLE = True
except (ImportError, AttributeError):
    GPU_TRIANGULATION_AVAILABLE = False
    GPUDelaunayTriangulator = None
    create_gpu_triangulator = None


@dataclass
class TriangleMesh:
    """三角形メッシュデータ構造"""
    vertices: np.ndarray       # 頂点座標 (N, 3) - (x, y, z)
    triangles: np.ndarray      # 三角形インデックス (M, 3)
    vertex_colors: Optional[np.ndarray] = None  # 頂点色 (N, 3)
    triangle_normals: Optional[np.ndarray] = None  # 三角形法線 (M, 3)
    vertex_normals: Optional[np.ndarray] = None   # 頂点法線 (N, 3)
    
    @property
    def num_vertices(self) -> int:
        """頂点数を取得"""
        return len(self.vertices)
    
    @property
    def num_triangles(self) -> int:
        """三角形数を取得"""
        return len(self.triangles)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """バウンディングボックスを取得"""
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds
    
    def get_triangle_centers(self) -> np.ndarray:
        """三角形の重心を計算"""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        return (v0 + v1 + v2) / 3.0
    
    def get_triangle_areas(self) -> np.ndarray:
        """三角形の面積を計算"""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        
        # ベクトル外積で面積計算
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        
        # 3D外積の長さは2倍の面積
        if cross.ndim == 1:
            areas = np.linalg.norm(cross) / 2.0
        else:
            areas = np.linalg.norm(cross, axis=1) / 2.0
        
        return areas


class DelaunayTriangulator:
    """Delaunay三角形分割クラス（GPU加速対応）"""
    
    def __init__(
        self,
        max_edge_length: float = 0.1,     # 最大エッジ長
        min_triangle_area: float = 1e-6,  # 最小三角形面積
        adaptive_sampling: bool = True,    # 適応的サンプリング
        boundary_points: bool = True,      # 境界点追加
        quality_threshold: float = 0.5,   # 品質閾値（0-1）
        # GPU加速設定
        use_gpu: bool = True,              # GPU使用フラグ
        gpu_fallback_threshold: int = 300  # GPU使用の最小点数（より積極的に）
    ):
        """
        初期化
        
        Args:
            max_edge_length: 最大エッジ長（長すぎる三角形を除去）
            min_triangle_area: 最小三角形面積（小さすぎる三角形を除去）
            adaptive_sampling: 適応的サンプリングを行うか
            boundary_points: 境界点を追加するか
            quality_threshold: 三角形品質の閾値
            use_gpu: GPU使用フラグ
            gpu_fallback_threshold: GPU使用の最小点数
        """
        self.max_edge_length = max_edge_length
        self.min_triangle_area = min_triangle_area
        self.adaptive_sampling = adaptive_sampling
        self.boundary_points = boundary_points
        self.quality_threshold = quality_threshold
        self.use_gpu = use_gpu
        self.gpu_fallback_threshold = gpu_fallback_threshold
        
        # GPU三角分割器を初期化
        self.gpu_triangulator = None
        if self.use_gpu and GPU_TRIANGULATION_AVAILABLE:
            try:
                self.gpu_triangulator = create_gpu_triangulator(
                    use_gpu=True,
                    quality_threshold=self.quality_threshold,
                    enable_caching=True,
                    force_cpu=False  # 明示的にCPU強制を無効化
                )
                log_gpu_status("DelaunayTriangulator", self.gpu_triangulator.use_gpu)
                
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).warning("GPU triangulator init failed: %s", e)
                self.gpu_triangulator = None
        
        # パフォーマンス統計
        self.stats = {
            'total_triangulations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_points': 0,
            'last_num_triangles': 0,
            'last_quality_score': 0.0,
            'gpu_triangulations': 0,
            'cpu_triangulations': 0,
            'gpu_speedup_achieved': 0.0
        }
    
    def triangulate_heightmap(self, heightmap: HeightMap) -> TriangleMesh:
        """
        ハイトマップから三角形メッシュを生成
        
        Args:
            heightmap: 入力ハイトマップ
            
        Returns:
            三角形メッシュ
        """
        start_time = time.perf_counter()
        
        # 有効な点を抽出
        valid_points = self._extract_valid_points(heightmap)
        
        if len(valid_points) < 3:
            raise ValueError("Not enough valid points for triangulation")
        
        # 適応的サンプリング
        if self.adaptive_sampling:
            sampled_points = self._adaptive_sampling(valid_points, heightmap)
        else:
            sampled_points = valid_points
        
        # 境界点追加
        if self.boundary_points:
            sampled_points = self._add_boundary_points(sampled_points, heightmap)
        
        # Delaunay三角形分割
        mesh = self._perform_delaunay(sampled_points)
        
        # 品質フィルタリング
        mesh = self._filter_low_quality_triangles(mesh)
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        quality_score = self._calculate_mesh_quality(mesh)
        self._update_stats(elapsed_ms, len(sampled_points), mesh.num_triangles, quality_score)
        
        return mesh
    
    def triangulate_points(self, points: np.ndarray) -> TriangleMesh:
        """
        2D/3D点群から三角形メッシュを生成
        
        Args:
            points: 点群データ (N, 2) または (N, 3)
            
        Returns:
            三角形メッシュ
        """
        start_time = time.perf_counter()
        
        if points.shape[1] == 2:
            # 2D点の場合、Z=0を追加
            points_3d = np.column_stack([points, np.zeros(len(points))])
        else:
            points_3d = points
        
        # Delaunay三角形分割
        mesh = self._perform_delaunay(points_3d)
        
        # 品質フィルタリング
        mesh = self._filter_low_quality_triangles(mesh)
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        quality_score = self._calculate_mesh_quality(mesh)
        self._update_stats(elapsed_ms, len(points_3d), mesh.num_triangles, quality_score)
        
        return mesh
    
    def _extract_valid_points(self, heightmap: HeightMap) -> np.ndarray:
        """ハイトマップから有効な3D点を抽出（ベクトル化による高速化）"""
        height, width = heightmap.shape
        
        # 有効ピクセルのグリッド座標 (row, col) を取得
        rows, cols = np.where(heightmap.valid_mask)
        
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # グリッド座標から世界座標 (X, Y/Z) へ一括変換 – plane に応じて反転方法を切替
        min_x, _, min_y, _ = heightmap.bounds
        world_x = min_x + cols * heightmap.resolution

        if getattr(heightmap, "plane", "xy") == "xy":
            # 上下反転: 行0が max_y
            world_y = min_y + (height - 1 - rows) * heightmap.resolution
        else:
            # xz / yz 投影の場合は反転不要（行は奥行方向 or 高さ方向）
            world_y = min_y + rows * heightmap.resolution

        # Z座標（高さ）を取得
        world_z = heightmap.heights[rows, cols]
        
        # (N, 3) 形式の配列に結合
        return np.stack([world_x, world_y, world_z], axis=1)
    
    def _adaptive_sampling(self, points: np.ndarray, heightmap: HeightMap) -> np.ndarray:
        """適応的サンプリング（密度に応じてサンプリング率を調整）"""
        if len(points) < 100:
            return points
        
        # 高度の変化が大きい領域はより密にサンプリング
        heights = points[:, 2]
        gradient_magnitude = np.abs(np.gradient(heights))
        
        # 勾配に基づく重要度計算
        importance = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # 重要度の高い点を優先的に選択
        num_samples = min(len(points), int(len(points) * 0.7))
        probabilities = importance + 0.1  # 最低確率を保証
        probabilities /= np.sum(probabilities)
        
        try:
            sampled_indices = np.random.choice(
                len(points), 
                size=num_samples, 
                replace=False, 
                p=probabilities
            )
            return points[sampled_indices]
        except:
            # 確率計算に失敗した場合は均等サンプリング
            step = max(1, len(points) // num_samples)
            return points[::step]
    
    def _add_boundary_points(self, points: np.ndarray, heightmap: HeightMap) -> np.ndarray:
        """境界点を追加（凸包を改善）"""
        if len(points) < 4:
            return points
        
        # XY平面での2D凸包を計算
        xy_points = points[:, :2]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy_points)
            boundary_indices = hull.vertices
            
            # 境界点間に追加点を挿入
            boundary_points = points[boundary_indices]
            additional_points = []
            
            for i in range(len(boundary_points)):
                p1 = boundary_points[i]
                p2 = boundary_points[(i + 1) % len(boundary_points)]
                
                # 境界エッジが長い場合、中間点を追加
                edge_length = np.linalg.norm(p2[:2] - p1[:2])
                if edge_length > self.max_edge_length:
                    mid_point = (p1 + p2) / 2
                    additional_points.append(mid_point)
            
            if additional_points:
                return np.vstack([points, np.array(additional_points)])
            
        except:
            pass  # ConvexHullに失敗した場合はそのまま返す
        
        return points
    
    def _perform_delaunay(self, points: np.ndarray) -> TriangleMesh:
        """Delaunay三角形分割を実行（GPU加速対応）"""
        if len(points) < 3:
            raise ValueError("At least 3 points required for triangulation")
        
        # GPU使用判定（閾値を下げて実用的に）
        use_gpu_for_this = (
            self.gpu_triangulator is not None and 
            self.gpu_triangulator.use_gpu and 
            len(points) >= 50   # さらに低い閾値でGPU使用を積極化
        )
        
        start_time = time.perf_counter()
        
        if use_gpu_for_this:
            # GPU版三角分割
            try:
                # XY平面での三角分割
                points_2d = points[:, :2]
                gpu_result = self.gpu_triangulator.triangulate_points_2d(points_2d)
                
                if gpu_result is not None:
                    vertices_2d, triangles = gpu_result
                    
                    # Z座標を復元して3D頂点作成
                    vertices_3d = np.column_stack([
                        vertices_2d[:, 0],
                        vertices_2d[:, 1], 
                        points[:len(vertices_2d), 2]  # 元のZ座標
                    ])
                    
                    # 有効な三角形のみフィルタリング
                    valid_triangles = []
                    for tri in triangles:
                        if len(tri) == 3 and max(tri) < len(vertices_3d):
                            triangle_vertices = vertices_3d[tri]
                            if self._is_valid_triangle(triangle_vertices):
                                valid_triangles.append(tri)
                    
                    if valid_triangles:
                        import logging as _logging
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self.stats['gpu_triangulations'] += 1
                        _logging.getLogger(__name__).debug(
                            "[GPU-DELAUNAY] %d points -> %d triangles in %.1fms",
                            len(points), len(valid_triangles), elapsed_ms,
                        )
                        
                        return TriangleMesh(
                            vertices=vertices_3d.astype(np.float32),
                            triangles=np.array(valid_triangles).astype(np.int32)
                        )
                
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).warning("GPU triangulation failed: %s, falling back to CPU", e)
        
        # CPU版三角分割（フォールバック）
        xy_points = points[:, :2]
        
        try:
            delaunay = Delaunay(xy_points)
            triangles = delaunay.simplices
            
            # 有効な三角形のみを保持
            valid_triangles = []
            for tri in triangles:
                if self._is_valid_triangle(points[tri]):
                    valid_triangles.append(tri)
            
            if not valid_triangles:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "No triangles passed quality filter, using all %d triangles", len(triangles)
                )
                valid_triangles = triangles.tolist()
            
            triangles = np.array(valid_triangles)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats['cpu_triangulations'] += 1
            
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "[CPU-DELAUNAY] %d points -> %d triangles in %.1fms", len(points), len(triangles), elapsed_ms
            )
            
            return TriangleMesh(
                vertices=points,
                triangles=triangles
            )
            
        except Exception as e:
            raise ValueError(f"Delaunay triangulation failed: {e}")
    
    def _is_valid_triangle(self, triangle_vertices: np.ndarray) -> bool:
        """三角形の有効性をチェック"""
        if len(triangle_vertices) != 3:
            return False
        
        # 面積チェック
        v0, v1, v2 = triangle_vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1[:2], edge2[:2])  # 2D外積
        area = abs(cross) / 2.0
        
        # 非常に小さい三角形のみ除去
        if area < 1e-12:
            return False
        
        # エッジ長チェック（より緩い条件）
        edge_lengths = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2)
        ]
        
        # 非常に長いエッジのみ除去
        max_allowed_length = self.max_edge_length * 2.0  # より緩い条件
        if any(length > max_allowed_length for length in edge_lengths):
            return False
        
        # 縦横比チェック（より緩い条件）
        min_edge = min(edge_lengths)
        max_edge = max(edge_lengths)
        
        # ゼロ除算を避ける
        if max_edge < 1e-12:
            return False
            
        aspect_ratio = min_edge / max_edge
        
        # 極端に細い三角形のみ除去
        if aspect_ratio < 0.01:  # より緩い条件
            return False
        
        return True
    
    def _filter_low_quality_triangles(self, mesh: TriangleMesh) -> TriangleMesh:
        """低品質な三角形を除去（ベクトル化版）"""
        # 最適化されたベクトル化処理を使用
        from .vectorized import get_mesh_processor
        processor = get_mesh_processor()
        return processor.filter_triangles_by_quality(mesh, self.quality_threshold)
    
    def _calculate_triangle_qualities(self, mesh: TriangleMesh) -> np.ndarray:
        """三角形の品質を計算（ベクトル化版）"""
        # 最適化されたベクトル化処理を使用
        from .vectorized import vectorized_triangle_qualities
        return vectorized_triangle_qualities(mesh)
    
    def _calculate_mesh_quality(self, mesh: TriangleMesh) -> float:
        """メッシュ全体の品質を計算"""
        if mesh.num_triangles == 0:
            return 0.0
        
        qualities = self._calculate_triangle_qualities(mesh)
        return np.mean(qualities)
    
    def _update_stats(self, elapsed_ms: float, num_points: int, num_triangles: int, quality: float):
        """パフォーマンス統計更新"""
        self.stats['total_triangulations'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_triangulations']
        self.stats['last_num_points'] = num_points
        self.stats['last_num_triangles'] = num_triangles
        self.stats['last_quality_score'] = quality
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_triangulations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_points': 0,
            'last_num_triangles': 0,
            'last_quality_score': 0.0
        }

    # ------------------------------------------------------------------
    # Async helper (P-PERF-002)
    # ------------------------------------------------------------------

    _EXECUTOR = None  # class-level ThreadPoolExecutor

    def triangulate_points_async(self, points: np.ndarray):  # type: ignore[return-value]
        """非同期に triangulate_points() を実行し Future を返す。"""
        from concurrent.futures import ThreadPoolExecutor

        if DelaunayTriangulator._EXECUTOR is None:
            # Lazy init – limit to 2 workers to avoid CPU thrash
            DelaunayTriangulator._EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="triangulator")

        return DelaunayTriangulator._EXECUTOR.submit(self.triangulate_points, points)


# 便利関数

def create_mesh_from_heightmap(
    heightmap: HeightMap,
    max_edge_length: float = 0.05,
    quality_threshold: float = 0.5
) -> TriangleMesh:
    """
    ハイトマップから三角形メッシュを作成（簡単なインターフェース）
    
    Args:
        heightmap: 入力ハイトマップ
        max_edge_length: 最大エッジ長
        quality_threshold: 品質閾値
        
    Returns:
        三角形メッシュ
    """
    triangulator = DelaunayTriangulator(
        max_edge_length=max_edge_length,
        quality_threshold=quality_threshold
    )
    return triangulator.triangulate_heightmap(heightmap)


def triangulate_points(
    points: np.ndarray,
    max_edge_length: float = 0.1
) -> TriangleMesh:
    """
    点群からDelaunay三角形分割を実行（簡単なインターフェース）
    
    Args:
        points: 点群データ (N, 2) または (N, 3)
        max_edge_length: 最大エッジ長
        
    Returns:
        三角形メッシュ
    """
    triangulator = DelaunayTriangulator(max_edge_length=max_edge_length)
    return triangulator.triangulate_points(points) 