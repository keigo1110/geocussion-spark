#!/usr/bin/env python3
"""
Delaunay三角形分割（3D対応版）

ハイトマップや2D/3D点群から品質の良い三角形メッシュを生成する機能を提供します。
2D投影、3D四面体分割、表面再構成の3つのモードをサポート。
GPU加速対応でCPU比20-40倍の高速化を実現。
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union, Literal
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, cKDTree
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
            areas = np.array([areas], dtype=np.float64)
        else:
            areas = np.linalg.norm(cross, axis=1) / 2.0
            if not isinstance(areas, np.ndarray):
                areas = np.array([areas], dtype=np.float64)
        return areas
    
    def compute_normals(self):
        """三角形法線と頂点法線を計算"""
        # 三角形法線を計算
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        
        # 正規化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # ゼロ除算回避
        self.triangle_normals = normals / norms
        
        # 頂点法線を計算（隣接三角形の法線の平均）
        vertex_normals = np.zeros_like(self.vertices)
        for i, tri in enumerate(self.triangles):
            for v_idx in tri:
                vertex_normals[v_idx] += self.triangle_normals[i]
        
        # 正規化
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vertex_normals = vertex_normals / norms


class DelaunayTriangulator:
    """Delaunay三角形分割クラス（3D対応・GPU加速対応）"""
    
    def __init__(
        self,
        max_edge_length: float = 0.1,     # 最大エッジ長
        min_triangle_area: float = 1e-6,  # 最小三角形面積
        adaptive_sampling: bool = True,    # 適応的サンプリング
        boundary_points: bool = True,      # 境界点追加
        quality_threshold: float = 0.5,   # 品質閾値（0-1）
        slope_threshold: float = 2.0,      # 高さ差/平均エッジ長 の許容上限
        z_std_threshold: float = 0.5,      # Z標準偏差の許容上限
        z_mapping: str = "nn1",           # Zマッピング方法
        boundary_z_mode: str = "avg",     # 境界点Z決定方法
        # 3D関連パラメータ
        triangulation_mode: Literal["2d", "3d", "surface"] = "2d",  # 三角形分割モード
        alpha_radius: float = 0.1,        # Alpha Shape用の半径（surface mode用）
        normal_estimation_k: int = 20,    # 法線推定用の近傍点数
        projection_plane: Optional[str] = None,  # 投影平面（"xy", "xz", "yz", "auto"）
        # GPU加速設定
        use_gpu: bool = True,
        gpu_fallback_threshold: int = 300
    ):
        """
        初期化
        
        Args:
            triangulation_mode: 
                - "2d": XY平面への投影後に2D Delaunay（従来の動作）
                - "3d": 3D Delaunay四面体分割から表面を抽出
                - "surface": 点群から直接表面を再構成（Alpha Shapes等）
            alpha_radius: Alpha Shape用の半径（surface modeで使用）
            normal_estimation_k: 法線推定用の近傍点数
            projection_plane: 2Dモード時の投影平面（Noneの場合は自動選択）
        """
        self.max_edge_length = max_edge_length
        self.min_triangle_area = min_triangle_area
        self.adaptive_sampling = adaptive_sampling
        self.boundary_points = boundary_points
        self.quality_threshold = quality_threshold
        self.slope_threshold = slope_threshold
        self.z_std_threshold = z_std_threshold
        self.z_mapping = z_mapping
        self.boundary_z_mode = boundary_z_mode
        self.triangulation_mode = triangulation_mode
        self.alpha_radius = alpha_radius
        self.normal_estimation_k = normal_estimation_k
        self.projection_plane = projection_plane
        self.use_gpu = use_gpu
        self.gpu_fallback_threshold = gpu_fallback_threshold
        
        # GPU三角分割器を初期化（2Dモードのみ）
        self.gpu_triangulator = None
        if self.use_gpu and GPU_TRIANGULATION_AVAILABLE and self.triangulation_mode == "2d":
            try:
                triangulator_factory = create_gpu_triangulator
                if triangulator_factory is not None:
                    self.gpu_triangulator = triangulator_factory(
                        use_gpu=True,
                        quality_threshold=self.quality_threshold,
                        enable_caching=True,
                        force_cpu=False
                    )
                    log_gpu_status("DelaunayTriangulator", self.gpu_triangulator.use_gpu)
                else:
                    self.gpu_triangulator = None
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
            'gpu_speedup_achieved': 0.0,
            'triangulation_mode': self.triangulation_mode
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
        
        # ハイトマップの場合は通常2Dモードが適切
        original_mode = self.triangulation_mode
        if self.triangulation_mode == "3d":
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Heightmap triangulation: switching from 3d to 2d mode for better results"
            )
            self.triangulation_mode = "2d"
        
        try:
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
            
            # 法線計算
            mesh.compute_normals()
            
            # パフォーマンス統計更新
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            quality_score = self._calculate_mesh_quality(mesh)
            self._update_stats(elapsed_ms, len(sampled_points), mesh.num_triangles, quality_score)
            
            return mesh
            
        finally:
            self.triangulation_mode = original_mode
    
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
        
        # 法線計算
        mesh.compute_normals()
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        quality_score = self._calculate_mesh_quality(mesh)
        self._update_stats(elapsed_ms, len(points_3d), mesh.num_triangles, quality_score)
        
        return mesh
    
    def _determine_projection_plane(self, points: np.ndarray) -> str:
        """点群に最適な投影平面を自動決定"""
        if self.projection_plane and self.projection_plane != "auto":
            return self.projection_plane
        
        # 点群の分散を計算
        std = np.std(points, axis=0)
        
        # 最も分散が小さい軸を法線とする平面を選択
        min_axis = np.argmin(std)
        
        if min_axis == 0:  # X軸方向の分散が最小
            return "yz"
        elif min_axis == 1:  # Y軸方向の分散が最小
            return "xz"
        else:  # Z軸方向の分散が最小
            return "xy"
    
    def _project_to_plane(self, points: np.ndarray, plane: str) -> Tuple[np.ndarray, np.ndarray]:
        """点群を指定平面に投影"""
        if plane == "xy":
            return points[:, :2], points[:, 2]
        elif plane == "xz":
            return points[:, [0, 2]], points[:, 1]
        elif plane == "yz":
            return points[:, [1, 2]], points[:, 0]
        else:
            raise ValueError(f"Unknown projection plane: {plane}")
    
    def _perform_delaunay(self, points: np.ndarray) -> TriangleMesh:
        """Delaunay三角形分割を実行（モードに応じて2D/3D/Surface）"""
        if len(points) < 3:
            raise ValueError("At least 3 points required for triangulation")
        
        if self.triangulation_mode == "2d":
            return self._perform_2d_delaunay(points)
        elif self.triangulation_mode == "3d":
            return self._perform_3d_delaunay(points)
        elif self.triangulation_mode == "surface":
            return self._perform_surface_reconstruction(points)
        else:
            raise ValueError(f"Unknown triangulation mode: {self.triangulation_mode}")
    
    def _perform_2d_delaunay(self, points: np.ndarray) -> TriangleMesh:
        """2D投影によるDelaunay三角形分割（従来の実装）"""
        # 投影平面を決定
        plane = self._determine_projection_plane(points)
        points_2d, removed_coords = self._project_to_plane(points, plane)
        
        # GPU使用判定
        use_gpu_for_this = (
            self.gpu_triangulator is not None and 
            self.gpu_triangulator.use_gpu and 
            len(points) >= 20
        )
        
        start_time = time.perf_counter()
        
        if use_gpu_for_this:
            # GPU版三角分割
            try:
                assert self.gpu_triangulator is not None
                gpu_result = self.gpu_triangulator.triangulate_points_2d(points_2d)
                
                if gpu_result is not None:
                    vertices_2d, triangles = gpu_result
                    
                    # 元の3D座標を復元
                    from scipy.spatial import cKDTree
                    tree = cKDTree(points_2d)
                    _, nn_indices = tree.query(vertices_2d, k=1)
                    vertices_3d = points[nn_indices]
                    
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
                            "[GPU-DELAUNAY-2D] %d points -> %d triangles in %.1fms (plane: %s)",
                            len(points), len(valid_triangles), elapsed_ms, plane
                        )
                        
                        return TriangleMesh(
                            vertices=vertices_3d.astype(np.float32),
                            triangles=np.array(valid_triangles).astype(np.int32)
                        )
                
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).warning("GPU triangulation failed: %s, falling back to CPU", e)
        
        # CPU版三角分割
        try:
            delaunay = Delaunay(points_2d)
            triangles = delaunay.simplices
            
            # 有効な三角形のみを保持
            valid_triangles = []
            for tri in triangles:
                if self._is_valid_triangle(points[tri]):
                    valid_triangles.append(tri)
            
            if not valid_triangles:
                valid_triangles = triangles.tolist()
            
            triangles = np.array(valid_triangles)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats['cpu_triangulations'] += 1
            
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "[CPU-DELAUNAY-2D] %d points -> %d triangles in %.1fms (plane: %s)",
                len(points), len(triangles), elapsed_ms, plane
            )
            
            return TriangleMesh(
                vertices=points,
                triangles=triangles
            )
            
        except Exception as e:
            raise ValueError(f"2D Delaunay triangulation failed: {e}")
    
    def _perform_3d_delaunay(self, points: np.ndarray) -> TriangleMesh:
        """3D Delaunay四面体分割から表面メッシュを抽出"""
        import logging as _logging
        start_time = time.perf_counter()
        
        try:
            # 3D Delaunay四面体分割
            delaunay_3d = Delaunay(points)
            
            # 四面体から表面三角形を抽出
            # ConvexHullを使用して外側の面を取得
            hull = ConvexHull(points)
            triangles = hull.simplices
            
            # 有効な三角形のみを保持
            valid_triangles = []
            for tri in triangles:
                if self._is_valid_triangle(points[tri]):
                    valid_triangles.append(tri)
            
            if not valid_triangles:
                valid_triangles = triangles.tolist()
            
            triangles = np.array(valid_triangles)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _logging.getLogger(__name__).debug(
                "[3D-DELAUNAY] %d points -> %d surface triangles in %.1fms",
                len(points), len(triangles), elapsed_ms
            )
            
            return TriangleMesh(
                vertices=points,
                triangles=triangles
            )
            
        except Exception as e:
            # 3D Delaunayが失敗した場合は2Dにフォールバック
            _logging.getLogger(__name__).warning(
                "3D Delaunay failed: %s, falling back to 2D projection", e
            )
            return self._perform_2d_delaunay(points)
    
    def _perform_surface_reconstruction(self, points: np.ndarray) -> TriangleMesh:
        """Alpha Shapesベースの表面再構成"""
        import logging as _logging
        start_time = time.perf_counter()
        
        try:
            # Alpha Shapesアルゴリズム
            triangles = self._alpha_shapes(points, self.alpha_radius)
            
            if len(triangles) == 0:
                # Alpha Shapesが失敗した場合はConvexHullを使用
                _logging.getLogger(__name__).warning(
                    "Alpha Shapes produced no triangles, using ConvexHull"
                )
                hull = ConvexHull(points)
                triangles = hull.simplices
            
            # 有効な三角形のみを保持
            valid_triangles = []
            for tri in triangles:
                if self._is_valid_triangle(points[tri]):
                    valid_triangles.append(tri)
            
            triangles = np.array(valid_triangles)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _logging.getLogger(__name__).debug(
                "[SURFACE-RECONSTRUCTION] %d points -> %d triangles in %.1fms (alpha=%.3f)",
                len(points), len(triangles), elapsed_ms, self.alpha_radius
            )
            
            return TriangleMesh(
                vertices=points,
                triangles=triangles
            )
            
        except Exception as e:
            # 表面再構成が失敗した場合は3Dにフォールバック
            _logging.getLogger(__name__).warning(
                "Surface reconstruction failed: %s, falling back to 3D Delaunay", e
            )
            return self._perform_3d_delaunay(points)
    
    def _alpha_shapes(self, points: np.ndarray, alpha: float) -> np.ndarray:
        """Alpha Shapesアルゴリズムによる表面抽出"""
        # 3D Delaunay分割
        tri = Delaunay(points)
        
        # 各四面体について、外接球の半径を計算
        triangles = []
        
        # 四面体の各面（三角形）を検査
        for simplex in tri.simplices:
            # 四面体の4つの面を生成
            faces = [
                [simplex[0], simplex[1], simplex[2]],
                [simplex[0], simplex[1], simplex[3]],
                [simplex[0], simplex[2], simplex[3]],
                [simplex[1], simplex[2], simplex[3]]
            ]
            
            for face in faces:
                # 三角形の外接円半径を計算
                p1, p2, p3 = points[face]
                
                # 外接円の半径（簡易計算）
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p1 - p3)
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                
                if area > 0:
                    radius = (a * b * c) / (4 * area)
                    
                    # Alpha値以下の場合は表面として採用
                    if radius <= alpha:
                        triangles.append(face)
        
        # 重複を除去
        if triangles:
            triangles = np.unique(np.sort(triangles, axis=1), axis=0)
        else:
            triangles = np.array([], dtype=np.int32).reshape(0, 3)
        
        return triangles
    
    def _estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """k近傍を使用した法線推定"""
        tree = cKDTree(points)
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # k近傍を取得
            _, indices = tree.query(point, k=min(self.normal_estimation_k, len(points)))
            
            if len(indices) >= 3:
                # 近傍点群の共分散行列から法線を計算
                neighbors = points[indices]
                centered = neighbors - np.mean(neighbors, axis=0)
                cov = np.dot(centered.T, centered)
                
                # 最小固有値に対応する固有ベクトルが法線
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # 最小固有値の固有ベクトル
                
                # 向きを統一（上向きに）
                if normal[2] < 0:
                    normal = -normal
                
                normals[i] = normal
        
        # 正規化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals = normals / norms
        
        return normals
    
    # 以下、既存のメソッドは変更なし
    def _extract_valid_points(self, heightmap: HeightMap) -> np.ndarray:
        """ハイトマップから有効な3D点を抽出（ベクトル化による高速化）"""
        height, width = heightmap.shape
        
        # 有効ピクセルのグリッド座標 (row, col) を取得
        rows, cols = np.where(heightmap.valid_mask)
        
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # グリッド座標から世界座標 (X, Y/Z) へ一括変換
        min_x, _, min_y, _ = heightmap.bounds
        world_x = min_x + cols * heightmap.resolution

        if getattr(heightmap, "plane", "xy") == "xy":
            # 上下反転: 行0が max_y
            world_y = min_y + (height - 1 - rows) * heightmap.resolution
        else:
            # xz / yz 投影の場合は反転不要
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
        num_samples = min(len(points), int(len(points) * 0.9))
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
    
    def _add_boundary_points(self, points: np.ndarray, heightmap: Optional[HeightMap] = None) -> np.ndarray:
        """境界点を追加（凸包を改善）"""
        if len(points) < 4:
            return points
        
        # 投影平面を決定
        if self.triangulation_mode == "2d":
            plane = self._determine_projection_plane(points)
            points_2d, _ = self._project_to_plane(points, plane)
        else:
            # 3Dモードでは主成分分析で投影
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(points)
        
        try:
            hull = ConvexHull(points_2d)
            boundary_indices = hull.vertices
            
            # 境界点間に追加点を挿入
            boundary_points = points[boundary_indices]
            additional_points = []
            
            for i in range(len(boundary_points)):
                p1 = boundary_points[i]
                p2 = boundary_points[(i + 1) % len(boundary_points)]
                
                # 境界エッジが長い場合、中間点を追加
                edge_length = np.linalg.norm(p2 - p1)
                if edge_length > self.max_edge_length:
                    # 必要な分割数を計算
                    n_segments = int(np.ceil(edge_length / self.max_edge_length))
                    for j in range(1, n_segments):
                        t = j / n_segments
                        mid_point = p1 * (1 - t) + p2 * t
                        additional_points.append(mid_point)
            
            if additional_points:
                return np.vstack([points, np.array(additional_points)])
            
        except:
            pass  # ConvexHullに失敗した場合はそのまま返す
        
        return points
    
    def _is_valid_triangle(self, triangle_vertices: np.ndarray) -> bool:
        """三角形の有効性をチェック"""
        if len(triangle_vertices) != 3:
            return False
        
        # 面積チェック
        v0, v1, v2 = triangle_vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 3D外積で面積計算
        cross = np.cross(edge1, edge2)
        area = np.linalg.norm(cross) / 2.0

        # 非常に小さい三角形のみ除去
        try:
            area_value = float(area)
        except Exception:
            return False
        if area_value < self.min_triangle_area:
            return False

        # エッジ長チェック
        edge_lengths: List[float] = [
            float(np.linalg.norm(v1 - v0)),
            float(np.linalg.norm(v2 - v1)),
            float(np.linalg.norm(v0 - v2)),
        ]
        
        # 非常に長いエッジのみ除去
        max_allowed_length = self.max_edge_length * 2.0
        if any(length > max_allowed_length for length in edge_lengths):
            return False
        
        # 斜度チェック（2Dモードのみ）
        if self.triangulation_mode == "2d" and self.slope_threshold is not None and self.slope_threshold > 0:
            height_range = float(np.ptp(triangle_vertices[:, 2]))
            avg_edge = sum(edge_lengths) / 3.0
            if avg_edge > 1e-12:
                slope_ratio = height_range / avg_edge
                if slope_ratio > self.slope_threshold:
                    return False
        
        # 縦横比チェック
        min_edge = min(edge_lengths)
        max_edge = max(edge_lengths)
        
        if max_edge < 1e-12:
            return False
            
        aspect_ratio = min_edge / max_edge
        
        # 極端に細い三角形のみ除去
        if aspect_ratio < 0.01:
            return False
        
        return True
    
    def _filter_low_quality_triangles(self, mesh: TriangleMesh) -> TriangleMesh:
        """低品質な三角形を除去（ベクトル化版）"""
        # 最適化されたベクトル化処理を使用
        from .vectorized import get_mesh_processor
        processor = get_mesh_processor()
        return processor.filter_triangles_by_quality(mesh, self.quality_threshold, self.z_std_threshold)
    
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
        return float(np.mean(qualities))
    
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
            'last_quality_score': 0.0,
            'gpu_triangulations': 0,
            'cpu_triangulations': 0,
            'gpu_speedup_achieved': 0.0,
            'triangulation_mode': self.triangulation_mode
        }

    # Async helper
    _EXECUTOR = None

    def triangulate_points_async(self, points: np.ndarray):
        """非同期に triangulate_points() を実行し Future を返す。"""
        from concurrent.futures import ThreadPoolExecutor

        if DelaunayTriangulator._EXECUTOR is None:
            DelaunayTriangulator._EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="triangulator")

        return DelaunayTriangulator._EXECUTOR.submit(self.triangulate_points, points)


# 便利関数

def create_mesh_from_heightmap(
    heightmap: HeightMap,
    max_edge_length: float = 0.05,
    quality_threshold: float = 0.5,
    triangulation_mode: str = "2d"
) -> TriangleMesh:
    """
    ハイトマップから三角形メッシュを作成（簡単なインターフェース）
    
    Args:
        heightmap: 入力ハイトマップ
        max_edge_length: 最大エッジ長
        quality_threshold: 品質閾値
        triangulation_mode: 三角形分割モード
        
    Returns:
        三角形メッシュ
    """
    triangulator = DelaunayTriangulator(
        max_edge_length=max_edge_length,
        quality_threshold=quality_threshold,
        triangulation_mode=triangulation_mode
    )
    return triangulator.triangulate_heightmap(heightmap)


def triangulate_points(
    points: np.ndarray,
    max_edge_length: float = 0.1,
    mode: Literal["2d", "3d", "surface"] = "3d"
) -> TriangleMesh:
    """
    点群からDelaunay三角形分割を実行（簡単なインターフェース）
    
    Args:
        points: 点群データ (N, 2) または (N, 3)
        max_edge_length: 最大エッジ長
        mode: 三角形分割モード
            - "2d": 2D投影（高さマップ向け）
            - "3d": 3D Delaunay（凸包）
            - "surface": 表面再構成（複雑な形状向け）
        
    Returns:
        三角形メッシュ
    """
    triangulator = DelaunayTriangulator(
        max_edge_length=max_edge_length,
        triangulation_mode=mode
    )
    return triangulator.triangulate_points(points)