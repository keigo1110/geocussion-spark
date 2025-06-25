"""
ベクトル化メッシュ演算モジュール

perf-003: Python ループによるメッシュ演算のパフォーマンス低下を解決
NumPy完全ベクトル化による高速メッシュ処理
"""

import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .delaunay import TriangleMesh
from .. import get_logger

logger = get_logger(__name__)


def vectorized_triangle_qualities(mesh: TriangleMesh) -> np.ndarray:
    """
    三角形品質を完全ベクトル化で計算
    
    Args:
        mesh: 入力メッシュ
        
    Returns:
        品質配列 (M,) - 各三角形の品質 (0-1, 1が最高)
    """
    if mesh.num_triangles == 0:
        return np.array([])
    
    # 全三角形の頂点を一括取得 (M, 3, 3)
    triangle_vertices = mesh.vertices[mesh.triangles]
    
    v0 = triangle_vertices[:, 0]  # (M, 3)
    v1 = triangle_vertices[:, 1]  # (M, 3)
    v2 = triangle_vertices[:, 2]  # (M, 3)
    
    # エッジベクトルを一括計算
    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0  # (M, 3)
    edge3 = v2 - v1  # (M, 3)
    
    # エッジ長を一括計算
    edge_lengths = np.array([
        np.linalg.norm(edge1, axis=1),  # (M,)
        np.linalg.norm(edge2, axis=1),  # (M,)
        np.linalg.norm(edge3, axis=1)   # (M,)
    ]).T  # (M, 3)
    
    # 面積を一括計算（外積の大きさ）
    cross_products = np.cross(edge1, edge2)  # (M, 3)
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)  # (M,)
    
    # 品質指標を一括計算（正規化された面積対周囲長比）
    # 品質 = 4 * sqrt(3) * area / (a² + b² + c²)
    edge_length_squares = np.sum(edge_lengths**2, axis=1)  # (M,)
    
    # ゼロ除算回避
    safe_denominator = np.maximum(edge_length_squares, 1e-12)
    qualities = (4 * np.sqrt(3) * areas) / safe_denominator
    
    # 0-1の範囲にクランプ
    qualities = np.clip(qualities, 0.0, 1.0)
    
    return qualities


def vectorized_is_valid_triangles(
    triangle_vertices: np.ndarray,
    max_edge_length: float = 0.2,
    min_area_threshold: float = 1e-12,
    min_aspect_ratio: float = 0.01
) -> np.ndarray:
    """
    複数三角形の有効性を一括判定（完全ベクトル化）
    
    Args:
        triangle_vertices: 三角形頂点配列 (M, 3, 3)
        max_edge_length: 最大エッジ長
        min_area_threshold: 最小面積閾値
        min_aspect_ratio: 最小縦横比
        
    Returns:
        有効性マスク (M,) - True=有効, False=無効
    """
    M = triangle_vertices.shape[0]
    if M == 0:
        return np.array([], dtype=bool)
    
    v0 = triangle_vertices[:, 0]  # (M, 3)
    v1 = triangle_vertices[:, 1]  # (M, 3)
    v2 = triangle_vertices[:, 2]  # (M, 3)
    
    # エッジベクトルと長さ
    edge1 = v1 - v0
    edge2 = v2 - v0
    edge3 = v2 - v1
    
    edge_lengths = np.array([
        np.linalg.norm(edge1, axis=1),
        np.linalg.norm(edge2, axis=1),
        np.linalg.norm(edge3, axis=1)
    ]).T  # (M, 3)
    
    # 面積計算
    cross_products = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)  # (M,)
    
    # 有効性チェック
    # 1. 面積チェック
    area_valid = areas >= min_area_threshold
    
    # 2. エッジ長チェック
    max_edge_per_triangle = np.max(edge_lengths, axis=1)  # (M,)
    edge_valid = max_edge_per_triangle <= (max_edge_length * 2.0)  # 緩い条件
    
    # 3. 縦横比チェック
    min_edge_per_triangle = np.min(edge_lengths, axis=1)  # (M,)
    max_edge_per_triangle = np.maximum(max_edge_per_triangle, 1e-12)  # ゼロ除算回避
    aspect_ratios = min_edge_per_triangle / max_edge_per_triangle
    aspect_valid = aspect_ratios >= min_aspect_ratio
    
    # 全条件を満たす三角形のみ有効
    valid_mask = area_valid & edge_valid & aspect_valid
    
    return valid_mask


def vectorized_triangle_areas(mesh: TriangleMesh) -> np.ndarray:
    """三角形面積を一括計算"""
    if mesh.num_triangles == 0:
        return np.array([])
    
    triangle_vertices = mesh.vertices[mesh.triangles]
    v0 = triangle_vertices[:, 0]
    v1 = triangle_vertices[:, 1]
    v2 = triangle_vertices[:, 2]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_products = np.cross(edge1, edge2)
    
    if cross_products.ndim == 1:
        # 2D case
        areas = np.abs(cross_products) / 2.0
    else:
        # 3D case
        areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    
    return areas


def vectorized_triangle_normals(mesh: TriangleMesh) -> np.ndarray:
    """三角形法線を一括計算"""
    if mesh.num_triangles == 0:
        return np.array([]).reshape(0, 3)
    
    triangle_vertices = mesh.vertices[mesh.triangles]
    v0 = triangle_vertices[:, 0]
    v1 = triangle_vertices[:, 1]
    v2 = triangle_vertices[:, 2]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    
    # 正規化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # ゼロ除算回避
    unit_normals = normals / norms
    
    return unit_normals


def vectorized_vertex_normals(mesh: TriangleMesh, smooth: bool = True) -> np.ndarray:
    """頂点法線を一括計算"""
    if mesh.num_vertices == 0:
        return np.array([]).reshape(0, 3)
    
    # 面法線を計算
    triangle_normals = vectorized_triangle_normals(mesh)
    
    if not smooth:
        # 平滑化なしの場合は各頂点に最初に関連する三角形の法線を使用
        vertex_normals = np.zeros((mesh.num_vertices, 3))
        for i, triangle in enumerate(mesh.triangles):
            for vertex_idx in triangle:
                if np.allclose(vertex_normals[vertex_idx], 0):
                    vertex_normals[vertex_idx] = triangle_normals[i]
        return vertex_normals
    
    # 面積重み付き平滑化
    triangle_areas = vectorized_triangle_areas(mesh)
    vertex_normals = np.zeros((mesh.num_vertices, 3))
    vertex_weights = np.zeros(mesh.num_vertices)
    
    # 各三角形の寄与を累積
    for i, triangle in enumerate(mesh.triangles):
        area = triangle_areas[i]
        normal = triangle_normals[i]
        
        for vertex_idx in triangle:
            vertex_normals[vertex_idx] += area * normal
            vertex_weights[vertex_idx] += area
    
    # 正規化
    valid_vertices = vertex_weights > 1e-12
    vertex_normals[valid_vertices] /= vertex_weights[valid_vertices, np.newaxis]
    
    # 単位ベクトル化
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vertex_normals = vertex_normals / norms
    
    return vertex_normals


def vectorized_edge_lengths(mesh: TriangleMesh) -> np.ndarray:
    """全エッジ長を一括計算"""
    if mesh.num_triangles == 0:
        return np.array([])
    
    triangle_vertices = mesh.vertices[mesh.triangles]
    v0 = triangle_vertices[:, 0]
    v1 = triangle_vertices[:, 1]
    v2 = triangle_vertices[:, 2]
    
    # 各三角形の3つのエッジ長
    edges = np.array([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1)
    ]).T  # (M, 3)
    
    return edges.flatten()  # 全エッジ長の1D配列


class VectorizedMeshProcessor:
    """ベクトル化メッシュ処理クラス"""
    
    def __init__(self):
        self.stats = {
            'total_operations': 0,
            'total_time_ms': 0.0,
            'last_operation_time_ms': 0.0,
            'last_mesh_size': 0
        }
    
    def filter_triangles_by_quality(
        self,
        mesh: TriangleMesh,
        quality_threshold: float = 0.5
    ) -> TriangleMesh:
        """品質に基づく三角形フィルタリング（ベクトル化版）"""
        start_time = time.perf_counter()
        
        if mesh.num_triangles == 0:
            return mesh
        
        # 品質を一括計算
        qualities = vectorized_triangle_qualities(mesh)
        
        # 閾値以上の三角形を選択
        good_mask = qualities >= quality_threshold
        good_triangles = mesh.triangles[good_mask]
        
        if len(good_triangles) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(elapsed_ms, mesh.num_triangles)
            return mesh  # 全て除去された場合は元のメッシュを返す
        
        # 使用される頂点のみを保持
        used_vertices = np.unique(good_triangles.flatten())
        new_vertices = mesh.vertices[used_vertices]
        
        # 三角形インデックスを再マッピング
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        new_triangles = np.array([
            [vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]]
            for tri in good_triangles
        ])
        
        # 新しいメッシュを作成
        from .delaunay import TriangleMesh
        filtered_mesh = TriangleMesh(vertices=new_vertices, triangles=new_triangles)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, mesh.num_triangles)
        
        logger.debug(f"Quality filtering: {mesh.num_triangles} → {filtered_mesh.num_triangles} "
                    f"triangles, {elapsed_ms:.1f}ms")
        
        return filtered_mesh
    
    def validate_mesh_triangles(
        self,
        mesh: TriangleMesh,
        max_edge_length: float = 0.2
    ) -> np.ndarray:
        """メッシュ三角形の有効性を一括検証"""
        start_time = time.perf_counter()
        
        if mesh.num_triangles == 0:
            return np.array([], dtype=bool)
        
        triangle_vertices = mesh.vertices[mesh.triangles]
        valid_mask = vectorized_is_valid_triangles(
            triangle_vertices,
            max_edge_length=max_edge_length
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, mesh.num_triangles)
        
        return valid_mask
    
    def compute_mesh_statistics(self, mesh: TriangleMesh) -> Dict[str, Any]:
        """メッシュ統計を一括計算"""
        start_time = time.perf_counter()
        
        if mesh.num_triangles == 0:
            return {
                'num_vertices': mesh.num_vertices,
                'num_triangles': 0,
                'total_area': 0.0,
                'average_area': 0.0,
                'average_quality': 0.0,
                'edge_length_stats': {}
            }
        
        # 各種統計を一括計算
        areas = vectorized_triangle_areas(mesh)
        qualities = vectorized_triangle_qualities(mesh)
        edge_lengths = vectorized_edge_lengths(mesh)
        
        stats = {
            'num_vertices': mesh.num_vertices,
            'num_triangles': mesh.num_triangles,
            'total_area': np.sum(areas),
            'average_area': np.mean(areas),
            'area_std': np.std(areas),
            'average_quality': np.mean(qualities),
            'quality_std': np.std(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'edge_length_stats': {
                'min': np.min(edge_lengths),
                'max': np.max(edge_lengths),
                'mean': np.mean(edge_lengths),
                'std': np.std(edge_lengths)
            }
        }
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, mesh.num_triangles)
        
        return stats
    
    def _update_stats(self, elapsed_ms: float, mesh_size: int):
        """統計更新"""
        self.stats['total_operations'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['last_operation_time_ms'] = elapsed_ms
        self.stats['last_mesh_size'] = mesh_size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        if self.stats['total_operations'] > 0:
            stats['average_time_per_operation_ms'] = (
                self.stats['total_time_ms'] / self.stats['total_operations']
            )
        return stats
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_operations': 0,
            'total_time_ms': 0.0,
            'last_operation_time_ms': 0.0,
            'last_mesh_size': 0
        }


# グローバルインスタンス
_global_mesh_processor: Optional[VectorizedMeshProcessor] = None


def get_mesh_processor() -> VectorizedMeshProcessor:
    """グローバルメッシュプロセッサを取得"""
    global _global_mesh_processor
    if _global_mesh_processor is None:
        _global_mesh_processor = VectorizedMeshProcessor()
    return _global_mesh_processor 