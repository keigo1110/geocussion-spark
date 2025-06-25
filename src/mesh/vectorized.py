"""
ベクトル化メッシュ演算モジュール

perf-003: Python ループによるメッシュ演算のパフォーマンス低下を解決
NumPy完全ベクトル化による高速メッシュ処理
Numba JIT対応による超高速化
"""

import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Numba未利用時のダミーデコレータ
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .delaunay import TriangleMesh
from .. import get_logger

logger = get_logger(__name__)


@njit(cache=True, fastmath=True)
def _triangle_areas_jit(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """JIT最適化された三角形面積計算"""
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 外積を手動計算（Numbaが対応するため）
    cross_x = edge1[:, 1] * edge2[:, 2] - edge1[:, 2] * edge2[:, 1]
    cross_y = edge1[:, 2] * edge2[:, 0] - edge1[:, 0] * edge2[:, 2]
    cross_z = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    
    cross_magnitude = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    return 0.5 * cross_magnitude


@njit(cache=True, fastmath=True)
def _triangle_qualities_jit(
    v0: np.ndarray, 
    v1: np.ndarray, 
    v2: np.ndarray
) -> np.ndarray:
    """JIT最適化された三角形品質計算"""
    # エッジベクトル
    edge1 = v1 - v0
    edge2 = v2 - v0
    edge3 = v2 - v1
    
    # エッジ長の二乗を計算
    edge1_sq = np.sum(edge1**2, axis=1)
    edge2_sq = np.sum(edge2**2, axis=1)
    edge3_sq = np.sum(edge3**2, axis=1)
    
    # 面積計算（外積の大きさ）
    areas = _triangle_areas_jit(v0, v1, v2)
    
    # 品質指標計算: 4 * sqrt(3) * area / (a² + b² + c²)
    edge_length_squares = edge1_sq + edge2_sq + edge3_sq
    
    # ゼロ除算回避
    safe_denominator = np.maximum(edge_length_squares, 1e-12)
    qualities = (4 * np.sqrt(3) * areas) / safe_denominator
    
    # 0-1の範囲にクランプ（手動）
    for i in range(len(qualities)):
        if qualities[i] < 0.0:
            qualities[i] = 0.0
        elif qualities[i] > 1.0:
            qualities[i] = 1.0
    
    return qualities


@njit(cache=True, fastmath=True)
def _is_valid_triangles_jit(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """JIT最適化された三角形有効性検証"""
    # 面積チェック
    areas = _triangle_areas_jit(v0, v1, v2)
    area_valid = areas > 1e-12
    
    # エッジ長チェック
    edge1_len = np.sqrt(np.sum((v1 - v0)**2, axis=1))
    edge2_len = np.sqrt(np.sum((v2 - v0)**2, axis=1))
    edge3_len = np.sqrt(np.sum((v2 - v1)**2, axis=1))
    
    min_edge_len = 1e-10
    edge_valid = (edge1_len > min_edge_len) & (edge2_len > min_edge_len) & (edge3_len > min_edge_len)
    
    # アスペクト比チェック（極端に細長い三角形を除外）
    max_edge = np.maximum(np.maximum(edge1_len, edge2_len), edge3_len)
    min_edge = np.minimum(np.minimum(edge1_len, edge2_len), edge3_len)
    
    aspect_ratio_valid = (max_edge / np.maximum(min_edge, 1e-12)) < 1000.0
    
    return area_valid & edge_valid & aspect_ratio_valid


def vectorized_triangle_qualities(mesh: TriangleMesh) -> np.ndarray:
    """
    三角形品質を完全ベクトル化で計算（JIT対応）
    
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
    
    if NUMBA_AVAILABLE:
        # JIT最適化版を使用
        return _triangle_qualities_jit(v0, v1, v2)
    else:
        # フォールバック版（既存のNumPy実装）
        logger.warning("Numba not available, using NumPy fallback for triangle qualities")
        return _triangle_qualities_fallback(v0, v1, v2)


def _triangle_qualities_fallback(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Numba無効時のフォールバック実装"""
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
    return np.clip(qualities, 0.0, 1.0)


def vectorized_is_valid_triangles(triangle_vertices: np.ndarray) -> np.ndarray:
    """
    複数三角形の有効性をベクトル化検証（JIT対応）
    
    Args:
        triangle_vertices: 三角形頂点配列 (M, 3, 3)
        
    Returns:
        有効性マスク (M,) - True: 有効, False: 無効
    """
    if triangle_vertices.size == 0:
        return np.array([], dtype=bool)
    
    v0 = triangle_vertices[:, 0]  # (M, 3)
    v1 = triangle_vertices[:, 1]  # (M, 3)
    v2 = triangle_vertices[:, 2]  # (M, 3)
    
    if NUMBA_AVAILABLE:
        # JIT最適化版を使用
        return _is_valid_triangles_jit(v0, v1, v2)
    else:
        # フォールバック版
        logger.warning("Numba not available, using NumPy fallback for triangle validation")
        return _is_valid_triangles_fallback(v0, v1, v2)


def _is_valid_triangles_fallback(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Numba無効時のフォールバック実装"""
    # 面積チェック
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_products = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    area_valid = areas > 1e-12
    
    # エッジ長チェック
    edge_lengths = np.array([
        np.linalg.norm(edge1, axis=1),
        np.linalg.norm(edge2, axis=1),
        np.linalg.norm(v2 - v1, axis=1)
    ]).T
    
    min_edge_len = 1e-10
    edge_valid = np.all(edge_lengths > min_edge_len, axis=1)
    
    # アスペクト比チェック
    max_edge = np.max(edge_lengths, axis=1)
    min_edge = np.min(edge_lengths, axis=1)
    aspect_ratio_valid = (max_edge / np.maximum(min_edge, 1e-12)) < 1000.0
    
    return area_valid & edge_valid & aspect_ratio_valid


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
        valid_mask = vectorized_is_valid_triangles(triangle_vertices)
        
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


@njit(cache=True, fastmath=True, parallel=True)
def _mesh_statistics_jit(
    triangle_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT最適化: メッシュ統計の一括計算
    
    Args:
        triangle_vertices: 三角形頂点 (M, 3, 3)
        
    Returns:
        areas, qualities, edge_lengths
    """
    M = triangle_vertices.shape[0]
    
    v0 = triangle_vertices[:, 0]
    v1 = triangle_vertices[:, 1]
    v2 = triangle_vertices[:, 2]
    
    # 面積計算
    areas = _triangle_areas_jit(v0, v1, v2)
    
    # 品質計算
    qualities = _triangle_qualities_jit(v0, v1, v2)
    
    # エッジ長計算（全エッジ）
    edge_lengths = np.zeros(M * 3, dtype=np.float64)
    for i in range(M):
        # エッジ1: v0 -> v1
        edge_lengths[i * 3] = np.sqrt(np.sum((v1[i] - v0[i])**2))
        # エッジ2: v1 -> v2
        edge_lengths[i * 3 + 1] = np.sqrt(np.sum((v2[i] - v1[i])**2))
        # エッジ3: v2 -> v0
        edge_lengths[i * 3 + 2] = np.sqrt(np.sum((v0[i] - v2[i])**2))
    
    return areas, qualities, edge_lengths


@njit(cache=True, fastmath=True)
def _filter_by_quality_jit(
    qualities: np.ndarray,
    triangles: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT最適化: 品質による三角形フィルタリング
    
    Args:
        qualities: 品質配列 (M,)
        triangles: 三角形インデックス (M, 3)
        threshold: 品質閾値
        
    Returns:
        (good_indices, good_triangles)
    """
    # 良い三角形のインデックスを特定
    good_indices = []
    for i in range(len(qualities)):
        if qualities[i] >= threshold:
            good_indices.append(i)
    
    # NumPy配列に変換
    good_idx_array = np.array(good_indices, dtype=np.int32)
    
    # 対応する三角形を抽出
    good_triangles = np.zeros((len(good_indices), 3), dtype=np.int32)
    for i, idx in enumerate(good_indices):
        for j in range(3):
            good_triangles[i, j] = triangles[idx, j]
    
    return good_idx_array, good_triangles


def compute_mesh_statistics_fast(mesh: TriangleMesh) -> Dict[str, Any]:
    """
    JIT最適化されたメッシュ統計計算
    
    Args:
        mesh: 入力メッシュ
        
    Returns:
        統計辞書
    """
    if mesh.num_triangles == 0:
        return {
            'num_vertices': mesh.num_vertices,
            'num_triangles': 0,
            'total_area': 0.0,
            'average_area': 0.0,
            'average_quality': 0.0,
            'edge_length_stats': {}
        }
    
    triangle_vertices = mesh.vertices[mesh.triangles]
    
    if NUMBA_AVAILABLE:
        # JIT最適化版
        areas, qualities, edge_lengths = _mesh_statistics_jit(triangle_vertices)
    else:
        # フォールバック版
        areas = vectorized_triangle_areas(mesh)
        qualities = vectorized_triangle_qualities(mesh)
        edge_lengths = vectorized_edge_lengths(mesh)
    
    stats = {
        'num_vertices': mesh.num_vertices,
        'num_triangles': mesh.num_triangles,
        'total_area': float(np.sum(areas)),
        'average_area': float(np.mean(areas)),
        'area_std': float(np.std(areas)),
        'average_quality': float(np.mean(qualities)),
        'quality_std': float(np.std(qualities)),
        'min_quality': float(np.min(qualities)),
        'max_quality': float(np.max(qualities)),
        'edge_length_stats': {
            'min': float(np.min(edge_lengths)),
            'max': float(np.max(edge_lengths)),
            'mean': float(np.mean(edge_lengths)),
            'std': float(np.std(edge_lengths))
        }
    }
    
    return stats 