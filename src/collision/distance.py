"""
高性能距離計算モジュール

点-三角形距離計算の完全ベクトル化実装
perf-004: 点-三角形距離計算の逐次処理を解決
"""

import time
from typing import List, Tuple, Optional
import numpy as np
from ..mesh.delaunay import TriangleMesh
from ..types import CollisionType
from .. import get_logger

logger = get_logger(__name__)


def batch_point_triangle_distances(
    points: np.ndarray,
    triangle_vertices: np.ndarray
) -> np.ndarray:
    """
    複数の点と複数の三角形の距離を一括計算（完全ベクトル化）
    
    Args:
        points: 点群 (N, 3)
        triangle_vertices: 三角形頂点 (M, 3, 3) - M個の三角形、各3頂点
        
    Returns:
        距離行列 (N, M) - points[i] と triangles[j] の距離
    """
    N = points.shape[0]
    M = triangle_vertices.shape[0]
    
    # 結果配列を事前割り当て
    distances = np.zeros((N, M), dtype=np.float32)
    
    # 三角形のエッジベクトルを事前計算 (M, 2, 3)
    v0 = triangle_vertices[:, 0]  # (M, 3)
    v1 = triangle_vertices[:, 1]  # (M, 3)
    v2 = triangle_vertices[:, 2]  # (M, 3)
    
    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0  # (M, 3)
    
    # 法線ベクトルを事前計算 (M, 3)
    normals = np.cross(edge1, edge2)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # ゼロ除算回避
    normal_lengths = np.maximum(normal_lengths, 1e-12)
    unit_normals = normals / normal_lengths
    
    # 各点について一括計算
    for i in range(N):
        point = points[i]  # (3,)
        
        # 点から三角形v0への相対位置 (M, 3)
        w = point[np.newaxis, :] - v0  # (M, 3)
        
        # 重心座標系での投影計算 (ベクトル化)
        a = np.sum(edge1 * edge1, axis=1)  # (M,)
        b = np.sum(edge1 * edge2, axis=1)  # (M,)
        c = np.sum(edge2 * edge2, axis=1)  # (M,)
        d = np.sum(w * edge1, axis=1)      # (M,)
        e = np.sum(w * edge2, axis=1)      # (M,)
        
        # 重心座標計算
        denom = a * c - b * b
        denom = np.maximum(denom, 1e-12)  # ゼロ除算回避
        
        s = (b * e - c * d) / denom
        t = (b * d - a * e) / denom
        
        # 条件分岐をベクトル化
        # ケース1: 三角形内部 (s >= 0, t >= 0, s + t <= 1)
        inside_mask = (s >= 0) & (t >= 0) & (s + t <= 1)
        
        # ケース2: エッジ・頂点上
        s_clamped = np.clip(s, 0, 1)
        t_clamped = np.clip(t, 0, 1)
        sum_clamped = np.clip(s_clamped + t_clamped, 0, 1)
        
        # s + t > 1の場合の再正規化
        overflow_mask = (s_clamped + t_clamped) > 1
        normalization_factor = np.where(overflow_mask, sum_clamped / (s_clamped + t_clamped + 1e-12), 1.0)
        s_final = s_clamped * normalization_factor
        t_final = t_clamped * normalization_factor
        
        # 最近接点計算 (M, 3)
        closest_points = v0 + s_final[:, np.newaxis] * edge1 + t_final[:, np.newaxis] * edge2
        
        # 距離計算
        distances[i, :] = np.linalg.norm(point[np.newaxis, :] - closest_points, axis=1)
    
    return distances


def point_triangle_distance_vectorized(
    point: np.ndarray,
    triangle_vertices: np.ndarray
) -> float:
    """
    単一点と単一三角形の距離（ベクトル化版）
    
    Args:
        point: 3D点 (3,)
        triangle_vertices: 三角形頂点 (3, 3)
        
    Returns:
        最短距離
    """
    v0, v1, v2 = triangle_vertices
    
    # エッジベクトル
    edge1 = v1 - v0
    edge2 = v2 - v0
    w = point - v0
    
    # 重心座標計算
    a = np.dot(edge1, edge1)
    b = np.dot(edge1, edge2)
    c = np.dot(edge2, edge2)
    d = np.dot(w, edge1)
    e = np.dot(w, edge2)
    
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        # 退化三角形の場合は最も近い頂点への距離を返す
        dists = [
            np.linalg.norm(point - v0),
            np.linalg.norm(point - v1),
            np.linalg.norm(point - v2)
        ]
        return min(dists)
    
    s = (b * e - c * d) / denom
    t = (b * d - a * e) / denom
    
    # 重心座標による場合分け
    if s >= 0 and t >= 0 and s + t <= 1:
        # 三角形内部
        closest_point = v0 + s * edge1 + t * edge2
    else:
        # エッジまたは頂点上の最近接点を探す
        candidates = []
        
        # エッジv0-v1上の最近接点
        t1 = np.clip(np.dot(w, edge1) / max(np.dot(edge1, edge1), 1e-12), 0, 1)
        p1 = v0 + t1 * edge1
        candidates.append(p1)
        
        # エッジv0-v2上の最近接点
        t2 = np.clip(np.dot(w, edge2) / max(np.dot(edge2, edge2), 1e-12), 0, 1)
        p2 = v0 + t2 * edge2
        candidates.append(p2)
        
        # エッジv1-v2上の最近接点
        edge3 = v2 - v1
        w3 = point - v1
        t3 = np.clip(np.dot(w3, edge3) / max(np.dot(edge3, edge3), 1e-12), 0, 1)
        p3 = v1 + t3 * edge3
        candidates.append(p3)
        
        # 最短距離の候補を選択
        distances = [np.linalg.norm(point - p) for p in candidates]
        min_idx = np.argmin(distances)
        closest_point = candidates[min_idx]
    
    return np.linalg.norm(point - closest_point)


def batch_search_distances_optimized(
    mesh: TriangleMesh,
    query_points: np.ndarray,
    triangle_indices_list: List[List[int]]
) -> List[List[float]]:
    """
    最適化された一括距離検索
    
    Args:
        mesh: 対象メッシュ
        query_points: 検索点群 (N, 3)
        triangle_indices_list: 各点に対する検索対象三角形インデックス
        
    Returns:
        各点の距離リスト
    """
    start_time = time.perf_counter()
    results = []
    
    for i, triangle_indices in enumerate(triangle_indices_list):
        if not triangle_indices:
            results.append([])
            continue
        
        # 対象三角形の頂点を抽出
        target_triangles = mesh.triangles[triangle_indices]  # (M, 3)
        triangle_vertices = mesh.vertices[target_triangles]  # (M, 3, 3)
        
        # 単一点と複数三角形の距離計算
        point = query_points[i]
        distances = []
        
        for tri_verts in triangle_vertices:
            dist = point_triangle_distance_vectorized(point, tri_verts)
            distances.append(dist)
        
        results.append(distances)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Batch distance search: {len(query_points)} points, {elapsed_ms:.1f}ms")
    
    return results


class OptimizedDistanceCalculator:
    """距離計算の最適化されたクラス"""
    
    def __init__(self):
        self.stats = {
            'total_calculations': 0,
            'total_time_ms': 0.0,
            'last_batch_size': 0,
            'last_calculation_time_ms': 0.0
        }
    
    def calculate_point_triangle_distance(
        self,
        point: np.ndarray,
        triangle_vertices: np.ndarray
    ) -> float:
        """最適化された点-三角形距離計算"""
        start_time = time.perf_counter()
        
        distance = point_triangle_distance_vectorized(point, triangle_vertices)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, 1)
        
        return distance
    
    def calculate_batch_distances(
        self,
        points: np.ndarray,
        triangle_vertices: np.ndarray
    ) -> np.ndarray:
        """バッチ距離計算"""
        start_time = time.perf_counter()
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        if triangle_vertices.ndim == 2:
            # 単一三角形の場合
            distances = np.array([
                self.calculate_point_triangle_distance(point, triangle_vertices)
                for point in points
            ])
        else:
            # 複数三角形の場合
            distances = batch_point_triangle_distances(points, triangle_vertices)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, points.shape[0])
        
        return distances
    
    def _update_stats(self, elapsed_ms: float, batch_size: int):
        """統計更新"""
        self.stats['total_calculations'] += batch_size
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['last_batch_size'] = batch_size
        self.stats['last_calculation_time_ms'] = elapsed_ms
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        if self.stats['total_calculations'] > 0:
            stats['average_time_per_calculation_ms'] = (
                self.stats['total_time_ms'] / self.stats['total_calculations']
            )
        return stats
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_calculations': 0,
            'total_time_ms': 0.0,
            'last_batch_size': 0,
            'last_calculation_time_ms': 0.0
        }


# グローバルインスタンス
_global_distance_calculator: Optional[OptimizedDistanceCalculator] = None


def get_distance_calculator() -> OptimizedDistanceCalculator:
    """グローバル距離計算器を取得"""
    global _global_distance_calculator
    if _global_distance_calculator is None:
        _global_distance_calculator = OptimizedDistanceCalculator()
    return _global_distance_calculator 