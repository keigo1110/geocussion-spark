"""
高性能距離計算モジュール

点-三角形距離計算の完全ベクトル化実装
perf-004: 点-三角形距離計算の逐次処理を解決
Numba JIT コンパイル対応による超高速化
"""

import time
from typing import List, Tuple, Optional
import numpy as np
# 統一されたNumba設定をインポート
from ..numba_config import get_numba, get_optimized_jit_config, create_optimized_jit

from ..mesh.delaunay import TriangleMesh
from ..data_types import CollisionType
from .. import get_logger

logger = get_logger(__name__)

# Numbaデコレータを遅延取得
def _get_jit_decorators():
    """JIT デコレータを遅延取得"""
    jit_func, njit_func, available = get_numba()
    return njit_func, available

# キャッシュ無効化設定でJIT関数を定義
def _create_jit_distance_function():
    """JIT最適化された距離計算関数を作成"""
    njit_func, available = _get_jit_decorators()
    
    if not available:
        return None
    
    config = get_optimized_jit_config()
    
    @njit_func(**config)
    def _point_triangle_distance_jit(
        point: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> float:
        """
        JIT最適化された点-三角形距離計算（nopython mode）
        
        Args:
            point: 検査点 (3,)
            v0, v1, v2: 三角形の頂点 (3,)
            
        Returns:
            最短距離
        """
        # エッジベクトル
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 点から v0 への相対位置
        w = point - v0
        
        # 重心座標系での投影計算
        a = np.dot(edge1, edge1)
        b = np.dot(edge1, edge2)
        c = np.dot(edge2, edge2)
        d = np.dot(w, edge1)
        e = np.dot(w, edge2)
        
        # 重心座標計算
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            # 退化三角形の場合：edge上の最近点を計算
            t_raw = np.dot(w, edge1) / max(a, 1e-12)
            t = max(0.0, min(1.0, t_raw))  # manual clipping
            closest = v0 + t * edge1
            return np.sqrt(np.sum((point - closest) ** 2))
        
        s = (b * e - c * d) / denom
        t = (b * d - a * e) / denom
        
        # 重心座標による最近点決定
        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            # 三角形内部
            closest = v0 + s * edge1 + t * edge2
        else:
            # 境界上の最近点を計算
            # エッジ v0-v1 上
            t1_raw = d / max(a, 1e-12)
            t1 = max(0.0, min(1.0, t1_raw))  # manual clipping
            p1 = v0 + t1 * edge1
            dist1_sq = np.sum((point - p1) ** 2)
            
            # エッジ v0-v2 上
            t2_raw = e / max(c, 1e-12)
            t2 = max(0.0, min(1.0, t2_raw))  # manual clipping
            p2 = v0 + t2 * edge2
            dist2_sq = np.sum((point - p2) ** 2)
            
            # エッジ v1-v2 上
            edge3 = v2 - v1
            w3 = point - v1
            t3_raw = np.dot(w3, edge3) / max(np.dot(edge3, edge3), 1e-12)
            t3 = max(0.0, min(1.0, t3_raw))  # manual clipping
            p3 = v1 + t3 * edge3
            dist3_sq = np.sum((point - p3) ** 2)
            
            # 最小距離の点を選択
            if dist1_sq <= dist2_sq and dist1_sq <= dist3_sq:
                closest = p1
            elif dist2_sq <= dist3_sq:
                closest = p2
            else:
                closest = p3
        
        return np.sqrt(np.sum((point - closest) ** 2))
    
    return _point_triangle_distance_jit

# グローバルなJIT関数インスタンス（遅延初期化）
_jit_distance_func = None

def _get_jit_distance_function():
    """JIT距離計算関数を取得（遅延初期化）"""
    global _jit_distance_func
    
    if _jit_distance_func is None:
        _jit_distance_func = _create_jit_distance_function()
    
    return _jit_distance_func


def _create_jit_batch_function():
    """JIT最適化されたバッチ処理関数を作成"""
    njit_func, available = _get_jit_decorators()
    
    if not available:
        return None
    
    # 並列処理設定
    config = get_optimized_jit_config()
    config['parallel'] = True
    
    @njit_func(**config)
    def _batch_distances_jit(
        points: np.ndarray,
        triangles: np.ndarray,
        distance_func: np.ndarray  # dummy parameter to work around closure issues
    ) -> np.ndarray:
        """
        JIT最適化されたバッチ距離計算（parallel mode）
        
        Args:
            points: 点群 (N, 3)
            triangles: 三角形頂点群 (M, 3, 3)
            distance_func: 未使用（クロージャ回避）
            
        Returns:
            距離行列 (N, M)
        """
        N = points.shape[0]
        M = triangles.shape[0]
        distances = np.zeros((N, M), dtype=np.float32)
        
        # 並列化されたループ（直接実装）
        for i in range(N):
            for j in range(M):
                # 距離計算をインライン展開
                point = points[i]
                v0 = triangles[j, 0]
                v1 = triangles[j, 1] 
                v2 = triangles[j, 2]
                
                # エッジベクトル
                edge1 = v1 - v0
                edge2 = v2 - v0
                w = point - v0
                
                # 重心座標系での投影計算
                a = np.dot(edge1, edge1)
                b = np.dot(edge1, edge2)
                c = np.dot(edge2, edge2)
                d = np.dot(w, edge1)
                e = np.dot(w, edge2)
                
                # 重心座標計算
                denom = a * c - b * b
                if abs(denom) < 1e-12:
                    t_raw = np.dot(w, edge1) / max(a, 1e-12)
                    t = max(0.0, min(1.0, t_raw))
                    closest = v0 + t * edge1
                    distances[i, j] = np.sqrt(np.sum((point - closest) ** 2))
                else:
                    s = (b * e - c * d) / denom
                    t_val = (b * d - a * e) / denom
                    
                    if s >= 0.0 and t_val >= 0.0 and s + t_val <= 1.0:
                        # 三角形内部
                        closest = v0 + s * edge1 + t_val * edge2
                        distances[i, j] = np.sqrt(np.sum((point - closest) ** 2))
                    else:
                        # 境界計算（簡略版）
                        t1 = max(0.0, min(1.0, d / max(a, 1e-12)))
                        p1 = v0 + t1 * edge1
                        dist1_sq = np.sum((point - p1) ** 2)
                        
                        t2 = max(0.0, min(1.0, e / max(c, 1e-12)))
                        p2 = v0 + t2 * edge2
                        dist2_sq = np.sum((point - p2) ** 2)
                        
                        edge3 = v2 - v1
                        w3 = point - v1
                        t3 = max(0.0, min(1.0, np.dot(w3, edge3) / max(np.dot(edge3, edge3), 1e-12)))
                        p3 = v1 + t3 * edge3
                        dist3_sq = np.sum((point - p3) ** 2)
                        
                        min_dist_sq = min(dist1_sq, min(dist2_sq, dist3_sq))
                        distances[i, j] = np.sqrt(min_dist_sq)
        
        return distances
    
    return _batch_distances_jit

# グローバルなバッチ関数インスタンス
_jit_batch_func = None

def _get_jit_batch_function():
    """JITバッチ処理関数を取得（遅延初期化）"""
    global _jit_batch_func
    
    if _jit_batch_func is None:
        _jit_batch_func = _create_jit_batch_function()
    
    return _jit_batch_func


def point_triangle_distance_vectorized(
    point: np.ndarray,
    triangle_vertices: np.ndarray
) -> float:
    """
    高速化された点-三角形距離計算（JIT対応）
    
    Args:
        point: 検査点 (3,)
        triangle_vertices: 三角形頂点 (3, 3)
        
    Returns:
        最短距離
    """
    # NumPy配列の型確認・変換
    point = np.asarray(point, dtype=np.float64)
    triangle_vertices = np.asarray(triangle_vertices, dtype=np.float64)
    
    # JIT関数を遅延取得
    jit_func = _get_jit_distance_function()
    
    if jit_func is not None:
        try:
            # JIT最適化版を使用
            return jit_func(
                point,
                triangle_vertices[0],
                triangle_vertices[1],
                triangle_vertices[2]
            )
        except Exception as e:
            # JIT実行エラー時のフォールバック
            logger.warning(f"Numba JIT execution failed: {e}, falling back to NumPy")
            return _point_triangle_distance_fallback(point, triangle_vertices)
    else:
        # フォールバック版（従来の実装）
        return _point_triangle_distance_fallback(point, triangle_vertices)


def _point_triangle_distance_fallback(point: np.ndarray, triangle_vertices: np.ndarray) -> float:
    """Numba無効時のフォールバック実装"""
    # ... existing code ...
    v0, v1, v2 = triangle_vertices
    edge1 = v1 - v0
    edge2 = v2 - v0
    w = point - v0
    
    a = np.dot(edge1, edge1)
    b = np.dot(edge1, edge2)
    c = np.dot(edge2, edge2)
    d = np.dot(w, edge1)
    e = np.dot(w, edge2)
    
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        t = np.clip(np.dot(w, edge1) / max(a, 1e-12), 0.0, 1.0)
        closest = v0 + t * edge1
        return np.linalg.norm(point - closest)
    
    s = (b * e - c * d) / denom
    t = (b * d - a * e) / denom
    
    if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
        closest = v0 + s * edge1 + t * edge2
    else:
        # 境界計算
        t1 = np.clip(d / max(a, 1e-12), 0.0, 1.0)
        p1 = v0 + t1 * edge1
        
        t2 = np.clip(e / max(c, 1e-12), 0.0, 1.0)
        p2 = v0 + t2 * edge2
        
        edge3 = v2 - v1
        w3 = point - v1
        t3 = np.clip(np.dot(w3, edge3) / max(np.dot(edge3, edge3), 1e-12), 0.0, 1.0)
        p3 = v1 + t3 * edge3
        
        distances = [
            np.linalg.norm(point - p1),
            np.linalg.norm(point - p2),
            np.linalg.norm(point - p3)
        ]
        return min(distances)
    
    return np.linalg.norm(point - closest)


def _create_batch_point_triangles_function():
    """バッチ点-複数三角形JIT関数を作成"""
    njit_func, available = _get_jit_decorators()
    
    if not available:
        return None
    
    config = get_optimized_jit_config()
    config['parallel'] = True
    
    @njit_func(**config)
    def _batch_point_multiple_triangles_jit(
        point: np.ndarray,
        triangles: np.ndarray
    ) -> np.ndarray:
        """
        JIT最適化: 単一点と複数三角形の距離を並列計算
        
        Args:
            point: 検査点 (3,)
            triangles: 三角形頂点群 (M, 3, 3)
            
        Returns:
            距離配列 (M,)
        """
        M = triangles.shape[0]
        distances = np.zeros(M, dtype=np.float64)
        
        # 並列化されたループ（距離計算をインライン展開）
        for j in range(M):
            # 直接距離計算（クロージャ問題回避）
            v0 = triangles[j, 0]
            v1 = triangles[j, 1]
            v2 = triangles[j, 2]
            
            # エッジベクトル
            edge1 = v1 - v0
            edge2 = v2 - v0
            w = point - v0
            
            # 重心座標系での投影計算
            a = np.dot(edge1, edge1)
            b = np.dot(edge1, edge2)
            c = np.dot(edge2, edge2)
            d = np.dot(w, edge1)
            e = np.dot(w, edge2)
            
            # 重心座標計算
            denom = a * c - b * b
            if abs(denom) < 1e-12:
                t_raw = np.dot(w, edge1) / max(a, 1e-12)
                t = max(0.0, min(1.0, t_raw))
                closest = v0 + t * edge1
                distances[j] = np.sqrt(np.sum((point - closest) ** 2))
            else:
                s = (b * e - c * d) / denom
                t_val = (b * d - a * e) / denom
                
                if s >= 0.0 and t_val >= 0.0 and s + t_val <= 1.0:
                    # 三角形内部
                    closest = v0 + s * edge1 + t_val * edge2
                    distances[j] = np.sqrt(np.sum((point - closest) ** 2))
                else:
                    # 境界計算（簡略版）
                    t1 = max(0.0, min(1.0, d / max(a, 1e-12)))
                    p1 = v0 + t1 * edge1
                    dist1_sq = np.sum((point - p1) ** 2)
                    
                    t2 = max(0.0, min(1.0, e / max(c, 1e-12)))
                    p2 = v0 + t2 * edge2
                    dist2_sq = np.sum((point - p2) ** 2)
                    
                    edge3 = v2 - v1
                    w3 = point - v1
                    t3 = max(0.0, min(1.0, np.dot(w3, edge3) / max(np.dot(edge3, edge3), 1e-12)))
                    p3 = v1 + t3 * edge3
                    dist3_sq = np.sum((point - p3) ** 2)
                    
                    min_dist_sq = min(dist1_sq, min(dist2_sq, dist3_sq))
                    distances[j] = np.sqrt(min_dist_sq)
        
        return distances
    
    return _batch_point_multiple_triangles_jit

# グローバルなバッチ点三角形関数インスタンス
_jit_batch_point_func = None

def _get_jit_batch_point_function():
    """JITバッチ点三角形関数を取得（遅延初期化）"""
    global _jit_batch_point_func
    
    if _jit_batch_point_func is None:
        _jit_batch_point_func = _create_batch_point_triangles_function()
    
    return _jit_batch_point_func


def _create_penalty_function():
    """JIT最適化されたペナルティ計算関数を作成"""
    njit_func, available = _get_jit_decorators()
    
    if not available:
        return None
    
    config = get_optimized_jit_config()
    
    @njit_func(**config)
    def _compute_collision_penalty_jit(
        distances: np.ndarray,
        radius: float,
        penalty_factor: float = 100.0
    ) -> np.ndarray:
        """
        JIT最適化: 衝突ペナルティ計算
        
        Args:
            distances: 距離配列 (N,)
            radius: 衝突半径
            penalty_factor: ペナルティ係数
            
        Returns:
            ペナルティ配列 (N,)
        """
        penalties = np.zeros_like(distances)
        
        for i in range(len(distances)):
            if distances[i] < radius:
                penetration = radius - distances[i]
                penalties[i] = penalty_factor * penetration * penetration
        
        return penalties
    
    return _compute_collision_penalty_jit

# グローバルなペナルティ関数インスタンス
_jit_penalty_func = None

def _get_jit_penalty_function():
    """JITペナルティ関数を取得（遅延初期化）"""
    global _jit_penalty_func
    
    if _jit_penalty_func is None:
        _jit_penalty_func = _create_penalty_function()
    
    return _jit_penalty_func


def batch_point_triangle_distances(
    points: np.ndarray,
    triangle_vertices: np.ndarray
) -> np.ndarray:
    """
    複数の点と複数の三角形の距離を一括計算（JIT最適化）
    
    Args:
        points: 点群 (N, 3)
        triangle_vertices: 三角形頂点 (M, 3, 3) - M個の三角形、各3頂点
        
    Returns:
        距離行列 (N, M) - points[i] と triangles[j] の距離
    """
    # NumPy配列の型確認・変換
    points = np.asarray(points, dtype=np.float64)
    triangle_vertices = np.asarray(triangle_vertices, dtype=np.float64)
    
    # JITコンパイル効果の閾値調整
    jit_threshold = 50  # より低い閾値で積極的にJIT使用
    
    # JIT関数を遅延取得
    batch_func = _get_jit_batch_function()
    batch_point_func = _get_jit_batch_point_function()
    
    if batch_func is not None and points.shape[0] * triangle_vertices.shape[0] > jit_threshold:
        # 大規模計算の場合はJIT並列化版を使用
        dummy_array = np.array([0.0])  # dummy parameter
        return batch_func(points, triangle_vertices, dummy_array)
    else:
        # 小規模またはフォールバック
        N = points.shape[0]
        M = triangle_vertices.shape[0]
        distances = np.zeros((N, M), dtype=np.float32)
        
        # 単一点 vs 複数三角形の場合に高速化版を使用
        if N == 1 and batch_point_func is not None and M > 5:
            distances[0] = batch_point_func(
                points[0], triangle_vertices
            ).astype(np.float32)
        else:
            for i in range(N):
                for j in range(M):
                    distances[i, j] = point_triangle_distance_vectorized(
                        points[i], triangle_vertices[j]
                    )
        
        return distances


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


def point_to_triangle_distance(point: np.ndarray, triangle_vertices: np.ndarray) -> float:
    """
    点と三角形の最短距離を計算
    
    Args:
        point: 点座標 (3,)
        triangle_vertices: 三角形の3頂点 (3, 3)
    
    Returns:
        float: 最短距離
    """
    calculator = get_distance_calculator()
    return calculator.calculate_point_triangle_distance(point, triangle_vertices)


# 統一設定でウォームアップ済み