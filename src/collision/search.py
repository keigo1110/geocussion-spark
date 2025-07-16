#!/usr/bin/env python3
"""
衝突検出 - 空間検索

BVH空間インデックスを使用して手の位置近傍の三角形を効率的に検索する機能を提供します。
メッシュフェーズで構築されたBVHを活用し、5ms予算内での高速検索を実現します。
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from enum import Enum
import numpy as np

# 他フェーズとの連携
from ..mesh.index import SpatialIndex, BVHNode
from ..mesh.delaunay import TriangleMesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detection.tracker import TrackedHand
from .sphere_tri import point_triangle_distance
from .distance import point_triangle_distance_vectorized  # Numba最適化された距離計算
from ..data_types import SearchStrategy, SearchResult
from ..config import get_config
from .optimization import optimize_array_operations
from ..utils.cache_manager import get_cache_manager
from .. import get_logger

logger = get_logger(__name__)


class CollisionSearcher:
    """衝突検出用空間検索クラス"""
    
    def __init__(
        self,
        spatial_index: SpatialIndex,
        default_radius: Optional[float] = None,
        max_radius: Optional[float] = None,
        strategy: SearchStrategy = SearchStrategy.ADAPTIVE_RADIUS,
        enable_caching: Optional[bool] = None,
        max_cache_size: Optional[int] = None
    ):
        """
        初期化
        
        Args:
            spatial_index: メッシュフェーズで構築されたBVH空間インデックス
            default_radius: デフォルト検索半径（Noneの場合は設定ファイルから取得）
            max_radius: 最大検索半径（Noneの場合は設定ファイルから取得）
            strategy: 検索戦略
            enable_caching: 結果キャッシュを有効にするか（Noneの場合は設定ファイルから取得）
            max_cache_size: キャッシュの最大サイズ（Noneの場合は設定ファイルから取得）
        """
        # 設定値を取得
        config = get_config()
        collision_config = config.collision
        
        self.spatial_index = spatial_index
        # BVH 早期打ち切りのデフォルト値（過度なノード訪問を防ぐ）
        if not hasattr(self.spatial_index, "max_nodes_per_query"):
            # 経験的に 400 ノード ≒ 深さ 9 相当で十分
            self.spatial_index.max_nodes_per_query = 400  # type: ignore[attr-defined]
        self.default_radius = default_radius if default_radius is not None else collision_config.default_search_radius
        self.max_radius = max_radius if max_radius is not None else collision_config.max_search_radius
        self.strategy = strategy
        self.enable_caching = enable_caching if enable_caching is not None else True  # デフォルトTrue
        self.max_cache_size = max_cache_size if max_cache_size is not None else collision_config.max_cache_size
        
        # 拡張キャッシュマネージャーを使用
        self.cache_manager = get_cache_manager('collision_search')
        
        # 従来のキャッシュ（統計目的で残す）
        self.search_cache = {}  # {(x, y, z, radius): SearchResult}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 適応的半径調整
        self.radius_history = []
        self.successful_radii = []
        
        # パフォーマンス統計
        self.stats = {
            'total_searches': 0,
            'total_search_time_ms': 0.0,
            'average_search_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'average_triangles_found': 0.0,
            'last_search_time_ms': 0.0,
            'last_triangles_found': 0,
            'nodes_visited_total': 0
        }
    
    @optimize_array_operations
    def search_near_hand(self, hand: 'TrackedHand', override_radius: Optional[float] = None) -> SearchResult:
        """
        手の位置周辺の三角形を検索
        
        Args:
            hand: トラッキングされた手の情報
            override_radius: 半径のオーバーライド
            
        Returns:
            検索結果
        """
        if hand.position is None:
            return SearchResult([], [], 0.0, np.zeros(3), 0.0, 0)
        
        # 検索半径を決定
        radius = self._determine_search_radius(hand, override_radius)
        
        # 予測的検索の場合、移動方向も考慮
        search_points = [hand.position]
        if self.strategy == SearchStrategy.PREDICTIVE and hand.velocity is not None:
            # 次フレームの予測位置も検索
            predicted_pos = hand.position + hand.velocity * 0.016  # 60fps想定
            search_points.append(predicted_pos)
        
        # 検索実行
        all_triangle_indices = set()
        all_distances = []
        total_nodes_visited = 0
        max_search_time = 0.0
        
        for search_point in search_points:
            result = self._search_point(search_point, radius)
            all_triangle_indices.update(result.triangle_indices)
            all_distances.extend(result.distances)
            total_nodes_visited += result.num_nodes_visited
            max_search_time = max(max_search_time, result.search_time_ms)
        
        # 結果統合
        triangle_indices = list(all_triangle_indices)
        
        # 距離を再計算（メイン検索点からの距離）
        if triangle_indices:
            distances = self._calculate_distances(hand.position, triangle_indices)
        else:
            distances = []
        
        # 適応的半径調整のためのフィードバック
        self._update_radius_feedback(radius, len(triangle_indices))
        
        return SearchResult(
            triangle_indices=triangle_indices,
            distances=distances,
            search_time_ms=max_search_time,
            query_point=hand.position,
            search_radius=radius,
            num_nodes_visited=total_nodes_visited
        )
    
    @optimize_array_operations
    def batch_search_hands(self, hands: List['TrackedHand']) -> List[SearchResult]:
        """
        複数の手を一括検索
        
        Args:
            hands: トラッキングされた手のリスト
            
        Returns:
            各手の検索結果リスト
        """
        start_time = time.perf_counter()
        results = []
        
        # メモリ効率的なコンテキストで処理
        with memory_efficient_context() as ctx:
            for hand in hands:
                if hand.position is not None:
                    result = self.search_near_hand(hand)
                    results.append(result)
                else:
                    # 無効な手の場合は空の結果
                    results.append(SearchResult([], [], 0.0, np.zeros(3), 0.0, 0))
        
        # パフォーマンス統計更新
        batch_time = (time.perf_counter() - start_time) * 1000
        self._update_batch_stats(batch_time, results)
        
        return results
    
    def _search_point(self, point: np.ndarray, radius: float) -> SearchResult:
        """単一点の検索"""
        start_time = time.perf_counter()
        
        # 拡張キャッシュからの取得試行
        if self.enable_caching and self.cache_manager:
            cache_key = self._make_cache_key_string(point, radius)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
        
        # 従来のキャッシュチェック（移行期間用）
        if self.enable_caching:
            cache_key = self._make_cache_key(point, radius)
            cached_result = self.search_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
            self.cache_misses += 1
        
        # BVHを使って検索
        triangle_indices = self.spatial_index.query_sphere(point, radius)
        
        # 距離計算
        distances = self._calculate_distances(point, triangle_indices)
        
        # ノード訪問数を推定（実際の実装では正確に追跡）
        nodes_visited = min(len(triangle_indices) * 2, 50)  # 推定値
        
        search_time = (time.perf_counter() - start_time) * 1000
        
        # 最適化版SearchResultを使用（point.copy()を回避）
        from .optimization import create_optimized_search_result
        result = create_optimized_search_result(
            triangle_indices=triangle_indices,
            distances=distances,
            search_time_ms=search_time,
            query_point=point,  # copy()せず参照を使用
            search_radius=radius,
            num_nodes_visited=nodes_visited
        )
        
        # 拡張キャッシュに保存
        if self.enable_caching and self.cache_manager:
            cache_key_str = self._make_cache_key_string(point, radius)
            # 結果サイズを推定
            result_size = self._estimate_result_size(result)
            self.cache_manager.put(cache_key_str, result, result_size)
        
        # 従来のキャッシュに保存
        if self.enable_caching:
            self._cache_result(cache_key, result)
        
        # 統計更新
        self._update_search_stats(result)
        
        return result
    
    def _determine_search_radius(self, hand: 'TrackedHand', override_radius: Optional[float]) -> float:
        """検索半径を決定"""
        if override_radius is not None:
            return min(override_radius, self.max_radius)
        
        if self.strategy == SearchStrategy.ADAPTIVE_RADIUS:
            # 過去の成功率に基づいて調整
            if self.successful_radii:
                avg_successful_radius = np.mean(self.successful_radii[-10:])  # 直近10回の平均
                adaptive_radius = min(avg_successful_radius * 1.1, self.max_radius)
                return max(adaptive_radius, self.default_radius * 0.5)
        
        elif self.strategy == SearchStrategy.FRUSTUM_QUERY and hand.velocity is not None:
            # 速度に応じて半径を拡大
            speed = np.linalg.norm(hand.velocity)
            radius = self.default_radius + speed * 0.1  # 速度比例
            return min(radius, self.max_radius)
        
        return self.default_radius
    
    def _calculate_distances(self, point: np.ndarray, triangle_indices: List[int]) -> List[float]:
        """点と各三角形の正確な最短距離を計算（最適化版）"""
        distances = []
        if not triangle_indices:
            return distances
        
        # 最適化された距離計算を使用
        from .distance import get_distance_calculator
        calculator = get_distance_calculator()
        
        mesh_vertices = self.spatial_index.mesh.vertices
        mesh_triangles = self.spatial_index.mesh.triangles
        
        # バッチ計算が可能な場合はそれを使用
        if len(triangle_indices) > 3:  # バッチ化の閾値
            triangle_vertices_batch = mesh_vertices[mesh_triangles[triangle_indices]]  # (M, 3, 3)
            points_batch = np.array([point])  # (1, 3)
            distance_matrix = calculator.calculate_batch_distances(points_batch, triangle_vertices_batch)
            distances = distance_matrix[0].tolist()
        else:
            # 少数の場合は従来通り個別計算
            for tri_idx in triangle_indices:
                triangle_vertices = mesh_vertices[mesh_triangles[tri_idx]]
                dist = calculator.calculate_point_triangle_distance(point, triangle_vertices)
                distances.append(dist)
            
        return distances
    
    def _update_radius_feedback(self, radius: float, triangles_found: int):
        """適応的半径のためのフィードバック更新"""
        self.radius_history.append((radius, triangles_found))
        
        # 成功した半径を記録（三角形が1個以上見つかった場合）
        if triangles_found > 0:
            self.successful_radii.append(radius)
            # 履歴サイズ制限
            if len(self.successful_radii) > 50:
                self.successful_radii = self.successful_radii[-30:]
    
    def _make_cache_key(self, point: np.ndarray, radius: float) -> Tuple[int, int, int, int]:
        """キャッシュキーを生成（量子化で近似）"""
        # 1mm精度で量子化
        quantized_point = (point * 1000).astype(int)
        quantized_radius = int(radius * 1000)
        return (quantized_point[0], quantized_point[1], quantized_point[2], quantized_radius)

    def _make_cache_key_string(self, point: np.ndarray, radius: float) -> str:
        """文字列キャッシュキーを生成（拡張キャッシュ用）"""
        # 1mm精度で量子化
        quantized_point = (point * 1000).astype(int)
        quantized_radius = int(radius * 1000)
        return f"search_{quantized_point[0]}_{quantized_point[1]}_{quantized_point[2]}_{quantized_radius}"

    def _estimate_result_size(self, result: SearchResult) -> int:
        """SearchResultのメモリサイズを推定"""
        if result is None:
            return 0
        
        # 基本サイズ + 三角形インデックス配列 + 距離配列
        base_size = 100  # 基本オブジェクトサイズ
        indices_size = len(result.triangle_indices) * 4  # int32想定
        distances_size = len(result.distances) * 8  # float64想定
        
        return base_size + indices_size + distances_size

    def _cache_result(self, cache_key: Tuple, result: SearchResult):
        """結果をキャッシュに保存"""
        if len(self.search_cache) >= self.max_cache_size:
            # LRU削除（簡易版）
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = result
    
    def _update_search_stats(self, result: SearchResult):
        """検索統計を更新"""
        self.stats['total_searches'] += 1
        self.stats['total_search_time_ms'] += result.search_time_ms
        self.stats['average_search_time_ms'] = (
            self.stats['total_search_time_ms'] / self.stats['total_searches']
        )
        self.stats['last_search_time_ms'] = result.search_time_ms
        self.stats['last_triangles_found'] = result.num_triangles
        self.stats['nodes_visited_total'] += result.num_nodes_visited
        
        # 平均三角形数更新
        total_triangles = self.stats.get('total_triangles_found', 0) + result.num_triangles
        self.stats['total_triangles_found'] = total_triangles
        self.stats['average_triangles_found'] = total_triangles / self.stats['total_searches']
        
        # キャッシュヒット率更新
        total_queries = self.cache_hits + self.cache_misses
        if total_queries > 0:
            self.stats['cache_hit_rate'] = self.cache_hits / total_queries
    
    def _update_batch_stats(self, batch_time_ms: float, results: List[SearchResult]):
        """バッチ検索統計を更新"""
        if results:
            avg_search_time = np.mean([r.search_time_ms for r in results])
            total_triangles = sum(r.num_triangles for r in results)
            
            self.stats['last_batch_time_ms'] = batch_time_ms
            self.stats['last_batch_avg_search_time_ms'] = avg_search_time
            self.stats['last_batch_total_triangles'] = total_triangles
    
    def clear_cache(self):
        """キャッシュをクリア"""
        if self.cache_manager:
            self.cache_manager.clear()
        
        # 従来のキャッシュもクリア
        self.search_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_searches': 0,
            'total_search_time_ms': 0.0,
            'average_search_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'average_triangles_found': 0.0,
            'last_search_time_ms': 0.0,
            'last_triangles_found': 0,
            'nodes_visited_total': 0
        }
        self.clear_cache()

    def search_near_hand_optimized(
        self,
        hand: 'TrackedHand',
        override_radius: Optional[float] = None,
        enable_broadphase: bool = True
    ) -> SearchResult:
        """
        最適化された手近傍三角形検索（ブロードフェーズ前処理付き）
        
        Args:
            hand: 追跡された手
            override_radius: 検索半径のオーバーライド
            enable_broadphase: ブロードフェーズ最適化の有効化
            
        Returns:
            検索結果（三角形インデックスと距離）
        """
        search_radius = override_radius or self.default_radius
        
        if enable_broadphase and hasattr(self.spatial_index, 'kdtree') and self.spatial_index.kdtree:
            # Phase 1: KD-Tree による粗い近傍検索
            candidate_indices = self.spatial_index.query_point(hand.position, search_radius * 1.5)
            
            if len(candidate_indices) == 0:
                return SearchResult(triangle_indices=[], distances=[])
                
            # Phase 2: 候補三角形のみで精密距離計算
            candidate_triangles = self.spatial_index.mesh.triangles[candidate_indices]
            triangle_vertices = self.spatial_index.mesh.vertices[candidate_triangles]
            
            # 距離計算（最適化された候補のみ）
            distances = []
            for i, triangle_verts in enumerate(triangle_vertices):
                dist = self._calculate_distance_to_triangle(hand.position, triangle_verts)
                distances.append(dist)
            
            # 検索半径内の三角形をフィルタ
            valid_mask = np.array(distances) <= search_radius
            final_indices = np.array(candidate_indices)[valid_mask]
            final_distances = np.array(distances)[valid_mask]
            
            # ソートして返す
            sort_indices = np.argsort(final_distances)
            return SearchResult(
                triangle_indices=final_indices[sort_indices].tolist(),
                distances=final_distances[sort_indices].tolist()
            )
        else:
            # Fallback: 従来の全探索
            return self.search_near_hand(hand, override_radius)
    
    def _calculate_distance_to_triangle(self, point: np.ndarray, triangle_vertices: np.ndarray) -> float:
        """点と三角形の距離を計算"""
        return point_triangle_distance_vectorized(point, triangle_vertices)


# 便利関数

def search_nearby_triangles(
    spatial_index: SpatialIndex,
    position: np.ndarray,
    radius: float = 0.05
) -> SearchResult:
    """
    近傍三角形を検索（簡単なインターフェース）
    
    Args:
        spatial_index: BVH空間インデックス
        position: 検索位置
        radius: 検索半径
        
    Returns:
        検索結果
    """
    searcher = CollisionSearcher(spatial_index)
    return searcher._search_point(position, radius)


def batch_search_triangles(
    spatial_index: SpatialIndex,
    positions: List[np.ndarray],
    radius: float = 0.05
) -> List[SearchResult]:
    """
    複数位置の一括検索（簡単なインターフェース）
    
    Args:
        spatial_index: BVH空間インデックス
        positions: 検索位置リスト
        radius: 検索半径
        
    Returns:
        検索結果リスト
    """
    searcher = CollisionSearcher(spatial_index)
    results = []
    
    for position in positions:
        result = searcher._search_point(position, radius)
        results.append(result)
    
    return results 