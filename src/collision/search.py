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
from ..detection.tracker import TrackedHand


class SearchStrategy(Enum):
    """検索戦略の列挙"""
    SPHERE_QUERY = "sphere"          # 球形検索
    FRUSTUM_QUERY = "frustum"        # 錐形検索（手の移動方向考慮）
    ADAPTIVE_RADIUS = "adaptive"     # 適応的半径調整
    PREDICTIVE = "predictive"        # 予測的検索


@dataclass
class SearchResult:
    """検索結果データ構造"""
    triangle_indices: List[int]      # 近傍三角形のインデックス
    distances: List[float]           # 各三角形への距離
    search_time_ms: float           # 検索時間
    query_point: np.ndarray         # 検索点
    search_radius: float            # 検索半径
    num_nodes_visited: int          # 訪問したBVHノード数
    
    @property
    def num_triangles(self) -> int:
        """検索された三角形数を取得"""
        return len(self.triangle_indices)
    
    @property
    def closest_triangle(self) -> Optional[int]:
        """最近傍三角形インデックスを取得"""
        if not self.triangle_indices:
            return None
        min_idx = np.argmin(self.distances)
        return self.triangle_indices[min_idx]
    
    @property
    def closest_distance(self) -> Optional[float]:
        """最近傍距離を取得"""
        if not self.distances:
            return None
        return min(self.distances)


class CollisionSearcher:
    """衝突検出用空間検索クラス"""
    
    def __init__(
        self,
        spatial_index: SpatialIndex,
        default_radius: float = 0.05,      # デフォルト検索半径 (5cm)
        max_radius: float = 0.2,           # 最大検索半径 (20cm)
        strategy: SearchStrategy = SearchStrategy.ADAPTIVE_RADIUS,
        enable_caching: bool = True,       # 結果キャッシュ
        max_cache_size: int = 100          # キャッシュサイズ
    ):
        """
        初期化
        
        Args:
            spatial_index: メッシュフェーズで構築されたBVH空間インデックス
            default_radius: デフォルト検索半径
            max_radius: 最大検索半径
            strategy: 検索戦略
            enable_caching: 結果キャッシュを有効にするか
            max_cache_size: キャッシュの最大サイズ
        """
        self.spatial_index = spatial_index
        self.default_radius = default_radius
        self.max_radius = max_radius
        self.strategy = strategy
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        
        # キャッシュ
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
    
    def search_near_hand(self, hand: TrackedHand, override_radius: Optional[float] = None) -> SearchResult:
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
    
    def batch_search_hands(self, hands: List[TrackedHand]) -> List[SearchResult]:
        """
        複数の手を一括検索
        
        Args:
            hands: トラッキングされた手のリスト
            
        Returns:
            各手の検索結果リスト
        """
        start_time = time.perf_counter()
        results = []
        
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
        
        # キャッシュチェック
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
        
        result = SearchResult(
            triangle_indices=triangle_indices,
            distances=distances,
            search_time_ms=search_time,
            query_point=point.copy(),
            search_radius=radius,
            num_nodes_visited=nodes_visited
        )
        
        # キャッシュに保存
        if self.enable_caching:
            self._cache_result(cache_key, result)
        
        # 統計更新
        self._update_search_stats(result)
        
        return result
    
    def _determine_search_radius(self, hand: TrackedHand, override_radius: Optional[float]) -> float:
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
            # 移動速度に基づいて調整
            speed = np.linalg.norm(hand.velocity)
            speed_factor = min(speed * 10.0 + 1.0, 3.0)  # 最大3倍まで拡大
            return min(self.default_radius * speed_factor, self.max_radius)
        
        return self.default_radius
    
    def _calculate_distances(self, point: np.ndarray, triangle_indices: List[int]) -> List[float]:
        """三角形への距離を計算"""
        if not triangle_indices:
            return []
        
        mesh = self.spatial_index.mesh
        distances = []
        
        for tri_idx in triangle_indices:
            triangle = mesh.triangles[tri_idx]
            triangle_vertices = mesh.vertices[triangle]
            
            # 三角形の重心への距離（簡略化）
            centroid = np.mean(triangle_vertices, axis=0)
            distance = np.linalg.norm(point - centroid)
            distances.append(distance)
        
        return distances
    
    def _update_radius_feedback(self, radius: float, triangles_found: int):
        """適応的半径調整のフィードバック更新"""
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