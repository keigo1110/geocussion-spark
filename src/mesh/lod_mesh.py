#!/usr/bin/env python3
"""
Level of Detail (LOD) メッシュ生成システム

ハンド近傍領域のみ高解像度でメッシュ生成し、遠い領域は低解像度または除外。
メッシュ生成時間を大幅短縮（5-10倍高速化）を実現。

主要機能:
- ハンド位置ベースの適応的解像度制御
- 距離ベースLOD計算
- 効率的な点群フィルタリング
- 動的メッシュ更新領域の決定
"""

import time
import gc
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import logging
from concurrent.futures import Future

from ..detection.tracker import TrackedHand, TrackingState
from .delaunay import DelaunayTriangulator, TriangleMesh
from ..utils.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class LODConfig:
    """LOD設定"""
    # 距離ベースLOD設定
    high_detail_radius: float = 0.20     # 高解像度半径（メートル）
    medium_detail_radius: float = 0.50   # 中解像度半径
    low_detail_radius: float = 1.0       # 低解像度半径
    
    # 解像度設定
    high_detail_density: float = 0.01    # 高解像度点間距離（0.5cm,以前は1cm）
    medium_detail_density: float = 0.02 # 中解像度点間距離（1.5cm,以前は3cm）
    low_detail_density: float = 0.05     # 低解像度点間距離（5cm,以前は10cm）
    
    # メッシュ更新制御
    update_threshold_move: float = 0.05   # 手移動による更新閾値
    update_threshold_time: float = 1.0    # 時間による強制更新間隔（秒）
    enable_temporal_stability: bool = True  # 時間的安定性制御
    
    # パフォーマンス設定
    max_points_per_lod: int = 4000       # LODレベル毎の最大点数（以前は2000）
    enable_caching: bool = True          # 結果キャッシュ
    cache_validity_time: float = 0.5     # キャッシュ有効時間（秒）


class LODMeshGenerator:
    """LOD (Level of Detail) メッシュ生成器"""
    
    def __init__(
        self,
        config: Optional[LODConfig] = None,
        triangulator: Optional[DelaunayTriangulator] = None
    ):
        self.config = config or LODConfig()
        self.triangulator = triangulator or DelaunayTriangulator()
        
        # 拡張キャッシュマネージャーを使用
        self.cache_manager = get_cache_manager('lod_mesh')
        
        # 従来のキャッシュは廃止
        self.cached_mesh = None
        self.cache_timestamp = 0.0
        
        # 手の位置履歴（更新判定用）
        self.last_hand_positions = []
        self.last_update_time = 0.0
        
        # 非同期処理用
        self._pending_future: Optional[Future] = None
        
        # 統計
        self.stats = {
            'total_updates': 0,
            'total_time_ms': 0.0,
            'lod_time_ms': 0.0,
            'triangulation_time_ms': 0.0,
            'total_lod_time_ms': 0.0,
            'total_triangulation_time_ms': 0.0,
            'cache_hits': 0,
            'original_points_total': 0,
            'filtered_points_total': 0,
            'average_points_reduction_ratio': 0.0,
            'points_processed_per_level': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        # メッシュ更新領域（メモリ解放用）
        self.mesh_update_regions = []
        
        logger.info(f"LODMeshGenerator initialized with config: "
                   f"cache_enabled={self.config.enable_caching}")

    def generate_mesh(
        self,
        points_3d: np.ndarray,
        tracked_hands: List[TrackedHand],
        force_update: bool = False
    ) -> Optional[TriangleMesh]:
        """
        LODベースでメッシュ生成
        
        Args:
            points_3d: 3D点群
            tracked_hands: 追跡中の手
            force_update: 強制更新フラグ
            
        Returns:
            三角形メッシュまたはNone
        """
        if points_3d is None or len(points_3d) < 10:
            return None
        
        start_time = time.perf_counter()
        current_time = time.perf_counter()
        
        # キャッシュキーを生成
        cache_key = self._generate_cache_key(points_3d, tracked_hands)
        
        # 拡張キャッシュから取得試行
        if self.config.enable_caching and not force_update and self.cache_manager:
            cached_mesh = self.cache_manager.get(cache_key)
            if cached_mesh is not None:
                self.stats['cache_hits'] += 1
                return cached_mesh
        
        # 更新判定
        should_update = self._should_update_mesh(tracked_hands, current_time, force_update)
        
        if not should_update and self.cached_mesh is not None:
            # 従来のキャッシュもチェック（移行期間用）
            return self.cached_mesh
        
        # LODベース点群フィルタリング
        lod_start = time.perf_counter()
        filtered_points = self._apply_lod_filtering(points_3d, tracked_hands)
        lod_time = (time.perf_counter() - lod_start) * 1000
        
        if len(filtered_points) < 3:
            logger.warning(f"Insufficient points after LOD filtering: {len(filtered_points)}")
            return self.cached_mesh
        
        # ---- Global cap to avoid CPU overload (P-PERF-002) ----
        global_cap = self.config.max_points_per_lod * 3  # high+medium+low approx.
        if len(filtered_points) > global_cap:
            # Random uniform subsample – reproducible using default_rng
            rng = np.random.default_rng(12345)
            idx = rng.choice(filtered_points.shape[0], size=global_cap, replace=False)
            filtered_points = filtered_points[idx]
            logger.debug("LODMeshGenerator: Downsampled to %d points (cap)", global_cap)
        
        # 三角分割実行
        tri_start = time.perf_counter()
        try:
            # Heavy CPU path: use async if GPU unavailable and many points
            if (
                not self.triangulator.use_gpu
                and len(filtered_points) > self.config.max_points_per_lod * 2
            ):
                if self._pending_future is None:
                    self._pending_future = self.triangulator.triangulate_points_async(filtered_points)
                    logger.debug("LODMeshGenerator: Triangulation offloaded async (%d pts)", len(filtered_points))
                # Return last cached mesh until future ready
                return self.cached_mesh

            mesh = self.triangulator.triangulate_points(filtered_points)

            tri_time = (time.perf_counter() - tri_start) * 1000
            
            # 拡張キャッシュに保存
            if self.config.enable_caching and self.cache_manager:
                # メッシュサイズを推定
                mesh_size = self._estimate_mesh_size(mesh)
                self.cache_manager.put(cache_key, mesh, mesh_size)
            
            # 従来のキャッシュ更新 (旧メッシュを確実に解放) – T-MESH-104
            if self.cached_mesh is not None:
                try:
                    del self.cached_mesh  # remove strong ref
                except Exception:
                    pass
                # 不要参照クリア
                self.mesh_update_regions.clear()
                gc.collect()

            self.cached_mesh = mesh
            self.cache_timestamp = current_time
            self.last_hand_positions = [hand.position.copy() for hand in tracked_hands]
            self.last_update_time = current_time
            
            # 統計更新
            total_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(total_time, lod_time, tri_time, len(points_3d), len(filtered_points))
            
            logger.debug(f"LOD mesh generated: {len(points_3d)} -> {len(filtered_points)} points "
                        f"({100 * len(filtered_points) / len(points_3d):.1f}%), "
                        f"total: {total_time:.1f}ms, triangulation: {tri_time:.1f}ms")
            
            return mesh
            
        except Exception as e:
            logger.error(f"LOD mesh generation failed: {e}")
            return self.cached_mesh

    def _generate_cache_key(self, points_3d: np.ndarray, tracked_hands: List[TrackedHand]) -> str:
        """キャッシュキーを生成"""
        # 点群のハッシュ
        points_hash = hash(points_3d.tobytes())
        
        # 手の位置のハッシュ
        hands_hash = 0
        for hand in tracked_hands:
            hands_hash ^= hash(hand.position.tobytes())
        
        # 設定のハッシュ
        config_hash = hash((
            self.config.high_detail_radius,
            self.config.medium_detail_radius,
            self.config.low_detail_radius,
            self.config.high_detail_density,
            self.config.medium_detail_density,
            self.config.low_detail_density
        ))
        
        return f"lod_mesh_{points_hash}_{hands_hash}_{config_hash}"

    def _estimate_mesh_size(self, mesh: TriangleMesh) -> int:
        """メッシュのメモリサイズを推定"""
        if mesh is None:
            return 0
        
        # 頂点とインデックスのサイズを概算
        vertices_size = mesh.vertices.nbytes if hasattr(mesh.vertices, 'nbytes') else 0
        triangles_size = mesh.triangles.nbytes if hasattr(mesh.triangles, 'nbytes') else 0
        
        return vertices_size + triangles_size

    def _should_update_mesh(
        self,
        tracked_hands: List[TrackedHand],
        current_time: float,
        force_update: bool
    ) -> bool:
        """メッシュ更新の必要性判定"""
        
        if force_update:
            return True
        
        # 初回更新
        if self.cached_mesh is None:
            return True
        
        # 時間による強制更新
        time_since_update = current_time - self.last_update_time
        if time_since_update > self.config.update_threshold_time:
            return True
        
        # 手の移動による更新
        if len(tracked_hands) != len(self.last_hand_positions):
            return True
        
        if len(tracked_hands) > 0:
            for i, hand in enumerate(tracked_hands):
                if i >= len(self.last_hand_positions):
                    return True
                
                movement = np.linalg.norm(hand.position - self.last_hand_positions[i])
                if movement > self.config.update_threshold_move:
                    return True
        
        return False
    
    def _is_cache_valid(self, current_time: float) -> bool:
        """キャッシュ有効性確認（従来のキャッシュ用）"""
        if not self.config.enable_caching or self.cached_mesh is None:
            return False
        
        return (current_time - self.cache_timestamp) < self.config.cache_validity_time
    
    def _apply_lod_filtering(
        self,
        points_3d: np.ndarray,
        tracked_hands: List[TrackedHand]
    ) -> np.ndarray:
        """LODベース点群フィルタリング"""
        
        if len(tracked_hands) == 0:
            # 手が検出されていない場合は全体を低解像度で処理
            return self._uniform_downsample(points_3d, self.config.low_detail_density)
        
        # 各点の最小距離を計算（全ての手からの最小距離）
        min_distances = np.full(len(points_3d), np.inf)
        
        for hand in tracked_hands:
            distances = np.linalg.norm(points_3d - hand.position, axis=1)
            min_distances = np.minimum(min_distances, distances)
        
        # LODレベルを決定
        high_detail_mask = min_distances <= self.config.high_detail_radius
        medium_detail_mask = (min_distances > self.config.high_detail_radius) & \
                            (min_distances <= self.config.medium_detail_radius)
        low_detail_mask = (min_distances > self.config.medium_detail_radius) & \
                         (min_distances <= self.config.low_detail_radius)
        
        # 各レベル毎に点群をサンプリング
        filtered_points = []
        
        # 高解像度領域
        if np.any(high_detail_mask):
            high_points = points_3d[high_detail_mask]
            sampled_high = self._adaptive_downsample(
                high_points, self.config.high_detail_density, self.config.max_points_per_lod
            )
            filtered_points.append(sampled_high)
            self.stats['points_processed_per_level']['high'] += len(sampled_high)
        
        # 中解像度領域
        if np.any(medium_detail_mask):
            medium_points = points_3d[medium_detail_mask]
            sampled_medium = self._adaptive_downsample(
                medium_points, self.config.medium_detail_density, self.config.max_points_per_lod
            )
            filtered_points.append(sampled_medium)
            self.stats['points_processed_per_level']['medium'] += len(sampled_medium)
        
        # 低解像度領域
        if np.any(low_detail_mask):
            low_points = points_3d[low_detail_mask]
            sampled_low = self._adaptive_downsample(
                low_points, self.config.low_detail_density, self.config.max_points_per_lod // 2
            )
            filtered_points.append(sampled_low)
            self.stats['points_processed_per_level']['low'] += len(sampled_low)
        
        if not filtered_points:
            # フォールバック: 全体を低解像度でサンプリング
            return self._uniform_downsample(points_3d, self.config.low_detail_density)
        
        return np.vstack(filtered_points)
    
    def _adaptive_downsample(
        self,
        points: np.ndarray,
        target_density: float,
        max_points: int
    ) -> np.ndarray:
        """適応的ダウンサンプリング"""
        if len(points) == 0:
            return points
        
        # 境界ボックス計算
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        volume = np.prod(max_bounds - min_bounds)
        
        # 目標点数計算
        if volume > 0:
            target_points = min(int(volume / (target_density ** 3)), max_points)
        else:
            target_points = max_points
        
        if len(points) <= target_points:
            return points
        
        # ランダムサンプリングまたはグリッドサンプリング
        if target_points < len(points) // 4:
            # 大幅な削減が必要な場合はグリッドサンプリング
            return self._grid_downsample(points, target_density)
        else:
            # ランダムサンプリング
            indices = np.random.choice(len(points), target_points, replace=False)
            return points[indices]
    
    def _uniform_downsample(self, points: np.ndarray, density: float) -> np.ndarray:
        """一様ダウンサンプリング"""
        return self._grid_downsample(points, density)
    
    def _grid_downsample(self, points: np.ndarray, grid_size: float) -> np.ndarray:
        """グリッドベースダウンサンプリング"""
        if len(points) == 0:
            return points
        
        # グリッド座標に量子化
        grid_coords = np.floor(points / grid_size).astype(np.int32)
        
        # 重複除去（各グリッドセルから1点のみ）
        unique_coords, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
        
        return points[unique_indices]
    
    def _update_stats(
        self,
        total_time_ms: float,
        lod_time_ms: float,
        tri_time_ms: float,
        original_points: int,
        filtered_points: int
    ):
        """統計更新"""
        self.stats['total_updates'] += 1
        self.stats['total_lod_time_ms'] += lod_time_ms
        self.stats['total_triangulation_time_ms'] += tri_time_ms
        
        # 点数削減率の更新
        reduction_ratio = filtered_points / original_points if original_points > 0 else 0
        prev_avg = self.stats['average_points_reduction_ratio']
        n = self.stats['total_updates']
        self.stats['average_points_reduction_ratio'] = (prev_avg * (n-1) + reduction_ratio) / n
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        
        if stats['total_updates'] > 0:
            stats['average_lod_time_ms'] = stats['total_lod_time_ms'] / stats['total_updates']
            stats['average_triangulation_time_ms'] = stats['total_triangulation_time_ms'] / stats['total_updates']
            stats['cache_hit_ratio'] = stats['cache_hits'] / (stats['total_updates'] + stats['cache_hits'])
        else:
            stats['average_lod_time_ms'] = 0.0
            stats['average_triangulation_time_ms'] = 0.0
            stats['cache_hit_ratio'] = 0.0
        
        return stats
    
    def print_performance_report(self):
        """パフォーマンスレポート出力"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*50)
        print("LOD Mesh Generator Performance Report")
        print("="*50)
        print(f"Total updates: {stats['total_updates']}")
        print(f"Cache hits: {stats['cache_hits']} (ratio: {stats['cache_hit_ratio']*100:.1f}%)")
        print(f"Average LOD filtering time: {stats['average_lod_time_ms']:.1f}ms")
        print(f"Average triangulation time: {stats['average_triangulation_time_ms']:.1f}ms")
        print(f"Average points reduction: {(1-stats['average_points_reduction_ratio'])*100:.1f}%")
        
        print("\nPoints processed per LOD level:")
        for level, count in stats['points_processed_per_level'].items():
            print(f"  {level}: {count:,} points")
        
        print("="*50)
    
    def clear_cache(self):
        """キャッシュクリア"""
        if self.cache_manager:
            self.cache_manager.clear()
        
        # 従来のキャッシュもクリア
        self.cached_mesh = None
        self.cache_timestamp = 0.0
        self.last_hand_positions = []
        logger.info("LOD mesh cache cleared")


# 便利関数

def create_lod_mesh_generator(
    high_radius: float = 0.20,
    medium_radius: float = 0.50,
    enable_gpu: bool = True,
    **kwargs
) -> LODMeshGenerator:
    """
    LODメッシュ生成器を作成（簡単なインターフェース）
    
    Args:
        high_radius: 高解像度半径
        medium_radius: 中解像度半径
        enable_gpu: GPU使用フラグ
        **kwargs: その他のLOD設定
        
    Returns:
        LODMeshGenerator インスタンス
    """
    config = LODConfig(
        high_detail_radius=high_radius,
        medium_detail_radius=medium_radius,
        **kwargs
    )
    
    triangulator = DelaunayTriangulator(use_gpu=enable_gpu)
    
    return LODMeshGenerator(config=config, triangulator=triangulator)


# テスト関数

def test_lod_mesh_generation():
    """LODメッシュ生成のテスト"""
    print("Testing LOD Mesh Generation...")
    
    # テストデータ生成
    np.random.seed(42)
    test_points = np.random.rand(10000, 3).astype(np.float32)
    test_points[:, 2] += 0.5  # Z座標調整
    
    # モック手位置
    from ..detection.hands2d import HandednessType
    mock_hand = TrackedHand(
        id="test_hand",
        handedness=HandednessType.RIGHT,
        state=TrackingState.TRACKING,
        position=np.array([0.5, 0.5, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        acceleration=np.array([0.0, 0.0, 0.0]),
        confidence_2d=0.9,
        confidence_3d=0.8,
        confidence_tracking=0.9,
        last_seen_time=time.time(),
        track_length=10,
        lost_frames=0,
        hand_size=0.18  # 18cm typical hand size
    )
    
    # LOD生成器作成
    lod_generator = create_lod_mesh_generator()
    
    # メッシュ生成テスト
    start_time = time.perf_counter()
    mesh = lod_generator.generate_mesh(test_points, [mock_hand])
    elapsed = (time.perf_counter() - start_time) * 1000
    
    if mesh is not None:
        print(f"✅ Success: {len(test_points)} -> {mesh.num_vertices} vertices, "
              f"{mesh.num_triangles} triangles in {elapsed:.1f}ms")
        
        # パフォーマンス統計
        lod_generator.print_performance_report()
        
    else:
        print("❌ LOD mesh generation failed")


if __name__ == "__main__":
    test_lod_mesh_generation() 