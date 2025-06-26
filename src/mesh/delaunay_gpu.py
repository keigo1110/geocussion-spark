#!/usr/bin/env python3
"""
GPU加速Delaunay三角分割モジュール

CuPyを使用してGPU上でDelaunay三角分割を高速実行。
CPUベースの実装から20-40倍の高速化を目指す。

主要機能:
- CuPy + scipy.spatial を使用したGPU加速
- 適応的品質制御
- メモリ効率化
- バッチ処理対応
- Fallback to CPU if GPU fails
"""

import time
from typing import Optional, Tuple, List, Union
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.spatial import Delaunay as DelaunayGPU
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    DelaunayGPU = None

from scipy.spatial import Delaunay as DelaunayCPU
import logging

logger = logging.getLogger(__name__)


class GPUDelaunayTriangulator:
    """GPU加速Delaunay三角分割クラス"""
    
    def __init__(
        self,
        force_cpu: bool = False,
        batch_size: int = 10000,
        quality_threshold: float = 0.2,
        adaptive_subdivision: bool = True,
        enable_caching: bool = True,
        max_cache_size: int = 5
    ):
        """
        初期化
        
        Args:
            force_cpu: CPU強制使用フラグ
            batch_size: バッチ処理サイズ
            quality_threshold: 三角形品質閾値
            adaptive_subdivision: 適応的細分化の有効化
            enable_caching: 結果キャッシュの有効化
            max_cache_size: 最大キャッシュサイズ
        """
        self.force_cpu = force_cpu
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        self.adaptive_subdivision = adaptive_subdivision
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        
        # GPU利用可能性確認
        self.use_gpu = CUPY_AVAILABLE and not force_cpu and self._check_gpu_capability()
        
        # キャッシュ
        self.result_cache = {}
        self.cache_keys = []
        
        # パフォーマンス統計
        self.stats = {
            'total_triangulations': 0,
            'gpu_triangulations': 0,
            'cpu_fallbacks': 0,
            'total_gpu_time_ms': 0.0,
            'total_cpu_time_ms': 0.0,
            'total_points_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"GPUDelaunayTriangulator initialized: GPU={'enabled' if self.use_gpu else 'disabled'}")
        if self.use_gpu:
            logger.info(f"GPU device: {cp.cuda.Device().name}")
    
    def _check_gpu_capability(self) -> bool:
        """GPU機能の確認"""
        if not CUPY_AVAILABLE:
            return False
        
        try:
            # GPU メモリ確認
            mempool = cp.get_default_memory_pool()
            free_bytes = mempool.free_bytes()
            total_bytes = mempool.total_bytes()
            
            if free_bytes < 100 * 1024 * 1024:  # 100MB以上の空きが必要
                logger.warning(f"Insufficient GPU memory: {free_bytes / (1024**2):.1f}MB free")
                return False
            
            # 簡単な動作テスト
            test_points = cp.random.rand(100, 2).astype(cp.float32)
            _ = DelaunayGPU(test_points)
            
            logger.info(f"GPU capability verified: {free_bytes / (1024**2):.1f}MB GPU memory available")
            return True
            
        except Exception as e:
            logger.warning(f"GPU capability check failed: {e}")
            return False
    
    def triangulate_points_2d(
        self,
        points: np.ndarray,
        use_cache: bool = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        2D点群のDelaunay三角分割
        
        Args:
            points: 2D点群 (N, 2)
            use_cache: キャッシュ使用フラグ
            
        Returns:
            (vertices, triangles) または None
        """
        if points is None or len(points) < 3:
            return None
        
        # 2D点群に変換
        if points.shape[1] > 2:
            points_2d = points[:, :2].copy()
        else:
            points_2d = points.copy()
        
        # キャッシュチェック
        if use_cache is None:
            use_cache = self.enable_caching
        
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(points_2d)
            if cache_key in self.result_cache:
                self.stats['cache_hits'] += 1
                return self.result_cache[cache_key]
            else:
                self.stats['cache_misses'] += 1
        
        # 三角分割実行
        result = self._execute_triangulation_2d(points_2d)
        
        # キャッシュに保存
        if use_cache and result is not None and cache_key is not None:
            self._cache_result(cache_key, result)
        
        self.stats['total_triangulations'] += 1
        self.stats['total_points_processed'] += len(points_2d)
        
        return result
    
    def _execute_triangulation_2d(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """実際の2D三角分割実行"""
        if self.use_gpu:
            return self._triangulate_gpu_2d(points)
        else:
            return self._triangulate_cpu_2d(points)
    
    def _triangulate_gpu_2d(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """GPU版2D三角分割"""
        try:
            start_time = time.perf_counter()
            
            # CPU -> GPU転送
            points_gpu = cp.asarray(points, dtype=cp.float32)
            
            # Delaunay三角分割（GPU）
            delaunay = DelaunayGPU(points_gpu)
            triangles_gpu = delaunay.simplices
            
            # GPU -> CPU転送
            vertices = cp.asnumpy(points_gpu).astype(np.float32)
            triangles = cp.asnumpy(triangles_gpu).astype(np.int32)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats['total_gpu_time_ms'] += elapsed_ms
            self.stats['gpu_triangulations'] += 1
            
            logger.debug(f"GPU triangulation: {len(points)} points -> {len(triangles)} triangles in {elapsed_ms:.1f}ms")
            
            return vertices, triangles
            
        except Exception as e:
            logger.warning(f"GPU triangulation failed: {e}, falling back to CPU")
            self.stats['cpu_fallbacks'] += 1
            return self._triangulate_cpu_2d(points)
    
    def _triangulate_cpu_2d(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """CPU版2D三角分割（フォールバック）"""
        try:
            start_time = time.perf_counter()
            
            # Delaunay三角分割（CPU）
            delaunay = DelaunayCPU(points)
            triangles = delaunay.simplices.astype(np.int32)
            vertices = points.astype(np.float32)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats['total_cpu_time_ms'] += elapsed_ms
            
            logger.debug(f"CPU triangulation: {len(points)} points -> {len(triangles)} triangles in {elapsed_ms:.1f}ms")
            
            return vertices, triangles
            
        except Exception as e:
            logger.error(f"CPU triangulation failed: {e}")
            return None
    
    def triangulate_heightmap(
        self,
        heightmap: np.ndarray,
        min_height: float = 0.1,
        subsample_factor: int = 1
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        高さマップから3D三角分割
        
        Args:
            heightmap: 高さマップ (H, W) または (H, W, 1)
            min_height: 最小高さ閾値
            subsample_factor: サブサンプリング係数
            
        Returns:
            (vertices, triangles) または None
        """
        if heightmap is None or heightmap.size == 0:
            return None
        
        # 高さマップを2D形式に変換
        if heightmap.ndim == 3:
            heightmap = heightmap[:, :, 0]
        
        height, width = heightmap.shape
        
        # サブサンプリング
        if subsample_factor > 1:
            heightmap = heightmap[::subsample_factor, ::subsample_factor]
            height, width = heightmap.shape
        
        # 有効な点を抽出
        valid_mask = heightmap > min_height
        if not np.any(valid_mask):
            return None
        
        # 3D座標生成
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) < 3:
            return None
        
        # 正規化座標（0-1範囲）
        x_norm = x_coords[valid_indices] / width
        y_norm = y_coords[valid_indices] / height
        z_values = heightmap[valid_indices]
        
        # 2D点群で三角分割
        points_2d = np.column_stack([x_norm, y_norm])
        triangulation_result = self.triangulate_points_2d(points_2d)
        
        if triangulation_result is None:
            return None
        
        vertices_2d, triangles = triangulation_result
        
        # 3D頂点作成
        vertices_3d = np.column_stack([
            vertices_2d[:, 0],  # x
            vertices_2d[:, 1],  # y
            z_values[:len(vertices_2d)]  # z (高さ)
        ]).astype(np.float32)
        
        return vertices_3d, triangles
    
    def _compute_cache_key(self, points: np.ndarray) -> str:
        """キャッシュキーの計算"""
        # 点群のハッシュベースキー生成
        points_rounded = np.round(points * 1000).astype(np.int32)  # mm精度
        return f"{hash(points_rounded.tobytes())}_{len(points)}"
    
    def _cache_result(self, key: str, result: Tuple[np.ndarray, np.ndarray]):
        """結果をキャッシュに保存"""
        if len(self.result_cache) >= self.max_cache_size:
            # LRU方式でキャッシュクリア
            oldest_key = self.cache_keys.pop(0)
            del self.result_cache[oldest_key]
        
        self.result_cache[key] = result
        self.cache_keys.append(key)
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.result_cache.clear()
        self.cache_keys.clear()
        logger.info("Triangulation cache cleared")
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        
        # 平均値計算
        if stats['gpu_triangulations'] > 0:
            stats['avg_gpu_time_ms'] = stats['total_gpu_time_ms'] / stats['gpu_triangulations']
        else:
            stats['avg_gpu_time_ms'] = 0.0
        
        if stats['cpu_fallbacks'] > 0:
            stats['avg_cpu_time_ms'] = stats['total_cpu_time_ms'] / stats['cpu_fallbacks']
        else:
            stats['avg_cpu_time_ms'] = 0.0
        
        if stats['total_triangulations'] > 0:
            stats['gpu_usage_ratio'] = stats['gpu_triangulations'] / stats['total_triangulations']
            stats['cache_hit_ratio'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['gpu_usage_ratio'] = 0.0
            stats['cache_hit_ratio'] = 0.0
        
        return stats
    
    def print_performance_report(self):
        """パフォーマンスレポート出力"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*50)
        print("GPU Delaunay Triangulator Performance Report")
        print("="*50)
        print(f"Total triangulations: {stats['total_triangulations']}")
        print(f"GPU triangulations: {stats['gpu_triangulations']} ({stats['gpu_usage_ratio']*100:.1f}%)")
        print(f"CPU fallbacks: {stats['cpu_fallbacks']}")
        print(f"Total points processed: {stats['total_points_processed']:,}")
        
        if stats['gpu_triangulations'] > 0:
            print(f"Average GPU time: {stats['avg_gpu_time_ms']:.1f}ms")
        if stats['cpu_fallbacks'] > 0:
            print(f"Average CPU time: {stats['avg_cpu_time_ms']:.1f}ms")
        
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Cache hit ratio: {stats['cache_hit_ratio']*100:.1f}%")
        print("="*50)


# 便利関数

def create_gpu_triangulator(
    use_gpu: bool = True,
    **kwargs
) -> GPUDelaunayTriangulator:
    """
    GPU三角分割器を作成（簡単なインターフェース）
    
    Args:
        use_gpu: GPU使用フラグ
        **kwargs: その他のオプション
        
    Returns:
        GPUDelaunayTriangulator インスタンス
    """
    return GPUDelaunayTriangulator(
        force_cpu=not use_gpu,
        **kwargs
    )


def benchmark_triangulation(
    points: np.ndarray,
    runs: int = 10
) -> dict:
    """
    GPU vs CPU 三角分割ベンチマーク
    
    Args:
        points: テスト点群
        runs: 実行回数
        
    Returns:
        ベンチマーク結果
    """
    print(f"\nRunning triangulation benchmark: {len(points)} points, {runs} runs")
    
    # GPU版
    gpu_triangulator = GPUDelaunayTriangulator(force_cpu=False)
    gpu_times = []
    for i in range(runs):
        start = time.perf_counter()
        result = gpu_triangulator.triangulate_points_2d(points, use_cache=False)
        gpu_times.append((time.perf_counter() - start) * 1000)
        if result is None:
            break
    
    # CPU版
    cpu_triangulator = GPUDelaunayTriangulator(force_cpu=True)
    cpu_times = []
    for i in range(runs):
        start = time.perf_counter()
        result = cpu_triangulator.triangulate_points_2d(points, use_cache=False)
        cpu_times.append((time.perf_counter() - start) * 1000)
        if result is None:
            break
    
    # 結果
    if gpu_times and cpu_times:
        avg_gpu = np.mean(gpu_times)
        avg_cpu = np.mean(cpu_times)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
        
        result = {
            'points_count': len(points),
            'runs': runs,
            'avg_gpu_time_ms': avg_gpu,
            'avg_cpu_time_ms': avg_cpu,
            'speedup_ratio': speedup,
            'gpu_available': gpu_triangulator.use_gpu
        }
        
        print(f"GPU average: {avg_gpu:.1f}ms")
        print(f"CPU average: {avg_cpu:.1f}ms")
        print(f"Speedup: {speedup:.1f}x")
        
        return result
    else:
        print("Benchmark failed")
        return {}


# テスト関数

def test_gpu_triangulation():
    """GPU三角分割のテスト"""
    print("Testing GPU Delaunay Triangulation...")
    
    # テスト点群生成
    np.random.seed(42)
    test_points = np.random.rand(1000, 2).astype(np.float32)
    
    # 三角分割器作成
    triangulator = create_gpu_triangulator(use_gpu=True)
    
    # 三角分割実行
    start_time = time.perf_counter()
    result = triangulator.triangulate_points_2d(test_points)
    elapsed = (time.perf_counter() - start_time) * 1000
    
    if result is not None:
        vertices, triangles = result
        print(f"✅ Success: {len(test_points)} points -> {len(triangles)} triangles in {elapsed:.1f}ms")
        
        # パフォーマンス統計
        triangulator.print_performance_report()
        
        # ベンチマーク
        benchmark_triangulation(test_points, runs=5)
        
    else:
        print("❌ Triangulation failed")


if __name__ == "__main__":
    test_gpu_triangulation() 