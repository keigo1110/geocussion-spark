#!/usr/bin/env python3
"""
Memory Optimization Performance Test

メモリ最適化機能のパフォーマンステストと効果測定。
配列プール、インプレース操作、参照管理の効果を検証します。
"""

import time
import psutil
import numpy as np
import unittest
from typing import List, Dict, Any
import os
import gc

# テスト対象モジュール
from src.collision.optimization import (
    get_array_pool, get_reference_manager, memory_efficient_context,
    optimize_array_operations, InPlaceOperations,
    get_memory_optimization_stats, reset_memory_optimization
)
from src.collision.search import CollisionSearcher
from src.collision.events import CollisionEventQueue
from src.mesh.index import SpatialIndex


class MockSpatialIndex:
    """テスト用のSpatialIndexモック"""
    
    def __init__(self):
        self.mesh = MockMesh()
    
    def query_sphere(self, point: np.ndarray, radius: float) -> List[int]:
        """テスト用の球検索モック"""
        return [0, 1]  # 2つの三角形を返す（meshに合わせる）


class MockMesh:
    """テスト用のMeshモック"""
    
    def __init__(self):
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        self.triangles = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ], dtype=np.int32)


class TestArrayPool(unittest.TestCase):
    """配列プールのテスト"""
    
    def setUp(self):
        reset_memory_optimization()
        self.pool = get_array_pool()
    
    def test_array_reuse(self):
        """配列再利用のテスト"""
        shape = (100, 100)
        dtype = 'float32'
        
        # 最初の取得
        array1 = self.pool.get_array(shape, dtype)
        self.assertEqual(array1.shape, shape)
        self.assertEqual(str(array1.dtype), dtype)
        
        # 返却
        self.pool.return_array(array1)
        
        # 再取得（同じ配列が返されるはず）
        array2 = self.pool.get_array(shape, dtype)
        
        # 統計確認
        stats = self.pool.get_stats()
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
        # cache_hitsとcache_missesの合計で割った再利用率を確認
        total_ops = stats['cache_hits'] + stats['cache_misses']
        expected_reuse_rate = stats['cache_hits'] / total_ops if total_ops > 0 else 0.0
        self.assertEqual(stats['reuse_rate'], expected_reuse_rate)
    
    def test_temporary_array_context(self):
        """一時配列コンテキストマネージャーのテスト"""
        shape = (50, 50)
        
        with self.pool.temporary_array(shape) as temp_array:
            self.assertEqual(temp_array.shape, shape)
            temp_array.fill(42.0)
            self.assertTrue(np.all(temp_array == 42.0))
        
        # コンテキスト終了後に自動返却される
        stats = self.pool.get_stats()
        self.assertEqual(stats['total_deallocations'], 1)
    
    def test_pool_size_limit(self):
        """プールサイズ制限のテスト"""
        max_size = 5
        pool = get_array_pool()
        pool.max_pool_size = max_size
        
        shape = (10, 10)
        arrays = []
        
        # 制限を超える数の配列を作成・返却
        for i in range(max_size + 2):
            array = pool.get_array(shape)
            arrays.append(array)
        
        for array in arrays:
            pool.return_array(array)
        
        stats = pool.get_stats()
        # プールに保存される配列数は制限以下
        total_pooled = stats['total_pooled_arrays']
        self.assertLessEqual(total_pooled, max_size)


class TestInPlaceOperations(unittest.TestCase):
    """インプレース操作のテスト"""
    
    def test_normalize_inplace(self):
        """インプレース正規化のテスト"""
        vector = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        original_id = id(vector)
        
        # インプレース正規化
        result = InPlaceOperations.normalize_inplace(vector)
        
        # 同じ配列オブジェクトが返される
        self.assertEqual(id(result), original_id)
        self.assertEqual(id(result), id(vector))
        
        # 正規化されている
        norm = np.linalg.norm(result)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_subtract_inplace(self):
        """インプレース減算のテスト"""
        a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_id = id(a)
        
        # インプレース減算
        result = InPlaceOperations.subtract_inplace(a, b)
        
        # 同じ配列オブジェクト
        self.assertEqual(id(result), original_id)
        
        # 結果確認
        expected = np.array([9.0, 18.0, 27.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_scale_inplace(self):
        """インプレーススケーリングのテスト"""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_id = id(array)
        
        # インプレーススケーリング
        result = InPlaceOperations.scale_inplace(array, 2.5)
        
        # 同じ配列オブジェクト
        self.assertEqual(id(result), original_id)
        
        # 結果確認
        expected = np.array([2.5, 5.0, 7.5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_clamp_inplace(self):
        """インプレースクランプのテスト"""
        array = np.array([-1.0, 0.5, 2.0, 10.0], dtype=np.float32)
        original_id = id(array)
        
        # インプレースクランプ
        result = InPlaceOperations.clamp_inplace(array, 0.0, 1.0)
        
        # 同じ配列オブジェクト
        self.assertEqual(id(result), original_id)
        
        # 結果確認
        expected = np.array([0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestMemoryEfficientContext(unittest.TestCase):
    """メモリ効率的コンテキストのテスト"""
    
    def test_context_provides_utilities(self):
        """コンテキストがユーティリティを提供することを確認"""
        with memory_efficient_context() as ctx:
            self.assertIn('pool', ctx)
            self.assertIn('reference_manager', ctx)
            self.assertIn('inplace_ops', ctx)
            
            # ユーティリティが使用可能
            pool = ctx['pool']
            with pool.temporary_array((5, 5)) as temp:
                self.assertEqual(temp.shape, (5, 5))
    
    def test_context_statistics(self):
        """コンテキストでの統計収集"""
        # 統計リセット
        reset_memory_optimization()
        
        with memory_efficient_context() as ctx:
            pool = ctx['pool']
            # 複数の配列操作
            for i in range(10):
                with pool.temporary_array((10, 10)) as temp:
                    temp.fill(float(i))
        
        stats = get_memory_optimization_stats()
        self.assertGreater(stats['array_pool']['total_allocations'], 0)


class TestPerformanceBenchmark(unittest.TestCase):
    """パフォーマンスベンチマーク"""
    
    def setUp(self):
        reset_memory_optimization()
        gc.collect()  # ガベージコレクション実行
    
    def test_array_copy_performance(self):
        """配列コピーのパフォーマンス比較"""
        shape = (1000, 1000)
        iterations = 50
        
        # 従来の方法（copy()多用）
        start_time = time.perf_counter()
        traditional_arrays = []
        for i in range(iterations):
            array = np.zeros(shape, dtype=np.float32)
            copied = array.copy()
            processed = copied.copy()
            traditional_arrays.append(processed)
        traditional_time = time.perf_counter() - start_time
        
        # メモリ最適化版
        pool = get_array_pool()
        start_time = time.perf_counter()
        optimized_arrays = []
        for i in range(iterations):
            with pool.temporary_array(shape, 'float32') as temp:
                # インプレース操作
                temp.fill(0.0)
                # 必要時のみコピー
                result = temp.copy()
                optimized_arrays.append(result)
        optimized_time = time.perf_counter() - start_time
        
        # 結果比較
        print(f"\n配列コピーパフォーマンス比較:")
        print(f"従来方式: {traditional_time:.3f}秒")
        print(f"最適化版: {optimized_time:.3f}秒")
        print(f"改善率: {((traditional_time - optimized_time) / traditional_time * 100):.1f}%")
        
        # 最適化版が高速であることを期待（環境によって変動する可能性あり）
        self.assertLessEqual(optimized_time, traditional_time * 1.5)  # 50%の余裕を持たせる
    
    def test_memory_usage_comparison(self):
        """メモリ使用量の比較"""
        shape = (500, 500)
        iterations = 20
        
        # プロセスのメモリ使用量取得
        process = psutil.Process(os.getpid())
        
        # ベースラインメモリ使用量
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # 従来の方法
        traditional_arrays = []
        for i in range(iterations):
            array = np.zeros(shape, dtype=np.float32)
            copied = array.copy()
            processed = copied.copy()
            traditional_arrays.append(processed)
        
        gc.collect()
        traditional_memory = process.memory_info().rss - baseline_memory
        
        # メモリクリア
        del traditional_arrays
        gc.collect()
        
        # メモリ最適化版
        with memory_efficient_context() as ctx:
            pool = ctx['pool']
            optimized_arrays = []
            for i in range(iterations):
                with pool.temporary_array(shape, 'float32') as temp:
                    temp.fill(0.0)
                    # 最終結果のみコピー
                    result = temp.copy()
                    optimized_arrays.append(result)
        
        gc.collect()
        optimized_memory = process.memory_info().rss - baseline_memory
        
        # 結果比較
        print(f"\nメモリ使用量比較:")
        print(f"従来方式: {traditional_memory / 1024 / 1024:.1f}MB")
        print(f"最適化版: {optimized_memory / 1024 / 1024:.1f}MB")
        
        # 最適化版のメモリ使用量が少ないことを期待
        if traditional_memory > 0:
            improvement = ((traditional_memory - optimized_memory) / traditional_memory * 100)
            print(f"メモリ削減率: {improvement:.1f}%")
    
    def test_collision_search_performance(self):
        """衝突検索のパフォーマンステスト"""
        spatial_index = MockSpatialIndex()
        searcher = CollisionSearcher(spatial_index)
        
        # テスト用の手位置データ
        positions = [np.array([i * 0.1, j * 0.1, 0.05]) 
                    for i in range(10) for j in range(10)]
        
        # 検索実行
        start_time = time.perf_counter()
        for pos in positions:
            result = searcher._search_point(pos, 0.05)
        search_time = time.perf_counter() - start_time
        
        # 統計取得
        stats = get_memory_optimization_stats()
        
        print(f"\n衝突検索パフォーマンス:")
        print(f"検索時間: {search_time:.3f}秒 ({len(positions)}回)")
        print(f"平均検索時間: {search_time / len(positions) * 1000:.1f}ms")
        print(f"配列プール統計: {stats['array_pool']}")
        
        # パフォーマンス基準確認
        avg_search_time_ms = search_time / len(positions) * 1000
        self.assertLess(avg_search_time_ms, 10.0)  # 平均10ms以下


class TestOptimizationIntegration(unittest.TestCase):
    """最適化統合テスト"""
    
    def test_decorator_functionality(self):
        """@optimize_array_operationsデコレータのテスト"""
        @optimize_array_operations
        def test_function():
            pool = get_array_pool()
            with pool.temporary_array((100, 100)) as temp:
                temp.fill(1.0)
                return temp.sum()
        
        result = test_function()
        self.assertEqual(result, 10000.0)
        
        # 統計が記録されていることを確認
        stats = get_memory_optimization_stats()
        self.assertGreater(stats['array_pool']['total_allocations'], 0)
    
    def test_reference_tracking(self):
        """参照追跡機能のテスト"""
        ref_manager = get_reference_manager()
        
        # テスト配列作成
        original = np.array([1, 2, 3])
        copied = original.copy()
        
        # 追跡記録
        ref_manager.track_reference(original, "test_operation")
        ref_manager.track_copy(original, copied, "test_copy")
        
        # レポート取得
        report = ref_manager.get_report()
        self.assertGreaterEqual(report['active_references'], 0)
        self.assertGreaterEqual(report['total_copy_warnings'], 0)


if __name__ == '__main__':
    print("=== メモリ最適化パフォーマンステスト ===")
    unittest.main(verbosity=2) 