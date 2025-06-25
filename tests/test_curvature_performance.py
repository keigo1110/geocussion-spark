"""
曲率計算パフォーマンステスト

perf-005対応のベクトル化曲率計算システムの性能検証
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch
import asyncio
from typing import List, Dict, Any

from src.mesh.delaunay import TriangleMesh
from src.mesh.attributes import AttributeCalculator
from src.mesh.curvature_vectorized import (
    VectorizedCurvatureCalculator,
    CurvatureResult,
    compute_curvatures_fast
)


class TestCurvaturePerformance:
    """曲率計算パフォーマンステスト"""
    
    @pytest.fixture
    def small_mesh(self) -> TriangleMesh:
        """小規模メッシュ (1000頂点)"""
        np.random.seed(42)
        vertices = np.random.rand(1000, 3) * 10
        triangles = []
        
        # 簡単な三角形生成
        for i in range(0, len(vertices) - 2, 3):
            triangles.append([i, i + 1, i + 2])
        
        return TriangleMesh(vertices=vertices, triangles=np.array(triangles))
    
    @pytest.fixture
    def medium_mesh(self) -> TriangleMesh:
        """中規模メッシュ (10000頂点)"""
        np.random.seed(42)
        vertices = np.random.rand(10000, 3) * 10
        triangles = []
        
        for i in range(0, min(len(vertices) - 2, 30000), 3):
            triangles.append([i, i + 1, i + 2])
        
        return TriangleMesh(vertices=vertices, triangles=np.array(triangles))
    
    @pytest.fixture
    def large_mesh(self) -> TriangleMesh:
        """大規模メッシュ (50000頂点)"""
        np.random.seed(42)
        vertices = np.random.rand(50000, 3) * 10
        triangles = []
        
        for i in range(0, min(len(vertices) - 2, 150000), 3):
            triangles.append([i, i + 1, i + 2])
        
        return TriangleMesh(vertices=vertices, triangles=np.array(triangles))
    
    def test_vectorized_vs_fallback_performance(self, medium_mesh):
        """ベクトル化版vs従来版のパフォーマンス比較"""
        
        # ベクトル化版測定
        calculator_vectorized = AttributeCalculator(use_vectorized=True)
        
        start_time = time.perf_counter()
        result_vectorized = calculator_vectorized.compute_attributes(medium_mesh)
        vectorized_time = (time.perf_counter() - start_time) * 1000
        
        # 従来版測定
        calculator_fallback = AttributeCalculator(use_vectorized=False)
        
        start_time = time.perf_counter()
        result_fallback = calculator_fallback.compute_attributes(medium_mesh)
        fallback_time = (time.perf_counter() - start_time) * 1000
        
        # パフォーマンス検証
        speedup = fallback_time / vectorized_time
        
        print(f"\nパフォーマンス比較 (メッシュ: {medium_mesh.num_vertices}頂点):")
        print(f"  ベクトル化版: {vectorized_time:.1f}ms")
        print(f"  従来版:      {fallback_time:.1f}ms")
        print(f"  高速化比:    {speedup:.1f}x")
        
        # 期待値: 3x以上の高速化
        assert speedup >= 3.0, f"Expected at least 3x speedup, got {speedup:.1f}x"
        
        # 結果の整合性確認（形状は同じである必要がある）
        assert result_vectorized.vertex_curvatures.shape == result_fallback.vertex_curvatures.shape
        assert result_vectorized.gradients.shape == result_fallback.gradients.shape
    
    def test_scalability_performance(self, small_mesh, medium_mesh, large_mesh):
        """スケーラビリティ性能テスト"""
        
        calculator = VectorizedCurvatureCalculator()
        mesh_sizes = []
        computation_times = []
        
        for mesh, label in [(small_mesh, "小"), (medium_mesh, "中"), (large_mesh, "大")]:
            start_time = time.perf_counter()
            result = calculator.compute_curvatures(mesh)
            computation_time = (time.perf_counter() - start_time) * 1000
            
            mesh_sizes.append(mesh.num_vertices)
            computation_times.append(computation_time)
            
            print(f"\n{label}規模メッシュ ({mesh.num_vertices}頂点):")
            print(f"  計算時間: {computation_time:.1f}ms")
            print(f"  頂点あたり: {computation_time/mesh.num_vertices:.4f}ms/vertex")
            
            # 頂点あたりの計算時間が 0.01ms 以下であることを確認
            per_vertex_time = computation_time / mesh.num_vertices
            assert per_vertex_time <= 0.01, f"Per-vertex time too high: {per_vertex_time:.4f}ms"
        
        # スケーラビリティチェック（線形に近いスケーリング）
        # 大規模メッシュの頂点あたり時間が小規模メッシュの5倍以内であることを確認
        small_per_vertex = computation_times[0] / mesh_sizes[0]
        large_per_vertex = computation_times[2] / mesh_sizes[2]
        scaling_factor = large_per_vertex / small_per_vertex
        
        print(f"\nスケーラビリティ:")
        print(f"  小規模 頂点あたり: {small_per_vertex:.6f}ms")
        print(f"  大規模 頂点あたり: {large_per_vertex:.6f}ms")
        print(f"  スケーリング比:   {scaling_factor:.1f}x")
        
        assert scaling_factor <= 5.0, f"Poor scalability: {scaling_factor:.1f}x degradation"
    
    def test_caching_performance(self, medium_mesh):
        """キャッシュ性能テスト"""
        
        calculator = VectorizedCurvatureCalculator(enable_caching=True)
        
        # 初回計算（キャッシュミス）
        start_time = time.perf_counter()
        result1 = calculator.compute_curvatures(medium_mesh)
        first_time = (time.perf_counter() - start_time) * 1000
        
        # 2回目計算（キャッシュヒット）
        start_time = time.perf_counter()
        result2 = calculator.compute_curvatures(medium_mesh)
        second_time = (time.perf_counter() - start_time) * 1000
        
        # キャッシュ効果確認
        cache_speedup = first_time / second_time
        
        print(f"\nキャッシュ性能:")
        print(f"  初回計算:    {first_time:.1f}ms (cache miss)")
        print(f"  2回目計算:   {second_time:.1f}ms (cache hit)")
        print(f"  キャッシュ効果: {cache_speedup:.1f}x")
        
        # キャッシュで大幅な高速化を期待
        assert cache_speedup >= 10.0, f"Cache not effective: {cache_speedup:.1f}x speedup"
        assert result2.cached, "Second result should be cached"
        
        # 結果の一致確認
        np.testing.assert_array_equal(result1.vertex_curvatures, result2.vertex_curvatures)
    
    @pytest.mark.asyncio
    async def test_async_performance(self, medium_mesh):
        """非同期計算性能テスト"""
        
        calculator = VectorizedCurvatureCalculator(enable_async=True)
        
        # 非同期計算
        start_time = time.perf_counter()
        result = calculator.compute_curvatures(medium_mesh, async_mode=True)
        async_time = (time.perf_counter() - start_time) * 1000
        
        # 結果確認（初回は通常フォールバック）
        if result.cached:
            print(f"\n非同期計算: {async_time:.1f}ms (fallback)")
            
            # 少し待って再度取得
            await asyncio.sleep(0.1)
            
            start_time = time.perf_counter()
            result2 = calculator.compute_curvatures(medium_mesh, async_mode=True)
            async_time2 = (time.perf_counter() - start_time) * 1000
            
            print(f"非同期結果取得: {async_time2:.1f}ms")
            
            # 結果が得られていることを確認
            assert not result2.cached or result2.computation_time_ms > 0
    
    def test_memory_efficiency(self, large_mesh):
        """メモリ効率性テスト"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        calculator = VectorizedCurvatureCalculator()
        
        # 複数回計算してメモリリークをチェック
        for i in range(5):
            result = calculator.compute_curvatures(large_mesh)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            print(f"反復 {i+1}: メモリ使用量 {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # メモリ増加が大きすぎないことを確認
            assert memory_increase <= 500, f"Memory usage too high: +{memory_increase:.1f}MB"
        
        # ガベージコレクション
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        final_increase = final_memory - initial_memory
        
        print(f"最終メモリ使用量: {final_memory:.1f}MB (+{final_increase:.1f}MB)")
        
        # 最終的なメモリ増加が合理的な範囲内であることを確認
        assert final_increase <= 200, f"Memory leak detected: +{final_increase:.1f}MB"
    
    def test_high_frequency_computation(self, small_mesh):
        """高頻度計算性能テスト（リアルタイム想定）"""
        
        calculator = VectorizedCurvatureCalculator(enable_caching=True)
        
        # 30FPS相当の連続計算
        target_fps = 30
        frame_count = 60  # 2秒分
        frame_time_limit = 1000 / target_fps  # 33.33ms
        
        computation_times = []
        
        for frame in range(frame_count):
            start_time = time.perf_counter()
            
            # フレームごとに少し変更したメッシュ（現実的なシミュレーション）
            mesh_copy = TriangleMesh(
                vertices=small_mesh.vertices + np.random.normal(0, 0.001, small_mesh.vertices.shape),
                triangles=small_mesh.triangles
            )
            
            result = calculator.compute_curvatures(mesh_copy)
            
            frame_time = (time.perf_counter() - start_time) * 1000
            computation_times.append(frame_time)
            
            # フレーム時間制限チェック
            if frame % 10 == 0:
                avg_time = np.mean(computation_times[-10:])
                print(f"フレーム {frame}: 平均 {avg_time:.1f}ms")
        
        # パフォーマンス統計
        avg_time = np.mean(computation_times)
        max_time = np.max(computation_times)
        p95_time = np.percentile(computation_times, 95)
        
        print(f"\n高頻度計算性能 (30FPS想定):")
        print(f"  平均フレーム時間: {avg_time:.1f}ms")
        print(f"  最大フレーム時間: {max_time:.1f}ms")
        print(f"  95%ile時間:      {p95_time:.1f}ms")
        print(f"  制限時間:        {frame_time_limit:.1f}ms")
        
        # 95%のフレームが制限時間内であることを確認
        within_limit = np.sum(np.array(computation_times) <= frame_time_limit)
        success_rate = within_limit / len(computation_times)
        
        print(f"  成功率:          {success_rate*100:.1f}%")
        
        assert success_rate >= 0.95, f"Too many slow frames: {success_rate*100:.1f}% success rate"
    
    def test_stress_test(self, large_mesh):
        """ストレステスト（長時間実行）"""
        
        calculator = VectorizedCurvatureCalculator()
        
        # 10分間の連続実行相当
        iterations = 100
        start_time = time.perf_counter()
        
        error_count = 0
        
        for i in range(iterations):
            try:
                result = calculator.compute_curvatures(large_mesh)
                
                # 結果の妥当性チェック
                assert result.vertex_curvatures.shape[0] == large_mesh.num_vertices
                assert not np.any(np.isnan(result.vertex_curvatures))
                assert not np.any(np.isinf(result.vertex_curvatures))
                
                if (i + 1) % 20 == 0:
                    elapsed = (time.perf_counter() - start_time) / 60
                    print(f"反復 {i+1}/{iterations} ({elapsed:.1f}分経過)")
                
            except Exception as e:
                error_count += 1
                print(f"エラー発生 (反復 {i+1}): {e}")
        
        total_time = (time.perf_counter() - start_time) / 60
        
        print(f"\nストレステスト結果:")
        print(f"  総実行時間: {total_time:.1f}分")
        print(f"  総反復数:   {iterations}")
        print(f"  エラー数:   {error_count}")
        print(f"  成功率:     {(iterations-error_count)/iterations*100:.1f}%")
        
        # エラー率が5%以下であることを確認
        error_rate = error_count / iterations
        assert error_rate <= 0.05, f"Too many errors: {error_rate*100:.1f}%"
        
        # 統計情報確認
        stats = calculator.get_performance_stats()
        print(f"  総計算回数: {stats.get('total_calculations', 0)}")
        print(f"  平均時間:   {stats.get('average_time_ms', 0):.1f}ms")

    def test_jit_vs_vectorized_performance_comparison(self, medium_mesh):
        """JIT最適化 vs ベクトル化版の性能比較"""
        
        # JIT最適化版
        calculator_jit = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
        
        # JITウォームアップ（初回コンパイル）
        warmup_mesh = TriangleMesh(
            vertices=medium_mesh.vertices[:100],
            triangles=medium_mesh.triangles[:50]
        )
        calculator_jit.compute_curvatures(warmup_mesh)
        
        # JIT版性能測定
        start_time = time.perf_counter()
        result_jit = calculator_jit.compute_curvatures(medium_mesh)
        jit_time = (time.perf_counter() - start_time) * 1000
        
        # ベクトル化版（JIT無効）
        calculator_vectorized = VectorizedCurvatureCalculator(use_jit=False, enable_caching=False)
        
        start_time = time.perf_counter()
        result_vectorized = calculator_vectorized.compute_curvatures(medium_mesh)
        vectorized_time = (time.perf_counter() - start_time) * 1000
        
        # パフォーマンス比較
        jit_speedup = vectorized_time / jit_time
        
        print(f"\nJIT vs ベクトル化 性能比較 (メッシュ: {medium_mesh.num_vertices}頂点):")
        print(f"  JIT最適化版:      {jit_time:.1f}ms")
        print(f"  ベクトル化版:      {vectorized_time:.1f}ms")
        print(f"  JIT高速化比:      {jit_speedup:.1f}x")
        
        # 期待値: JITで1.5x以上の高速化
        assert jit_speedup >= 1.5, f"Expected at least 1.5x JIT speedup, got {jit_speedup:.1f}x"
        
        # 結果の整合性確認
        np.testing.assert_allclose(
            result_jit.vertex_curvatures, 
            result_vectorized.vertex_curvatures, 
            rtol=1e-10, atol=1e-12
        )
        
        # JIT統計確認
        jit_stats = calculator_jit.get_performance_stats()
        assert jit_stats['jit_calculations'] > 0, "JIT calculations should have been used"
        
        vectorized_stats = calculator_vectorized.get_performance_stats()
        assert vectorized_stats['fallback_calculations'] > 0, "Fallback calculations should have been used"
    
    def test_jit_compilation_overhead(self, small_mesh):
        """JITコンパイル時間のオーバーヘッド測定"""
        
        # 初回実行（コンパイル時間込み）
        calculator = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
        
        start_time = time.perf_counter()
        first_result = calculator.compute_curvatures(small_mesh)
        first_time = (time.perf_counter() - start_time) * 1000
        
        # 2回目実行（コンパイル済み）
        start_time = time.perf_counter()
        second_result = calculator.compute_curvatures(small_mesh)
        second_time = (time.perf_counter() - start_time) * 1000
        
        # コンパイル効果
        compilation_overhead = first_time - second_time
        compilation_speedup = first_time / second_time
        
        print(f"\nJITコンパイル効果:")
        print(f"  初回実行時間:     {first_time:.1f}ms (コンパイル込み)")
        print(f"  2回目実行時間:    {second_time:.1f}ms (コンパイル済み)")
        print(f"  コンパイル時間:   {compilation_overhead:.1f}ms")
        print(f"  コンパイル後高速化: {compilation_speedup:.1f}x")
        
        # 結果一致確認
        np.testing.assert_array_equal(first_result.vertex_curvatures, second_result.vertex_curvatures)
        
        # コンパイル後は高速化されることを確認
        assert compilation_speedup >= 1.2, f"Expected speedup after compilation, got {compilation_speedup:.1f}x"
    
    def test_jit_parallel_performance(self, large_mesh):
        """JIT並列処理の性能確認"""
        
        # 並列JIT版
        calculator_parallel = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
        
        # ウォームアップ
        warmup_mesh = TriangleMesh(
            vertices=large_mesh.vertices[:500],
            triangles=large_mesh.triangles[:200]
        )
        calculator_parallel.compute_curvatures(warmup_mesh)
        
        # 並列処理性能測定
        start_time = time.perf_counter()
        result_parallel = calculator_parallel.compute_curvatures(large_mesh)
        parallel_time = (time.perf_counter() - start_time) * 1000
        
        # 頂点あたりの処理時間
        per_vertex_time = parallel_time / large_mesh.num_vertices
        
        print(f"\nJIT並列処理性能 (メッシュ: {large_mesh.num_vertices}頂点):")
        print(f"  総処理時間:       {parallel_time:.1f}ms")
        print(f"  頂点あたり:       {per_vertex_time:.4f}ms/vertex")
        
        # 性能目標: 頂点あたり0.005ms以下（JIT効果による目標向上）
        assert per_vertex_time <= 0.005, f"JIT per-vertex time too high: {per_vertex_time:.4f}ms/vertex"
        
        # 結果の妥当性確認
        assert result_parallel.vertex_curvatures.shape[0] == large_mesh.num_vertices
        assert not np.any(np.isnan(result_parallel.vertex_curvatures))
        assert not np.any(np.isinf(result_parallel.vertex_curvatures))
    
    def test_jit_memory_efficiency(self, medium_mesh):
        """JIT版のメモリ効率確認"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        calculator = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
        
        # 複数回実行してメモリリークをチェック
        memory_measurements = []
        
        for i in range(10):
            result = calculator.compute_curvatures(medium_mesh)
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)
        
        final_memory = memory_measurements[-1]
        max_memory = max(memory_measurements)
        memory_increase = final_memory - initial_memory
        
        print(f"\nJIT版メモリ効率:")
        print(f"  初期メモリ:       {initial_memory:.1f}MB")
        print(f"  最終メモリ:       {final_memory:.1f}MB")
        print(f"  最大メモリ:       {max_memory:.1f}MB")
        print(f"  メモリ増加:       {memory_increase:.1f}MB")
        
        # メモリ増加は50MB以下であることを確認
        assert memory_increase <= 50.0, f"Too much memory increase: {memory_increase:.1f}MB"
    
    def test_jit_fallback_behavior(self, small_mesh):
        """JIT無効時のフォールバック動作確認"""
        
        # Numba無効環境をシミュレート
        with patch('src.mesh.curvature_vectorized.NUMBA_AVAILABLE', False):
            calculator = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
            
            start_time = time.perf_counter()
            result_fallback = calculator.compute_curvatures(small_mesh)
            fallback_time = (time.perf_counter() - start_time) * 1000
            
            # フォールバック統計確認
            stats = calculator.get_performance_stats()
            assert stats['fallback_calculations'] > 0, "Should use fallback when Numba unavailable"
            assert stats['jit_calculations'] == 0, "Should not use JIT when unavailable"
        
        # JIT有効版と比較
        calculator_jit = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
        start_time = time.perf_counter()
        result_jit = calculator_jit.compute_curvatures(small_mesh)
        jit_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\nJITフォールバック動作:")
        print(f"  JIT有効:          {jit_time:.1f}ms")
        print(f"  フォールバック:   {fallback_time:.1f}ms")
        
        # 結果は一致する必要がある
        np.testing.assert_allclose(
            result_jit.vertex_curvatures, 
            result_fallback.vertex_curvatures, 
            rtol=1e-10, atol=1e-12
        )
    
    def test_jit_batch_processing_optimization(self, large_mesh):
        """JITバッチ処理最適化効果"""
        
        calculator = VectorizedCurvatureCalculator(use_jit=True, enable_caching=False)
        
        # 大規模バッチ処理
        batch_meshes = []
        for i in range(5):
            # 大規模メッシュから部分メッシュを作成
            vertex_subset = np.random.choice(large_mesh.num_vertices, size=1000, replace=False)
            triangle_subset = []
            
            for triangle in large_mesh.triangles:
                if all(v in vertex_subset for v in triangle):
                    # 新しいインデックスにマッピング
                    mapped_triangle = [np.where(vertex_subset == v)[0][0] for v in triangle]
                    triangle_subset.append(mapped_triangle)
            
            if len(triangle_subset) > 100:  # 十分な三角形がある場合のみ
                batch_mesh = TriangleMesh(
                    vertices=large_mesh.vertices[vertex_subset],
                    triangles=np.array(triangle_subset[:500])  # 最大500三角形
                )
                batch_meshes.append(batch_mesh)
        
        # バッチ処理時間測定
        start_time = time.perf_counter()
        
        batch_results = []
        for mesh in batch_meshes:
            result = calculator.compute_curvatures(mesh)
            batch_results.append(result)
        
        batch_time = (time.perf_counter() - start_time) * 1000
        
        # 統計
        total_vertices = sum(mesh.num_vertices for mesh in batch_meshes)
        per_vertex_batch_time = batch_time / total_vertices
        
        print(f"\nJITバッチ処理最適化:")
        print(f"  バッチ数:         {len(batch_meshes)}")
        print(f"  総頂点数:         {total_vertices}")
        print(f"  総処理時間:       {batch_time:.1f}ms")
        print(f"  頂点あたり:       {per_vertex_batch_time:.4f}ms/vertex")
        
        # バッチ処理での頂点あたり時間が目標以下
        assert per_vertex_batch_time <= 0.003, f"Batch per-vertex time too high: {per_vertex_batch_time:.4f}ms"
        
        # 全結果が妥当
        for result in batch_results:
            assert not np.any(np.isnan(result.vertex_curvatures))
            assert not np.any(np.isinf(result.vertex_curvatures))


@pytest.mark.performance
class TestResourceManagerPerformance:
    """リソース管理器パフォーマンステスト"""
    
    @pytest.mark.asyncio
    async def test_async_resource_manager_performance(self):
        """非同期リソース管理器の性能テスト"""
        from src.resource_manager_async import AsyncResourceManager, AsyncManagedResource
        
        class TestResource(AsyncManagedResource):
            def __init__(self, resource_id: str):
                super().__init__(resource_id)
                self.initialized = False
                self.cleaned = False
            
            async def initialize(self) -> bool:
                await asyncio.sleep(0.001)  # 1ms の初期化時間
                self.initialized = True
                return True
            
            async def cleanup(self) -> bool:
                await asyncio.sleep(0.001)  # 1ms のクリーンアップ時間
                self.cleaned = True
                return True
            
            def get_memory_usage(self) -> int:
                return 1024  # 1KB
            
            @property
            def resource_type(self) -> str:
                return "test_resource"
        
        manager = AsyncResourceManager()
        
        # 大量リソースの登録・削除性能テスト
        resource_count = 100
        
        # 登録性能
        start_time = time.perf_counter()
        
        resources = []
        for i in range(resource_count):
            resource = TestResource(f"test_resource_{i}")
            await manager.register_resource(resource)
            await resource.initialize()
            resources.append(resource)
        
        registration_time = (time.perf_counter() - start_time) * 1000
        
        # クリーンアップ性能
        start_time = time.perf_counter()
        
        cleanup_tasks = []
        for resource in resources:
            cleanup_tasks.append(manager.unregister_resource(resource.resource_id))
        
        await asyncio.gather(*cleanup_tasks)
        
        cleanup_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\n非同期リソース管理器性能:")
        print(f"  リソース数:   {resource_count}")
        print(f"  登録時間:     {registration_time:.1f}ms ({registration_time/resource_count:.2f}ms/resource)")
        print(f"  クリーンアップ時間: {cleanup_time:.1f}ms ({cleanup_time/resource_count:.2f}ms/resource)")
        
        # 性能要件チェック
        assert registration_time / resource_count <= 1.0, "Registration too slow"
        assert cleanup_time / resource_count <= 2.0, "Cleanup too slow"
        
        # 統計確認
        stats = manager.get_stats()
        print(f"  総作成数:     {stats['total_resources_created']}")
        print(f"  総クリーンアップ数: {stats['total_resources_cleaned']}")
        print(f"  アクティブ数: {stats['current_active_resources']}")
        
        assert stats['current_active_resources'] == 0, "Resources not properly cleaned up"
        
        await manager.shutdown()


if __name__ == "__main__":
    # 単体実行用
    pytest.main([__file__, "-v", "--tb=short"]) 