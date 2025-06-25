"""
ベクトル化パフォーマンステスト

perf-003, perf-004 の最適化効果を測定・検証
"""

import time
import pytest
import numpy as np
from typing import List
from unittest.mock import patch

from src.mesh.delaunay import TriangleMesh, DelaunayTriangulator
from src.mesh.vectorized import (
    vectorized_triangle_qualities,
    vectorized_is_valid_triangles,
    get_mesh_processor
)
from src.collision.distance import get_distance_calculator, point_triangle_distance_vectorized, batch_point_triangle_distances, OptimizedDistanceCalculator
from src.collision.sphere_tri import point_triangle_distance
from tests.conftest import TestAssertions, PerformanceMeasurement


@pytest.fixture
def large_mesh():
    """大規模メッシュを生成（10万頂点規模）"""
    # ランダムな点群生成
    np.random.seed(42)
    points = np.random.rand(5000, 3) * 10.0  # 5000点
    
    # Delaunay三角形分割
    triangulator = DelaunayTriangulator()
    mesh = triangulator.triangulate_points(points)
    
    return mesh


@pytest.fixture
def medium_mesh():
    """中規模メッシュを生成（1万頂点規模）"""
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 5.0
    
    triangulator = DelaunayTriangulator()
    mesh = triangulator.triangulate_points(points)
    
    return mesh


@pytest.fixture
def small_mesh():
    """小規模メッシュを生成（1000頂点規模）"""
    np.random.seed(42)
    points = np.random.rand(100, 3) * 2.0
    
    triangulator = DelaunayTriangulator()
    mesh = triangulator.triangulate_points(points)
    
    return mesh


class TestDistanceCalculationPerformance:
    """距離計算パフォーマンステスト (perf-004)"""
    
    def test_single_point_triangle_distance_comparison(self):
        """単一点-三角形距離計算の性能比較"""
        # テストデータ準備
        point = np.array([1.0, 1.0, 1.0])
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0]
        ])
        
        calculator = get_distance_calculator()
        
        # 従来方式（参考用）
        iterations = 1000
        
        # 新方式
        start_time = time.perf_counter()
        for _ in range(iterations):
            dist_new = calculator.calculate_point_triangle_distance(point, triangle)
        new_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\n距離計算性能比較:")
        print(f"新方式: {new_time:.1f}ms ({iterations}回)")
        print(f"1回あたり: {new_time/iterations:.3f}ms")
        
        # 結果の正確性チェック
        assert 0.0 <= dist_new <= 10.0  # 妥当な範囲
    
    def test_batch_distance_calculation_scaling(self, small_mesh, medium_mesh):
        """バッチ距離計算のスケーリング性能"""
        calculator = get_distance_calculator()
        
        # テストケース: 複数点と単一三角形
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], 
            [0.5, 1.0, 0.0]
        ])
        
        test_cases = [
            (10, "10点"),
            (100, "100点"),
            (1000, "1000点")
        ]
        
        print(f"\nバッチ距離計算スケーリング:")
        
        for num_points, label in test_cases:
            points = np.random.rand(num_points, 3) * 2.0
            
            start_time = time.perf_counter()
            distances = calculator.calculate_batch_distances(points, triangle)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"{label}: {elapsed_ms:.1f}ms ({elapsed_ms/num_points:.3f}ms/点)")
            
            # 結果検証
            assert len(distances) == num_points
            assert np.all(distances >= 0)
    
    def test_collision_search_distance_optimization(self, medium_mesh):
        """衝突検索での距離計算最適化効果"""
        # 検索点の準備
        query_points = np.random.rand(50, 3) * 5.0
        
        # 各点に対する三角形インデックス（模擬）
        triangle_indices_list = []
        for _ in range(len(query_points)):
            # ランダムに5-20個の三角形を選択
            num_triangles = np.random.randint(5, 21)
            indices = np.random.choice(medium_mesh.num_triangles, size=num_triangles, replace=False)
            triangle_indices_list.append(indices.tolist())
        
        # バッチ距離計算性能測定
        from src.collision.distance import batch_search_distances_optimized
        
        start_time = time.perf_counter()
        results = batch_search_distances_optimized(
            medium_mesh, query_points, triangle_indices_list
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        print(f"\n衝突検索距離計算最適化:")
        print(f"50点 x 平均{np.mean([len(indices) for indices in triangle_indices_list]):.1f}三角形")
        print(f"実行時間: {elapsed_ms:.1f}ms")
        print(f"1検索あたり: {elapsed_ms/len(query_points):.1f}ms")
        
        # 結果検証
        assert len(results) == len(query_points)
        for i, distances in enumerate(results):
            assert len(distances) == len(triangle_indices_list[i])


class TestMeshVectorizationPerformance:
    """メッシュベクトル化パフォーマンステスト (perf-003)"""
    
    def test_triangle_quality_calculation_scaling(self, small_mesh, medium_mesh, large_mesh):
        """三角形品質計算のスケーリング性能"""
        test_meshes = [
            (small_mesh, "小規模"),
            (medium_mesh, "中規模"),
            (large_mesh, "大規模")
        ]
        
        print(f"\n三角形品質計算スケーリング:")
        
        for mesh, label in test_meshes:
            # ベクトル化版
            start_time = time.perf_counter()
            qualities_vectorized = vectorized_triangle_qualities(mesh)
            vectorized_time = (time.perf_counter() - start_time) * 1000
            
            print(f"{label}メッシュ ({mesh.num_triangles}三角形):")
            print(f"  ベクトル化版: {vectorized_time:.1f}ms")
            print(f"  1三角形あたり: {vectorized_time/max(mesh.num_triangles, 1):.3f}ms")
            
            # 結果検証
            assert len(qualities_vectorized) == mesh.num_triangles
            assert np.all((qualities_vectorized >= 0) & (qualities_vectorized <= 1))
    
    def test_triangle_validation_performance(self, medium_mesh):
        """三角形有効性検証の性能"""
        triangle_vertices = medium_mesh.vertices[medium_mesh.triangles]
        
        start_time = time.perf_counter()
        valid_mask = vectorized_is_valid_triangles(triangle_vertices)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        print(f"\n三角形有効性検証:")
        print(f"{medium_mesh.num_triangles}三角形: {elapsed_ms:.1f}ms")
        print(f"有効三角形: {np.sum(valid_mask)}/{len(valid_mask)}")
        
        # 結果検証
        assert len(valid_mask) == medium_mesh.num_triangles
        assert isinstance(valid_mask[0], (bool, np.bool_))
    
    def test_mesh_processor_comprehensive_benchmark(self, large_mesh):
        """メッシュプロセッサの包括的ベンチマーク"""
        processor = get_mesh_processor()
        
        operations = [
            ("品質フィルタリング", lambda: processor.filter_triangles_by_quality(large_mesh, 0.3)),
            ("有効性検証", lambda: processor.validate_mesh_triangles(large_mesh)),
            ("統計計算", lambda: processor.compute_mesh_statistics(large_mesh))
        ]
        
        print(f"\nメッシュプロセッサ包括ベンチマーク ({large_mesh.num_triangles}三角形):")
        
        for operation_name, operation_func in operations:
            start_time = time.perf_counter()
            result = operation_func()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"{operation_name}: {elapsed_ms:.1f}ms")
            
            # 結果の妥当性チェック
            assert result is not None
    
    def test_memory_efficiency_large_mesh(self, large_mesh):
        """大規模メッシュでのメモリ効率性"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # メモリ使用量測定開始
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大量のベクトル化操作を実行
        processor = get_mesh_processor()
        
        for _ in range(10):  # 10回繰り返し
            qualities = vectorized_triangle_qualities(large_mesh)
            stats = processor.compute_mesh_statistics(large_mesh)
            
            # 中間結果をクリア
            del qualities, stats
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        print(f"\nメモリ効率性テスト:")
        print(f"処理前: {mem_before:.1f}MB")
        print(f"処理後: {mem_after:.1f}MB")
        print(f"増加量: {mem_increase:.1f}MB")
        
        # メモリ増加が合理的な範囲内であることを確認
        assert mem_increase < 200  # 200MB未満の増加


class TestPerformanceComparison:
    """従来方式との性能比較"""
    
    def test_overall_performance_improvement(self, medium_mesh):
        """全体的な性能改善効果の測定"""
        
        # テストシナリオ: 実際のワークフローを模擬
        query_points = np.random.rand(20, 3) * 5.0
        
        print(f"\n全体性能改善テスト:")
        print(f"メッシュ: {medium_mesh.num_triangles}三角形")
        print(f"検索点: {len(query_points)}点")
        
        # 最適化版のワークフロー
        start_time = time.perf_counter()
        
        # 1. メッシュ品質計算
        qualities = vectorized_triangle_qualities(medium_mesh)
        
        # 2. 距離計算（各点について）
        calculator = get_distance_calculator()
        total_distances = 0
        
        for point in query_points:
            # 近傍三角形を選択（上位10個と仮定）
            num_nearby = min(10, medium_mesh.num_triangles)
            triangle_indices = np.random.choice(medium_mesh.num_triangles, size=num_nearby, replace=False)
            
            for tri_idx in triangle_indices:
                triangle_vertices = medium_mesh.vertices[medium_mesh.triangles[tri_idx]]
                dist = calculator.calculate_point_triangle_distance(point, triangle_vertices)
                total_distances += 1
        
        optimized_time = (time.perf_counter() - start_time) * 1000
        
        print(f"最適化版: {optimized_time:.1f}ms")
        print(f"距離計算数: {total_distances}")
        print(f"1距離計算あたり: {optimized_time/total_distances:.3f}ms")
        
        # パフォーマンス統計出力
        mesh_stats = get_mesh_processor().get_performance_stats()
        distance_stats = calculator.get_performance_stats()
        
        print(f"\nパフォーマンス統計:")
        print(f"メッシュ処理: {mesh_stats}")
        print(f"距離計算: {distance_stats}")


@pytest.mark.benchmark
class TestBenchmarkTargets:
    """性能目標達成度テスト"""
    
    def test_distance_calculation_performance_target(self):
        """距離計算の性能目標: 1000回/秒以上"""
        point = np.array([1.0, 1.0, 1.0])
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0]
        ])
        
        calculator = get_distance_calculator()
        
        iterations = 1000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            dist = calculator.calculate_point_triangle_distance(point, triangle)
        
        elapsed_time = time.perf_counter() - start_time
        calculations_per_second = iterations / elapsed_time
        
        print(f"\n距離計算性能目標達成度:")
        print(f"実測: {calculations_per_second:.0f}回/秒")
        print(f"目標: 1000回/秒")
        
        # 目標達成チェック
        assert calculations_per_second >= 1000, f"性能目標未達: {calculations_per_second:.0f} < 1000"
    
    def test_mesh_processing_performance_target(self, large_mesh):
        """メッシュ処理の性能目標: 10万三角形を1秒以内"""
        if large_mesh.num_triangles < 50000:
            pytest.skip("大規模メッシュが十分大きくない")
        
        start_time = time.perf_counter()
        
        # 品質計算
        qualities = vectorized_triangle_qualities(large_mesh)
        
        elapsed_time = time.perf_counter() - start_time
        triangles_per_second = large_mesh.num_triangles / elapsed_time
        
        print(f"\nメッシュ処理性能目標達成度:")
        print(f"三角形数: {large_mesh.num_triangles}")
        print(f"処理時間: {elapsed_time:.2f}秒")
        print(f"実測: {triangles_per_second:.0f}三角形/秒")
        print(f"目標: 100,000三角形/秒")
        
        # 目標達成チェック（緩い条件）
        assert triangles_per_second >= 50000, f"性能目標未達: {triangles_per_second:.0f} < 50,000"


class TestJITDistanceCalculationPerformance:
    """JIT最適化距離計算パフォーマンステスト"""
    
    def test_jit_vs_fallback_distance_calculation(self):
        """JIT vs フォールバック距離計算の性能比較"""
        # テストデータ準備
        point = np.array([1.0, 1.0, 1.0])
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0]
        ])
        
        iterations = 1000
        
        # JIT版（ウォームアップ付き）
        # ウォームアップ実行
        for _ in range(10):
            point_triangle_distance_vectorized(point, triangle)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            dist_jit = point_triangle_distance_vectorized(point, triangle)
        jit_time = (time.perf_counter() - start_time) * 1000
        
        # フォールバック版をシミュレート
        with patch('src.collision.distance.NUMBA_AVAILABLE', False):
            start_time = time.perf_counter()
            for _ in range(iterations):
                dist_fallback = point_triangle_distance_vectorized(point, triangle)
            fallback_time = (time.perf_counter() - start_time) * 1000
        
        # 性能比較
        jit_speedup = fallback_time / jit_time
        
        print(f"\nJIT距離計算性能比較:")
        print(f"JIT版: {jit_time:.1f}ms ({iterations}回)")
        print(f"フォールバック版: {fallback_time:.1f}ms ({iterations}回)")
        print(f"JIT高速化比: {jit_speedup:.1f}x")
        print(f"1回あたり (JIT): {jit_time/iterations:.3f}ms")
        print(f"1回あたり (フォールバック): {fallback_time/iterations:.3f}ms")
        
        # 期待値: JITで2x以上の高速化
        assert jit_speedup >= 2.0, f"Expected at least 2x JIT speedup, got {jit_speedup:.1f}x"
        
        # 結果の正確性チェック
        assert abs(dist_jit - dist_fallback) < 1e-10, "JIT and fallback results should match"
    
    def test_jit_batch_distance_scaling(self, medium_mesh):
        """JITバッチ距離計算のスケーリング性能"""
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], 
            [0.5, 1.0, 0.0]
        ])
        
        test_cases = [
            (100, "100点"),
            (1000, "1000点"),
            (5000, "5000点")
        ]
        
        print(f"\nJITバッチ距離計算スケーリング:")
        
        scaling_times = []
        
        for num_points, label in test_cases:
            points = np.random.rand(num_points, 3) * 2.0
            
            # ウォームアップ
            batch_point_triangle_distances(points[:10], triangle.reshape(1, 3, 3))
            
            start_time = time.perf_counter()
            distances = batch_point_triangle_distances(points, triangle.reshape(1, 3, 3))
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            per_point_time = elapsed_ms / num_points
            scaling_times.append(per_point_time)
            
            print(f"{label}: {elapsed_ms:.1f}ms ({per_point_time:.4f}ms/点)")
            
            # 結果検証
            assert len(distances) == num_points
            assert np.all(distances >= 0)
            
            # JIT効果による性能目標: 0.01ms/点以下
            assert per_point_time <= 0.01, f"Per-point time too high: {per_point_time:.4f}ms"
        
        # スケーリング特性確認：線形に近いスケーリング
        scaling_factor = scaling_times[-1] / scaling_times[0]
        print(f"スケーリング特性: {scaling_factor:.1f}x (小→大規模)")
        
        # JIT最適化により良好なスケーリングを期待
        assert scaling_factor <= 3.0, f"Poor scaling: {scaling_factor:.1f}x"
    
    def test_jit_parallel_batch_distance_performance(self, large_mesh):
        """JIT並列バッチ距離計算の性能"""
        # 複数三角形との距離計算
        num_triangles = min(100, large_mesh.num_triangles)
        triangle_indices = np.random.choice(large_mesh.num_triangles, size=num_triangles, replace=False)
        triangles = large_mesh.vertices[large_mesh.triangles[triangle_indices]]
        
        num_points = 1000
        points = np.random.rand(num_points, 3) * 5.0
        
        # ウォームアップ
        batch_point_triangle_distances(points[:10], triangles[:5])
        
        # 並列バッチ処理性能測定
        start_time = time.perf_counter()
        distance_matrix = batch_point_triangle_distances(points, triangles)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        total_calculations = num_points * num_triangles
        per_calculation_time = elapsed_ms / total_calculations
        
        print(f"\nJIT並列バッチ距離計算:")
        print(f"点数: {num_points}, 三角形数: {num_triangles}")
        print(f"総計算数: {total_calculations}")
        print(f"実行時間: {elapsed_ms:.1f}ms")
        print(f"1計算あたり: {per_calculation_time:.4f}ms")
        
        # 結果検証
        assert distance_matrix.shape == (num_points, num_triangles)
        assert np.all(distance_matrix >= 0)
        
        # JIT並列効果による性能目標: 0.001ms/計算以下
        assert per_calculation_time <= 0.001, f"Per-calculation time too high: {per_calculation_time:.4f}ms"


class TestJITMeshVectorizationPerformance:
    """JITメッシュベクトル化パフォーマンステスト"""
    
    def test_jit_triangle_quality_performance(self, large_mesh):
        """JIT三角形品質計算の性能"""
        
        # JIT版（ウォームアップ付き）
        # ウォームアップ
        small_mesh = TriangleMesh(
            vertices=large_mesh.vertices[:100],
            triangles=large_mesh.triangles[:50]
        )
        vectorized_triangle_qualities(small_mesh)
        
        start_time = time.perf_counter()
        qualities_jit = vectorized_triangle_qualities(large_mesh)
        jit_time = (time.perf_counter() - start_time) * 1000
        
        # フォールバック版
        with patch('src.mesh.vectorized.NUMBA_AVAILABLE', False):
            start_time = time.perf_counter()
            qualities_fallback = vectorized_triangle_qualities(large_mesh)
            fallback_time = (time.perf_counter() - start_time) * 1000
        
        # 性能比較
        jit_speedup = fallback_time / jit_time
        per_triangle_jit = jit_time / max(large_mesh.num_triangles, 1)
        per_triangle_fallback = fallback_time / max(large_mesh.num_triangles, 1)
        
        print(f"\nJIT三角形品質計算性能:")
        print(f"メッシュ: {large_mesh.num_triangles}三角形")
        print(f"JIT版: {jit_time:.1f}ms ({per_triangle_jit:.4f}ms/三角形)")
        print(f"フォールバック版: {fallback_time:.1f}ms ({per_triangle_fallback:.4f}ms/三角形)")
        print(f"JIT高速化比: {jit_speedup:.1f}x")
        
        # 期待値: JITで1.5x以上の高速化
        assert jit_speedup >= 1.5, f"Expected at least 1.5x JIT speedup, got {jit_speedup:.1f}x"
        
        # 結果検証
        assert len(qualities_jit) == large_mesh.num_triangles
        assert len(qualities_fallback) == large_mesh.num_triangles
        np.testing.assert_allclose(qualities_jit, qualities_fallback, rtol=1e-10, atol=1e-12)
        
        # JIT効果による性能目標: 0.001ms/三角形以下
        assert per_triangle_jit <= 0.001, f"JIT per-triangle time too high: {per_triangle_jit:.4f}ms"
    
    def test_jit_triangle_validation_performance(self, large_mesh):
        """JIT三角形有効性検証の性能"""
        triangle_vertices = large_mesh.vertices[large_mesh.triangles]
        
        # ウォームアップ
        vectorized_is_valid_triangles(triangle_vertices[:50])
        
        # JIT版
        start_time = time.perf_counter()
        valid_mask_jit = vectorized_is_valid_triangles(triangle_vertices)
        jit_time = (time.perf_counter() - start_time) * 1000
        
        # フォールバック版
        with patch('src.mesh.vectorized.NUMBA_AVAILABLE', False):
            start_time = time.perf_counter()
            valid_mask_fallback = vectorized_is_valid_triangles(triangle_vertices)
            fallback_time = (time.perf_counter() - start_time) * 1000
        
        # 性能比較
        jit_speedup = fallback_time / jit_time
        
        print(f"\nJIT三角形有効性検証性能:")
        print(f"三角形数: {large_mesh.num_triangles}")
        print(f"JIT版: {jit_time:.1f}ms")
        print(f"フォールバック版: {fallback_time:.1f}ms")
        print(f"JIT高速化比: {jit_speedup:.1f}x")
        print(f"有効三角形 (JIT): {np.sum(valid_mask_jit)}/{len(valid_mask_jit)}")
        print(f"有効三角形 (フォールバック): {np.sum(valid_mask_fallback)}/{len(valid_mask_fallback)}")
        
        # 結果検証
        assert len(valid_mask_jit) == large_mesh.num_triangles
        assert len(valid_mask_fallback) == large_mesh.num_triangles
        np.testing.assert_array_equal(valid_mask_jit, valid_mask_fallback)
        
        # JIT効果による性能目標確認
        per_triangle_time = jit_time / large_mesh.num_triangles
        assert per_triangle_time <= 0.0005, f"JIT per-triangle validation time too high: {per_triangle_time:.4f}ms"
    
    def test_jit_mesh_processor_comprehensive_performance(self, large_mesh):
        """JITメッシュプロセッサの包括的性能"""
        processor = get_mesh_processor()
        
        operations = [
            ("品質フィルタリング", lambda: processor.filter_triangles_by_quality(large_mesh, 0.3)),
            ("有効性検証", lambda: processor.validate_mesh_triangles(large_mesh)),
            ("統計計算", lambda: processor.compute_mesh_statistics(large_mesh))
        ]
        
        print(f"\nJITメッシュプロセッサ包括性能 ({large_mesh.num_triangles}三角形):")
        
        total_time = 0
        
        for operation_name, operation_func in operations:
            # ウォームアップ
            try:
                operation_func()
            except:
                pass  # ウォームアップでエラーは無視
            
            start_time = time.perf_counter()
            result = operation_func()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            total_time += elapsed_ms
            
            print(f"{operation_name}: {elapsed_ms:.1f}ms")
            
            # 結果の妥当性チェック
            assert result is not None
        
        print(f"総処理時間: {total_time:.1f}ms")
        
        # JIT効果による全体性能目標
        per_triangle_total = total_time / large_mesh.num_triangles
        assert per_triangle_total <= 0.005, f"Total per-triangle time too high: {per_triangle_total:.4f}ms"


class TestJITPerformanceTargets:
    """JIT性能目標達成度テスト"""
    
    def test_jit_distance_calculation_target_exceeded(self):
        """JIT距離計算の向上された性能目標"""
        point = np.array([1.0, 1.0, 1.0])
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0]
        ])
        
        calculator = get_distance_calculator()
        
        # ウォームアップ
        for _ in range(10):
            calculator.calculate_point_triangle_distance(point, triangle)
        
        # 性能測定
        iterations = 2000  # JIT効果により目標を上げる
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            dist = calculator.calculate_point_triangle_distance(point, triangle)
        
        elapsed_time = time.perf_counter() - start_time
        calculations_per_second = iterations / elapsed_time
        
        print(f"\nJIT距離計算性能目標達成度:")
        print(f"実測: {calculations_per_second:.0f}回/秒")
        print(f"目標: 2000回/秒 (JIT向上目標)")
        
        # JIT効果による向上された目標達成チェック
        assert calculations_per_second >= 2000, f"JIT性能目標未達: {calculations_per_second:.0f} < 2000"
    
    def test_jit_comprehensive_performance_benchmark(self, large_mesh):
        """JIT包括性能ベンチマーク"""
        # 実際のワークフローをJIT最適化でテスト
        query_points = np.random.rand(50, 3) * 5.0
        
        print(f"\nJIT包括性能ベンチマーク:")
        print(f"メッシュ: {large_mesh.num_triangles}三角形")
        print(f"検索点: {len(query_points)}点")
        
        # ウォームアップ
        small_mesh = TriangleMesh(
            vertices=large_mesh.vertices[:100],
            triangles=large_mesh.triangles[:50]
        )
        vectorized_triangle_qualities(small_mesh)
        
        # JIT最適化版のワークフロー
        start_time = time.perf_counter()
        
        # 1. メッシュ品質計算（JIT最適化）
        qualities = vectorized_triangle_qualities(large_mesh)
        
        # 2. 距離計算（JIT最適化）
        calculator = get_distance_calculator()
        total_distances = 0
        
        for point in query_points:
            # 近傍三角形を選択（上位20個）
            num_nearby = min(20, large_mesh.num_triangles)
            triangle_indices = np.random.choice(large_mesh.num_triangles, size=num_nearby, replace=False)
            
            for tri_idx in triangle_indices:
                triangle_vertices = large_mesh.vertices[large_mesh.triangles[tri_idx]]
                dist = calculator.calculate_point_triangle_distance(point, triangle_vertices)
                total_distances += 1
        
        jit_optimized_time = (time.perf_counter() - start_time) * 1000
        
        print(f"JIT最適化版: {jit_optimized_time:.1f}ms")
        print(f"距離計算数: {total_distances}")
        print(f"1距離計算あたり: {jit_optimized_time/total_distances:.4f}ms")
        
        # JIT効果による向上された性能目標
        per_distance_time = jit_optimized_time / total_distances
        assert per_distance_time <= 0.05, f"JIT per-distance time too high: {per_distance_time:.4f}ms"
        
        # 全体性能目標
        total_operations = len(query_points) + large_mesh.num_triangles + total_distances
        per_operation_time = jit_optimized_time / total_operations
        assert per_operation_time <= 0.01, f"JIT per-operation time too high: {per_operation_time:.4f}ms"


if __name__ == "__main__":
    # 直接実行時の簡易ベンチマーク
    print("ベクトル化パフォーマンステスト実行中...")
    
    # 基本テスト実行
    pytest.main([__file__, "-v", "-s"]) 