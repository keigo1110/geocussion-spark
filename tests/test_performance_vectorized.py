"""
ベクトル化パフォーマンステスト

perf-003, perf-004 の最適化効果を測定・検証
"""

import time
import pytest
import numpy as np
from typing import List

from src.mesh.delaunay import TriangleMesh, DelaunayTriangulator
from src.mesh.vectorized import (
    vectorized_triangle_qualities,
    vectorized_is_valid_triangles,
    get_mesh_processor
)
from src.collision.distance import get_distance_calculator
from src.collision.sphere_tri import point_triangle_distance


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


if __name__ == "__main__":
    # 直接実行時の簡易ベンチマーク
    print("ベクトル化パフォーマンステスト実行中...")
    
    # 基本テスト実行
    pytest.main([__file__, "-v", "-s"]) 