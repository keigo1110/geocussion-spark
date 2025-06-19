#!/usr/bin/env python3
"""
地形メッシュ生成フェーズのテスト

点群投影、Delaunay三角形分割、メッシュ簡略化、属性計算、
空間インデックス構築の各コンポーネントをテストします。
"""

import unittest
import time
import numpy as np
import tempfile
import os

# テスト対象モジュール
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mesh import (
    # 投影
    PointCloudProjector, ProjectionMethod, HeightMap, create_height_map,
    # 三角形分割
    DelaunayTriangulator, TriangleMesh, create_mesh_from_heightmap,
    # 簡略化
    MeshSimplifier, SimplificationMethod, simplify_mesh,
    # 属性
    AttributeCalculator, MeshAttributes, compute_mesh_attributes,
    # インデックス
    SpatialIndex, IndexType, build_bvh_index
)


class TestPointCloudProjection(unittest.TestCase):
    """点群投影テスト"""
    
    def setUp(self):
        """テスト用データ生成"""
        # 簡単な山型の点群を生成
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        xx, yy = np.meshgrid(x, y)
        zz = np.exp(-(xx**2 + yy**2))  # ガウシアン山
        
        self.test_points = np.column_stack([
            xx.flatten(), yy.flatten(), zz.flatten()
        ])
        
        # ノイズを追加
        noise = np.random.normal(0, 0.05, self.test_points.shape)
        self.test_points += noise
    
    def test_projector_initialization(self):
        """プロジェクター初期化テスト"""
        projector = PointCloudProjector(resolution=0.1)
        self.assertEqual(projector.resolution, 0.1)
        self.assertEqual(projector.method, ProjectionMethod.MEAN_HEIGHT)
        self.assertTrue(projector.fill_holes)
    
    def test_basic_projection(self):
        """基本投影テスト"""
        projector = PointCloudProjector(resolution=0.2)
        heightmap = projector.project_points(self.test_points)
        
        self.assertIsInstance(heightmap, HeightMap)
        self.assertGreater(heightmap.width, 0)
        self.assertGreater(heightmap.height, 0)
        self.assertEqual(heightmap.heights.shape, heightmap.valid_mask.shape)
        
        # 有効な高度値が存在することを確認
        valid_heights = heightmap.heights[heightmap.valid_mask]
        self.assertGreater(len(valid_heights), 0)
        self.assertTrue(np.all(np.isfinite(valid_heights)))
    
    def test_projection_methods(self):
        """投影方式テスト"""
        methods = [
            ProjectionMethod.MIN_HEIGHT,
            ProjectionMethod.MAX_HEIGHT,
            ProjectionMethod.MEAN_HEIGHT,
            ProjectionMethod.MEDIAN_HEIGHT
        ]
        
        for method in methods:
            with self.subTest(method=method):
                projector = PointCloudProjector(resolution=0.2, method=method)
                heightmap = projector.project_points(self.test_points)
                
                self.assertIsInstance(heightmap, HeightMap)
                self.assertGreater(np.sum(heightmap.valid_mask), 0)
    
    def test_coordinate_conversion(self):
        """座標変換テスト"""
        projector = PointCloudProjector(resolution=0.1)
        heightmap = projector.project_points(self.test_points)
        
        # グリッド座標 -> 世界座標 -> グリッド座標
        for row in range(0, heightmap.height, 5):
            for col in range(0, heightmap.width, 5):
                world_x, world_y = heightmap.get_world_coordinates(row, col)
                back_row, back_col = heightmap.get_grid_coordinates(world_x, world_y)
                
                self.assertAlmostEqual(row, back_row, delta=1)
                self.assertAlmostEqual(col, back_col, delta=1)
    
    def test_performance(self):
        """パフォーマンステスト"""
        large_points = np.random.rand(10000, 3) * 2 - 1
        
        projector = PointCloudProjector(resolution=0.05)
        start_time = time.perf_counter()
        heightmap = projector.project_points(large_points)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # 50ms以内で完了することを確認
        self.assertLess(elapsed_ms, 50.0)
        
        # 統計確認
        stats = projector.get_performance_stats()
        self.assertGreater(stats['total_projections'], 0)
        self.assertGreater(stats['last_num_points'], 0)


class TestDelaunayTriangulation(unittest.TestCase):
    """Delaunay三角形分割テスト"""
    
    def setUp(self):
        """テスト用ハイトマップ生成"""
        # 格子状の点群
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        xx, yy = np.meshgrid(x, y)
        zz = xx**2 + yy**2  # 放物面
        
        points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        projector = PointCloudProjector(resolution=0.2)
        self.test_heightmap = projector.project_points(points)
    
    def test_triangulator_initialization(self):
        """三角分割器初期化テスト"""
        triangulator = DelaunayTriangulator(max_edge_length=0.5)
        self.assertEqual(triangulator.max_edge_length, 0.5)
        self.assertTrue(triangulator.adaptive_sampling)
        self.assertTrue(triangulator.boundary_points)
    
    def test_heightmap_triangulation(self):
        """ハイトマップ三角分割テスト"""
        triangulator = DelaunayTriangulator(max_edge_length=1.0)
        mesh = triangulator.triangulate_heightmap(self.test_heightmap)
        
        self.assertIsInstance(mesh, TriangleMesh)
        self.assertGreater(mesh.num_vertices, 0)
        self.assertGreater(mesh.num_triangles, 0)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.triangles.shape[1], 3)
        
        # 三角形インデックスが有効範囲内にあることを確認
        self.assertTrue(np.all(mesh.triangles >= 0))
        self.assertTrue(np.all(mesh.triangles < mesh.num_vertices))
    
    def test_points_triangulation(self):
        """点群三角分割テスト"""
        # 2D点
        points_2d = np.random.rand(50, 2) * 2 - 1
        
        triangulator = DelaunayTriangulator()
        mesh = triangulator.triangulate_points(points_2d)
        
        self.assertIsInstance(mesh, TriangleMesh)
        self.assertEqual(mesh.vertices.shape[1], 3)  # 3Dに拡張される
        self.assertGreater(mesh.num_triangles, 0)
    
    def test_triangle_quality(self):
        """三角形品質テスト"""
        triangulator = DelaunayTriangulator(quality_threshold=0.3)
        mesh = triangulator.triangulate_heightmap(self.test_heightmap)
        
        # 面積計算
        areas = mesh.get_triangle_areas()
        self.assertTrue(np.all(areas > 0))
        
        # 重心計算
        centers = mesh.get_triangle_centers()
        self.assertEqual(centers.shape, (mesh.num_triangles, 3))
        
        # バウンディングボックス
        min_bounds, max_bounds = mesh.get_bounds()
        self.assertEqual(len(min_bounds), 3)
        self.assertEqual(len(max_bounds), 3)
        self.assertTrue(np.all(min_bounds <= max_bounds))
    
    def test_performance(self):
        """パフォーマンステスト"""
        triangulator = DelaunayTriangulator()
        
        start_time = time.perf_counter()
        mesh = triangulator.triangulate_heightmap(self.test_heightmap)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # 20ms以内で完了することを確認
        self.assertLess(elapsed_ms, 20.0)
        
        # 統計確認
        stats = triangulator.get_performance_stats()
        self.assertGreater(stats['total_triangulations'], 0)
        self.assertGreater(stats['last_quality_score'], 0)


class TestMeshSimplification(unittest.TestCase):
    """メッシュ簡略化テスト"""
    
    def setUp(self):
        """テスト用メッシュ生成"""
        # 密なメッシュを生成
        x = np.linspace(-1, 1, 15)
        y = np.linspace(-1, 1, 15)
        xx, yy = np.meshgrid(x, y)
        zz = np.sin(xx * np.pi) * np.cos(yy * np.pi)
        
        points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        heightmap = create_height_map(points, resolution=0.15)
        self.test_mesh = create_mesh_from_heightmap(heightmap, max_edge_length=0.5)
    
    def test_simplifier_initialization(self):
        """簡略化器初期化テスト"""
        simplifier = MeshSimplifier(target_reduction=0.5)
        self.assertEqual(simplifier.target_reduction, 0.5)
        self.assertEqual(simplifier.method, SimplificationMethod.QUADRIC_ERROR)
        self.assertTrue(simplifier.preserve_boundary)
    
    def test_basic_simplification(self):
        """基本簡略化テスト"""
        original_triangles = self.test_mesh.num_triangles
        
        simplifier = MeshSimplifier(target_reduction=0.5)
        simplified_mesh = simplifier.simplify_mesh(self.test_mesh)
        
        self.assertIsInstance(simplified_mesh, TriangleMesh)
        self.assertLessEqual(simplified_mesh.num_triangles, original_triangles)
        self.assertGreater(simplified_mesh.num_triangles, 0)
        
        # バリデーション
        self.assertEqual(simplified_mesh.vertices.shape[1], 3)
        self.assertEqual(simplified_mesh.triangles.shape[1], 3)
    
    def test_simplification_methods(self):
        """簡略化手法テスト"""
        methods = [
            SimplificationMethod.QUADRIC_ERROR,
            SimplificationMethod.VERTEX_CLUSTERING,
            SimplificationMethod.EDGE_COLLAPSE
        ]
        
        for method in methods:
            with self.subTest(method=method):
                simplifier = MeshSimplifier(method=method, target_reduction=0.3)
                simplified_mesh = simplifier.simplify_mesh(self.test_mesh)
                
                self.assertIsInstance(simplified_mesh, TriangleMesh)
                self.assertGreater(simplified_mesh.num_triangles, 0)
    
    def test_target_count_simplification(self):
        """目標数簡略化テスト"""
        target_triangles = max(10, self.test_mesh.num_triangles // 3)
        
        simplifier = MeshSimplifier()
        simplified_mesh = simplifier.simplify_to_target_count(self.test_mesh, target_triangles)
        
        # 目標数に近いことを確認（±20%の範囲）
        self.assertLessEqual(simplified_mesh.num_triangles, target_triangles * 1.2)
        self.assertGreaterEqual(simplified_mesh.num_triangles, target_triangles * 0.8)
    
    def test_adaptive_simplification(self):
        """適応的簡略化テスト"""
        max_triangles = max(20, self.test_mesh.num_triangles // 2)
        
        simplifier = MeshSimplifier()
        simplified_mesh = simplifier.adaptive_simplify(self.test_mesh, max_triangles)
        
        self.assertLessEqual(simplified_mesh.num_triangles, max_triangles)
        self.assertGreater(simplified_mesh.num_triangles, 0)


class TestMeshAttributes(unittest.TestCase):
    """メッシュ属性テスト"""
    
    def setUp(self):
        """テスト用メッシュ生成"""
        # 球面の一部を生成
        phi = np.linspace(0, np.pi/2, 10)
        theta = np.linspace(0, np.pi, 10)
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        x = np.sin(phi_mesh) * np.cos(theta_mesh)
        y = np.sin(phi_mesh) * np.sin(theta_mesh)
        z = np.cos(phi_mesh)
        
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        heightmap = create_height_map(points, resolution=0.2)
        self.test_mesh = create_mesh_from_heightmap(heightmap)
    
    def test_calculator_initialization(self):
        """属性計算器初期化テスト"""
        calculator = AttributeCalculator(smooth_normals=True)
        self.assertTrue(calculator.smooth_normals)
        self.assertEqual(calculator.gradient_method, "finite_diff")
        self.assertTrue(calculator.normalize_attributes)
    
    def test_vertex_normals(self):
        """頂点法線テスト"""
        calculator = AttributeCalculator()
        normals = calculator.calculate_vertex_normals(self.test_mesh)
        
        self.assertEqual(normals.shape, (self.test_mesh.num_vertices, 3))
        
        # 法線の長さが1に正規化されていることを確認
        norms = np.linalg.norm(normals, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))
    
    def test_triangle_normals(self):
        """三角形法線テスト"""
        calculator = AttributeCalculator()
        normals = calculator.calculate_triangle_normals(self.test_mesh)
        
        self.assertEqual(normals.shape, (self.test_mesh.num_triangles, 3))
        
        # 法線の長さが1に正規化されていることを確認
        norms = np.linalg.norm(normals, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))
    
    def test_curvature_calculation(self):
        """曲率計算テスト"""
        calculator = AttributeCalculator()
        vertex_curvatures, gaussian_curvatures, mean_curvatures = calculator.calculate_curvatures(self.test_mesh)
        
        self.assertEqual(len(vertex_curvatures), self.test_mesh.num_vertices)
        self.assertEqual(len(gaussian_curvatures), self.test_mesh.num_vertices)
        self.assertEqual(len(mean_curvatures), self.test_mesh.num_vertices)
        
        # 曲率値が有限であることを確認
        self.assertTrue(np.all(np.isfinite(vertex_curvatures)))
        self.assertTrue(np.all(np.isfinite(gaussian_curvatures)))
        self.assertTrue(np.all(np.isfinite(mean_curvatures)))
    
    def test_gradient_calculation(self):
        """勾配計算テスト"""
        calculator = AttributeCalculator()
        gradients, gradient_magnitudes = calculator.calculate_gradients(self.test_mesh)
        
        self.assertEqual(gradients.shape, (self.test_mesh.num_vertices, 3))
        self.assertEqual(len(gradient_magnitudes), self.test_mesh.num_vertices)
        
        # 勾配値が有限であることを確認
        self.assertTrue(np.all(np.isfinite(gradients)))
        self.assertTrue(np.all(np.isfinite(gradient_magnitudes)))
        self.assertTrue(np.all(gradient_magnitudes >= 0))
    
    def test_complete_attributes(self):
        """全属性計算テスト"""
        calculator = AttributeCalculator()
        attributes = calculator.compute_attributes(self.test_mesh)
        
        self.assertIsInstance(attributes, MeshAttributes)
        self.assertEqual(attributes.num_vertices, self.test_mesh.num_vertices)
        self.assertEqual(attributes.num_triangles, self.test_mesh.num_triangles)
        
        # 統計計算
        surface_roughness = attributes.get_surface_roughness()
        self.assertIsInstance(surface_roughness, float)
        self.assertGreaterEqual(surface_roughness, 0)
        
        curvature_stats = attributes.get_curvature_statistics()
        self.assertIn('mean_curvature_avg', curvature_stats)
        self.assertIn('max_curvature', curvature_stats)


class TestSpatialIndex(unittest.TestCase):
    """空間インデックステスト"""
    
    def setUp(self):
        """テスト用メッシュ生成"""
        # グリッドベースのメッシュ
        x = np.linspace(-2, 2, 12)
        y = np.linspace(-2, 2, 12)
        xx, yy = np.meshgrid(x, y)
        zz = xx**2 + yy**2
        
        points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        heightmap = create_height_map(points, resolution=0.4)
        self.test_mesh = create_mesh_from_heightmap(heightmap)
    
    def test_bvh_index_creation(self):
        """BVHインデックス作成テスト"""
        index = SpatialIndex(self.test_mesh, index_type=IndexType.BVH)
        
        self.assertEqual(index.index_type, IndexType.BVH)
        self.assertIsNotNone(index.root_node)
        self.assertIsNotNone(index.triangle_centers)
        
        # 統計確認
        stats = index.get_performance_stats()
        self.assertGreater(stats['build_time_ms'], 0)
        self.assertGreater(stats['num_nodes'], 0)
    
    def test_kdtree_index_creation(self):
        """KD-Treeインデックス作成テスト"""
        index = SpatialIndex(self.test_mesh, index_type=IndexType.KDTREE)
        
        self.assertEqual(index.index_type, IndexType.KDTREE)
        self.assertIsNotNone(index.kdtree)
        self.assertIsNotNone(index.triangle_centers)
    
    def test_point_query(self):
        """点検索テスト"""
        index = SpatialIndex(self.test_mesh, index_type=IndexType.BVH)
        
        # メッシュ中心付近を検索
        center_point = np.array([0.0, 0.0, 0.5])
        nearby_triangles = index.query_point(center_point, radius=1.0)
        
        self.assertIsInstance(nearby_triangles, list)
        self.assertGreater(len(nearby_triangles), 0)
        
        # 全ての三角形インデックスが有効範囲内にあることを確認
        for tri_idx in nearby_triangles:
            self.assertGreaterEqual(tri_idx, 0)
            self.assertLess(tri_idx, self.test_mesh.num_triangles)
    
    def test_sphere_query(self):
        """球検索テスト"""
        index = SpatialIndex(self.test_mesh, index_type=IndexType.BVH)
        
        center = np.array([1.0, 1.0, 2.0])
        radius = 0.5
        
        triangles = index.query_sphere(center, radius)
        self.assertIsInstance(triangles, list)
        
        # 半径を大きくすると結果が増えることを確認
        larger_triangles = index.query_sphere(center, radius * 2)
        self.assertGreaterEqual(len(larger_triangles), len(triangles))
    
    def test_ray_query(self):
        """レイ検索テスト"""
        index = SpatialIndex(self.test_mesh, index_type=IndexType.BVH)
        
        origin = np.array([0.0, 0.0, 5.0])
        direction = np.array([0.0, 0.0, -1.0])  # 下向き
        
        triangles = index.query_ray(origin, direction, max_distance=10.0)
        self.assertIsInstance(triangles, list)
        self.assertGreater(len(triangles), 0)
    
    def test_query_performance(self):
        """検索パフォーマンステスト"""
        index = SpatialIndex(self.test_mesh, index_type=IndexType.BVH)
        
        # 複数回検索してパフォーマンスを測定
        query_points = np.random.rand(100, 3) * 4 - 2
        
        start_time = time.perf_counter()
        for point in query_points:
            triangles = index.query_point(point, radius=0.5)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # 平均検索時間が1ms以下であることを確認
        avg_query_time = elapsed_ms / len(query_points)
        self.assertLess(avg_query_time, 1.0)
        
        # 統計確認
        stats = index.get_performance_stats()
        self.assertGreater(stats['total_queries'], 0)
        self.assertGreater(stats['average_query_time_ms'], 0)


class TestMeshIntegration(unittest.TestCase):
    """統合テスト"""
    
    def test_full_pipeline(self):
        """完全パイプラインテスト"""
        # 1. 点群生成
        np.random.seed(42)  # 再現性のため
        x = np.linspace(-2, 2, 25)
        y = np.linspace(-2, 2, 25)
        xx, yy = np.meshgrid(x, y)
        zz = np.sin(xx) * np.cos(yy) + np.random.normal(0, 0.1, xx.shape)
        
        points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        pipeline_start = time.perf_counter()
        
        # 2. 投影
        projector = PointCloudProjector(resolution=0.2)
        heightmap = projector.project_points(points)
        
        # 3. 三角形分割
        triangulator = DelaunayTriangulator(max_edge_length=0.5)
        mesh = triangulator.triangulate_heightmap(heightmap)
        
        # 4. 簡略化
        simplifier = MeshSimplifier(target_reduction=0.4)
        simplified_mesh = simplifier.simplify_mesh(mesh)
        
        # 5. 属性計算
        calculator = AttributeCalculator()
        attributes = calculator.compute_attributes(simplified_mesh)
        
        # 6. 空間インデックス
        index = SpatialIndex(simplified_mesh, index_type=IndexType.BVH)
        
        total_time = (time.perf_counter() - pipeline_start) * 1000
        
        # パフォーマンス要件: 15ms以内
        self.assertLess(total_time, 15.0)
        
        # 結果検証
        self.assertIsInstance(heightmap, HeightMap)
        self.assertIsInstance(mesh, TriangleMesh)
        self.assertIsInstance(simplified_mesh, TriangleMesh)
        self.assertIsInstance(attributes, MeshAttributes)
        self.assertIsInstance(index, SpatialIndex)
        
        # 各段階で妥当なデータが生成されていることを確認
        self.assertGreater(heightmap.width * heightmap.height, 0)
        self.assertGreater(mesh.num_triangles, 0)
        self.assertLessEqual(simplified_mesh.num_triangles, mesh.num_triangles)
        self.assertEqual(attributes.num_vertices, simplified_mesh.num_vertices)
        self.assertGreater(index.stats['num_nodes'], 0)
        
        print(f"Full pipeline completed in {total_time:.2f}ms")
        print(f"  - Original mesh: {mesh.num_triangles} triangles")
        print(f"  - Simplified mesh: {simplified_mesh.num_triangles} triangles")
        print(f"  - BVH nodes: {index.stats['num_nodes']}")
    
    def test_pipeline_stability(self):
        """パイプライン安定性テスト"""
        # 複数回実行して安定性を確認
        results = []
        
        for run in range(5):
            try:
                # ランダムデータで実行
                np.random.seed(100 + run)
                points = np.random.rand(200, 3) * 4 - 2
                points[:, 2] = np.sin(points[:, 0]) * np.cos(points[:, 1])
                
                # パイプライン実行
                heightmap = create_height_map(points, resolution=0.3)
                mesh = create_mesh_from_heightmap(heightmap)
                simplified_mesh = simplify_mesh(mesh, target_reduction=0.5)
                attributes = compute_mesh_attributes(simplified_mesh)
                index = build_bvh_index(simplified_mesh)
                
                results.append({
                    'heightmap_size': heightmap.width * heightmap.height,
                    'mesh_triangles': mesh.num_triangles,
                    'simplified_triangles': simplified_mesh.num_triangles,
                    'attributes_vertices': attributes.num_vertices,
                    'index_nodes': index.stats['num_nodes']
                })
                
            except Exception as e:
                self.fail(f"Pipeline failed on run {run}: {e}")
        
        # 全実行が成功したことを確認
        self.assertEqual(len(results), 5)
        
        # 結果の妥当性を確認
        for result in results:
            self.assertGreater(result['mesh_triangles'], 0)
            self.assertGreater(result['simplified_triangles'], 0)
            self.assertGreater(result['attributes_vertices'], 0)
            self.assertGreater(result['index_nodes'], 0)


if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2) 