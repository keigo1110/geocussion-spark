#!/usr/bin/env python3
"""
統合パフォーマンステスト

collision_performance_test.pyとmemory_optimization_test.pyの
パフォーマンステストを統合し、print()をlogging/assertに変更します。
"""

import pytest
import logging
import numpy as np
from typing import List
import time

from src.collision.search import CollisionSearcher
from src.collision.sphere_tri import SphereTriangleCollision
from src.collision.events import CollisionEventQueue, process_collision_events
from src.collision.optimization import ArrayPool, memory_efficient_context
from src.mesh.delaunay import TriangleMesh
from src.mesh.index import SpatialIndex, IndexType
from src.detection.tracker import TrackedHand
from src.types import SearchResult, CollisionInfo, Hand3D, HandednessType, TrackingState
from src.constants import COLLISION_SEARCH_TIME_LIMIT_MS, COLLISION_DETECTION_TIME_LIMIT_MS


@pytest.mark.performance
@pytest.mark.slow
class TestUnifiedPerformance:
    """統合パフォーマンステストスイート"""
    
    def test_search_performance_targets(self, test_logger, performance_tracker, assert_helper, mesh_test_data):
        """空間検索パフォーマンス目標テスト"""
        test_logger.info("=== 空間検索パフォーマンステスト開始 ===")
        
        complexities = ["simple", "medium", "complex"]
        target_time_ms = 2.0  # 目標: 2ms以内
        
        for complexity in complexities:
            vertices, triangles = mesh_test_data(complexity)
            mesh = TriangleMesh(vertices, triangles)
            spatial_index = SpatialIndex.create_bvh_index(mesh)
            searcher = CollisionSearcher(spatial_index, default_radius=0.1)
            
            test_logger.info(f"{complexity.upper()}メッシュ ({mesh.num_triangles}三角形):")
            
            # テスト用手位置生成
            hand_positions = [
                np.array([0.5, 0.5, 0.5]),
                np.array([0.1, 0.1, 0.8]),
                np.array([0.9, 0.9, 0.2]),
                np.array([0.2, 0.8, 0.6]),
                np.array([0.8, 0.2, 0.4])
            ]
            
            search_times = []
            for i, position in enumerate(hand_positions):
                # モックTrackedHand作成
                tracked_hand = TrackedHand(
                    hand_id=f"test_hand_{i}",
                    position=position,
                    velocity=np.array([0.0, 0.0, 0.0]),
                    tracking_state=TrackingState.STABLE
                )
                
                performance_tracker.start()
                result = searcher.search_near_hand(tracked_hand)
                measurement = performance_tracker.stop(1)
                
                search_times.append(measurement.execution_time_ms)
                test_logger.info(f"  手 {i+1}: {measurement.execution_time_ms:.3f}ms, "
                               f"見つかった三角形: {len(result.triangle_indices)}")
            
            avg_time = np.mean(search_times)
            max_time = np.max(search_times)
            
            test_logger.info(f"  平均時間: {avg_time:.3f}ms")
            test_logger.info(f"  最大時間: {max_time:.3f}ms")
            
            # アサーション（print()の代替）
            assert_helper.assert_performance_target(
                type('Measurement', (), {'execution_time_ms': max_time})(),
                target_time_ms,
                f"{complexity}メッシュ空間検索"
            )
            
            test_logger.info(f"  目標達成: ✓ (目標: {target_time_ms}ms以内)")
    
    def test_collision_detection_performance(self, test_logger, performance_tracker, assert_helper, mesh_test_data):
        """衝突判定パフォーマンス目標テスト"""
        test_logger.info("=== 衝突判定パフォーマンステスト開始 ===")
        
        vertices, triangles = mesh_test_data("medium")
        mesh = TriangleMesh(vertices, triangles)
        spatial_index = SpatialIndex.create_bvh_index(mesh)
        collision_detector = SphereTriangleCollision(mesh)
        searcher = CollisionSearcher(spatial_index)
        
        # テスト球位置生成（衝突しやすい位置）
        sphere_positions = [
            np.array([0.5, 0.1, 0.5]),  # メッシュ表面近く
            np.array([0.3, 0.05, 0.7]),
            np.array([0.7, 0.08, 0.3]),
            np.array([0.1, 0.12, 0.9]),
            np.array([0.9, 0.06, 0.1])
        ]
        sphere_radius = 0.05
        
        collision_times = []
        collisions = 0
        target_time_ms = 2.0
        
        for i, sphere_center in enumerate(sphere_positions):
            # 近傍検索
            tracked_hand = TrackedHand(
                hand_id=f"collision_test_{i}",
                position=sphere_center,
                velocity=np.array([0.0, 0.0, 0.0]),
                tracking_state=TrackingState.STABLE
            )
            search_result = searcher.search_near_hand(tracked_hand)
            
            performance_tracker.start()
            collision_info = collision_detector.test_sphere_collision(
                sphere_center, sphere_radius, search_result
            )
            measurement = performance_tracker.stop(1)
            
            collision_times.append(measurement.execution_time_ms)
            
            if collision_info.has_collision:
                collisions += 1
            
            test_logger.info(f"  球 {i+1}: {measurement.execution_time_ms:.3f}ms, "
                           f"衝突: {'あり' if collision_info.has_collision else 'なし'}")
        
        avg_time = np.mean(collision_times)
        max_time = np.max(collision_times)
        
        test_logger.info(f"  平均時間: {avg_time:.3f}ms")
        test_logger.info(f"  最大時間: {max_time:.3f}ms")
        test_logger.info(f"  衝突検出数: {collisions}/{len(sphere_positions)}")
        
        # パフォーマンス目標アサーション
        assert_helper.assert_performance_target(
            type('Measurement', (), {'execution_time_ms': max_time})(),
            target_time_ms,
            "衝突判定"
        )
        
        test_logger.info(f"  目標達成: ✓ (目標: {target_time_ms}ms以内)")
    
    def test_event_generation_performance(self, test_logger, performance_tracker, assert_helper):
        """イベント生成パフォーマンステスト"""
        test_logger.info("=== イベント生成パフォーマンステスト開始 ===")
        
        event_queue = CollisionEventQueue()
        target_time_ms = 1.0
        
        # テスト用衝突情報生成
        test_collisions = []
        for i in range(10):
            collision_info = CollisionInfo(
                has_collision=True,
                contact_points=[],
                closest_point=None,
                total_penetration_depth=0.01,
                collision_normal=np.array([0.0, 1.0, 0.0]),
                collision_time_ms=0.5
            )
            test_collisions.append(collision_info)
        
        event_times = []
        events_created = 0
        
        for i, collision_info in enumerate(test_collisions):
            hand_position = np.array([0.5 + i * 0.05, 0.1, 0.5])
            hand_velocity = np.array([0.1, 0.0, 0.0])
            
            performance_tracker.start()
            event = event_queue.create_event(
                collision_info, f"test_hand_{i}", hand_position, hand_velocity
            )
            measurement = performance_tracker.stop(1)
            
            event_times.append(measurement.execution_time_ms)
            
            if event is not None:
                events_created += 1
                
            test_logger.info(f"  イベント {i+1}: {measurement.execution_time_ms:.3f}ms")
        
        avg_time = np.mean(event_times)
        max_time = np.max(event_times)
        
        test_logger.info(f"  平均時間: {avg_time:.3f}ms")
        test_logger.info(f"  最大時間: {max_time:.3f}ms")
        test_logger.info(f"  作成イベント数: {events_created}/{len(test_collisions)}")
        
        # パフォーマンス目標アサーション
        assert_helper.assert_performance_target(
            type('Measurement', (), {'execution_time_ms': max_time})(),
            target_time_ms,
            "イベント生成"
        )
        
        test_logger.info(f"  目標達成: ✓ (目標: {target_time_ms}ms以内)")
    
    @pytest.mark.benchmark
    def test_array_copy_optimization_benchmark(self, test_logger, performance_tracker, assert_helper):
        """配列コピー最適化ベンチマーク"""
        test_logger.info("=== 配列コピー最適化ベンチマーク開始 ===")
        
        # 大きな配列でテスト
        large_array = np.random.random((1000, 3)).astype(np.float32)
        iterations = 1000
        
        # 従来方式（コピー多用）
        performance_tracker.start()
        for _ in range(iterations):
            copied_array = large_array.copy()
            result = np.linalg.norm(copied_array, axis=1)
        traditional_measurement = performance_tracker.stop(iterations)
        
        # 最適化版（インプレース操作）
        performance_tracker.start()
        with memory_efficient_context() as ctx:
            pool = ctx['pool']
            for _ in range(iterations):
                with pool.temporary_array(large_array.shape, large_array.dtype) as temp_array:
                    temp_array[:] = large_array
                    result = np.linalg.norm(temp_array, axis=1)
        optimized_measurement = performance_tracker.stop(iterations)
        
        # 改善率計算
        improvement = ((traditional_measurement.execution_time_ms - optimized_measurement.execution_time_ms) 
                      / traditional_measurement.execution_time_ms * 100)
        
        test_logger.info(f"配列コピーパフォーマンス比較:")
        test_logger.info(f"従来方式: {traditional_measurement.execution_time_ms:.3f}ms")
        test_logger.info(f"最適化版: {optimized_measurement.execution_time_ms:.3f}ms")
        test_logger.info(f"改善率: {improvement:.1f}%")
        
        # 最低限の改善を検証
        assert improvement > 10.0, f"配列コピー最適化の改善率が不十分: {improvement:.1f}% < 10%"
        
        test_logger.info("✓ 配列コピー最適化が有効に機能")
    
    def test_memory_usage_optimization(self, test_logger, performance_tracker, assert_helper):
        """メモリ使用量最適化テスト"""
        test_logger.info("=== メモリ使用量最適化テスト開始 ===")
        
        # 従来方式（多数のコピー）
        performance_tracker.start()
        arrays = []
        for _ in range(100):
            arr = np.random.random((100, 3)).astype(np.float32)
            arrays.append(arr.copy())  # 毎回コピー
        traditional_measurement = performance_tracker.stop()
        traditional_memory = traditional_measurement.memory_usage_mb
        del arrays
        
        # 最適化版（プール活用）
        performance_tracker.start()
        with memory_efficient_context() as ctx:
            pool = ctx['pool']
            for _ in range(100):
                with pool.temporary_array((100, 3), 'float32') as temp_array:
                    temp_array[:] = np.random.random((100, 3)).astype(np.float32)
        optimized_measurement = performance_tracker.stop()
        optimized_memory = optimized_measurement.memory_usage_mb
        
        # メモリ使用量の改善（負の値は削減を意味）
        memory_improvement = ((traditional_memory - optimized_memory) / abs(traditional_memory) * 100) if traditional_memory != 0 else 0
        
        test_logger.info(f"メモリ使用量比較:")
        test_logger.info(f"従来方式: {traditional_memory:.1f}MB")
        test_logger.info(f"最適化版: {optimized_memory:.1f}MB")
        
        if memory_improvement > 0:
            test_logger.info(f"メモリ削減率: {memory_improvement:.1f}%")
        else:
            test_logger.info(f"メモリ増加: {abs(memory_improvement):.1f}%（プール初期化のため）")
        
        # メモリ効率の検証（プールの場合、初期化コストがあるため緩い条件）
        assert optimized_memory < traditional_memory * 2.0, "メモリ使用量が予想以上に増加"
        
        test_logger.info("✓ メモリ最適化システムが正常に機能")
    
    def test_full_collision_pipeline_performance(self, test_logger, performance_tracker, assert_helper, mesh_test_data):
        """フルパイプラインパフォーマンステスト"""
        test_logger.info("=== フルパイプライン統合パフォーマンステスト開始 ===")
        
        # メッシュとインデックス構築
        vertices, triangles = mesh_test_data("medium")
        mesh = TriangleMesh(vertices, triangles)
        spatial_index = SpatialIndex.create_bvh_index(mesh)
        collision_detector = SphereTriangleCollision(mesh)
        searcher = CollisionSearcher(spatial_index)
        
        # 複数フレームのシミュレーション
        num_frames = 10
        target_frame_time_ms = 5.0  # フレーム目標: 5ms以内
        
        frame_times = []
        total_events = 0
        
        for frame in range(num_frames):
            # フレーム毎の手位置生成
            hands = []
            for hand_id in range(3):  # 3つの手をシミュレート
                position = np.array([
                    0.3 + hand_id * 0.2,
                    0.1 + frame * 0.01,
                    0.5 + hand_id * 0.1
                ])
                hands.append(TrackedHand(
                    hand_id=f"hand_{hand_id}",
                    position=position,
                    velocity=np.array([0.0, 0.01, 0.0]),
                    tracking_state=TrackingState.STABLE
                ))
            
            performance_tracker.start()
            
            # フレーム処理
            frame_events = 0
            for hand in hands:
                # 1. 空間検索
                search_result = searcher.search_near_hand(hand)
                
                # 2. 衝突判定
                collision_info = collision_detector.test_sphere_collision(
                    hand.position, 0.02, search_result
                )
                
                # 3. イベント生成
                if collision_info.has_collision:
                    frame_events += 1
            
            frame_measurement = performance_tracker.stop()
            frame_times.append(frame_measurement.execution_time_ms)
            total_events += frame_events
            
            test_logger.info(f"  フレーム {frame+1}: {frame_measurement.execution_time_ms:.3f}ms, "
                           f"イベント: {frame_events}")
        
        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)
        
        test_logger.info(f"  平均フレーム時間: {avg_frame_time:.3f}ms")
        test_logger.info(f"  最大フレーム時間: {max_frame_time:.3f}ms")
        test_logger.info(f"  総イベント数: {total_events}")
        
        # フレーム時間目標のアサーション
        assert_helper.assert_performance_target(
            type('Measurement', (), {'execution_time_ms': max_frame_time})(),
            target_frame_time_ms,
            "フルパイプライン"
        )
        
        test_logger.info(f"  目標達成: ✓ (目標: {target_frame_time_ms}ms以内)")
        
        # コンポーネント統計
        search_stats = searcher.get_performance_stats()
        collision_stats = collision_detector.get_performance_stats()
        
        test_logger.info(f"  コンポーネント統計:")
        test_logger.info(f"    検索: 平均 {search_stats.get('average_search_time_ms', 0):.3f}ms")
        test_logger.info(f"    衝突: 平均 {collision_stats.get('average_test_time_ms', 0):.3f}ms")
        
        # 全体目標の検証
        assert avg_frame_time <= target_frame_time_ms, (
            f"平均フレーム時間が目標を超過: {avg_frame_time:.3f}ms > {target_frame_time_ms}ms"
        )
        
        test_logger.info("✓ フルパイプラインが性能目標を達成")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 