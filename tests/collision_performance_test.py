#!/usr/bin/env python3
"""
衝突検出フェーズのパフォーマンステスト

5ms予算内での処理時間とスループットを検証します。
"""

import time
import numpy as np
from typing import List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.collision.search import CollisionSearcher, SearchResult
from src.collision.sphere_tri import SphereTriangleCollision, CollisionInfo, ContactPoint, CollisionType
from src.collision.events import CollisionEventQueue, CollisionEvent
from src.mesh.delaunay import TriangleMesh
from src.mesh.index import SpatialIndex, IndexType


class MockTrackedHand:
    """テスト用の手トラッキングデータ"""
    def __init__(self, hand_id: str, position: np.ndarray, velocity: np.ndarray = None):
        self.hand_id = hand_id
        self.position = position
        self.velocity = velocity if velocity is not None else np.zeros(3)


def create_test_mesh(complexity: str = "medium") -> TriangleMesh:
    """テスト用メッシュを作成"""
    if complexity == "simple":
        # シンプルなメッシュ（9頂点、8三角形）
        vertices = np.array([
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0], [0.5, 0.5, 0.1], [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0]
        ])
        triangles = np.array([
            [0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4],
            [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]
        ])
    elif complexity == "medium":
        # 中程度のメッシュ（25頂点、32三角形）
        x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        z = 0.1 * np.sin(x * np.pi) * np.sin(y * np.pi)
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        triangles = []
        for i in range(4):
            for j in range(4):
                idx = i * 5 + j
                triangles.extend([
                    [idx, idx + 1, idx + 5],
                    [idx + 1, idx + 6, idx + 5]
                ])
        triangles = np.array(triangles)
    else:  # complex
        # 複雑なメッシュ（100頂点、162三角形）
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        z = 0.2 * (np.sin(x * 2 * np.pi) * np.cos(y * 2 * np.pi) + 
                   0.5 * np.sin(x * 4 * np.pi) * np.sin(y * 4 * np.pi))
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        triangles = []
        for i in range(9):
            for j in range(9):
                idx = i * 10 + j
                triangles.extend([
                    [idx, idx + 1, idx + 10],
                    [idx + 1, idx + 11, idx + 10]
                ])
        triangles = np.array(triangles)
    
    return TriangleMesh(vertices=vertices, triangles=triangles)


def test_search_performance():
    """空間検索のパフォーマンステスト"""
    print("\n=== 空間検索パフォーマンステスト ===")
    
    # 異なる複雑度のメッシュでテスト
    complexities = ["simple", "medium", "complex"]
    
    for complexity in complexities:
        mesh = create_test_mesh(complexity)
        spatial_index = SpatialIndex(mesh, index_type=IndexType.BVH)
        searcher = CollisionSearcher(spatial_index, default_radius=0.1)
        
        print(f"\n{complexity.upper()}メッシュ ({mesh.num_triangles}三角形):")
        
        # 複数の手位置でテスト
        hand_positions = [
            np.array([0.2, 0.2, 0.1]),
            np.array([0.5, 0.5, 0.1]),
            np.array([0.8, 0.8, 0.1]),
            np.array([0.3, 0.7, 0.1]),
            np.array([0.7, 0.3, 0.1])
        ]
        
        times = []
        for i, position in enumerate(hand_positions):
            hand = MockTrackedHand(f"hand_{i}", position)
            
            start_time = time.perf_counter()
            result = searcher.search_near_hand(hand, override_radius=0.3)
            elapsed = (time.perf_counter() - start_time) * 1000
            
            times.append(elapsed)
            print(f"  手 {i+1}: {elapsed:.3f}ms, 見つかった三角形: {len(result.triangle_indices)}")
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        print(f"  平均時間: {avg_time:.3f}ms")
        print(f"  最大時間: {max_time:.3f}ms")
        print(f"  目標達成: {'✓' if max_time < 2.0 else '✗'} (目標: 2ms以内)")


def test_collision_performance():
    """衝突判定のパフォーマンステスト"""
    print("\n=== 衝突判定パフォーマンステスト ===")
    
    mesh = create_test_mesh("medium")
    spatial_index = SpatialIndex(mesh, index_type=IndexType.BVH)
    searcher = CollisionSearcher(spatial_index)
    collision_tester = SphereTriangleCollision(mesh)
    
    # テスト用の球と検索結果
    sphere_positions = [
        np.array([0.2, 0.2, 0.05]),
        np.array([0.5, 0.5, 0.05]),
        np.array([0.8, 0.8, 0.05]),
        np.array([0.3, 0.7, 0.05]),
        np.array([0.7, 0.3, 0.05])
    ]
    sphere_radius = 0.08
    
    times = []
    collisions = 0
    
    for i, position in enumerate(sphere_positions):
        hand = MockTrackedHand(f"hand_{i}", position)
        
        # 空間検索
        search_result = searcher.search_near_hand(hand, override_radius=0.2)
        
        # 衝突判定
        start_time = time.perf_counter()
        collision_info = collision_tester.test_sphere_collision(
            position, sphere_radius, search_result
        )
        elapsed = (time.perf_counter() - start_time) * 1000
        
        times.append(elapsed)
        if collision_info.has_collision:
            collisions += 1
        
        print(f"  球 {i+1}: {elapsed:.3f}ms, 衝突: {'あり' if collision_info.has_collision else 'なし'}")
    
    avg_time = np.mean(times)
    max_time = np.max(times)
    
    print(f"\n  平均時間: {avg_time:.3f}ms")
    print(f"  最大時間: {max_time:.3f}ms")
    print(f"  衝突検出数: {collisions}/{len(sphere_positions)}")
    print(f"  目標達成: {'✓' if max_time < 2.0 else '✗'} (目標: 2ms以内)")


def test_event_performance():
    """イベント生成のパフォーマンステスト"""
    print("\n=== イベント生成パフォーマンステスト ===")
    
    event_queue = CollisionEventQueue()
    
    # テスト用の衝突データ
    test_collisions = []
    for i in range(10):
        contact_point = ContactPoint(
            position=np.array([i * 0.1, 0.5, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            depth=0.01 + i * 0.005,
            triangle_index=i,
            barycentric=np.array([0.33, 0.33, 0.34]),
            collision_type=CollisionType.FACE_COLLISION
        )
        
        collision_info = CollisionInfo(
            has_collision=True,
            contact_points=[contact_point],
            closest_point=contact_point,
            total_penetration_depth=contact_point.depth,
            collision_normal=np.array([0.0, 0.0, 1.0]),
            collision_time_ms=1.0
        )
        
        test_collisions.append(collision_info)
    
    times = []
    events_created = 0
    
    for i, collision_info in enumerate(test_collisions):
        hand_position = np.array([i * 0.1, 0.5, 0.01])
        hand_velocity = np.array([0.05, 0.0, 0.0])
        
        start_time = time.perf_counter()
        event = event_queue.create_event(
            collision_info, f"hand_{i}", hand_position, hand_velocity
        )
        elapsed = (time.perf_counter() - start_time) * 1000
        
        times.append(elapsed)
        if event:
            events_created += 1
        
        print(f"  イベント {i+1}: {elapsed:.3f}ms")
    
    avg_time = np.mean(times)
    max_time = np.max(times)
    
    print(f"\n  平均時間: {avg_time:.3f}ms")
    print(f"  最大時間: {max_time:.3f}ms")
    print(f"  作成イベント数: {events_created}/{len(test_collisions)}")
    print(f"  目標達成: {'✓' if max_time < 1.0 else '✗'} (目標: 1ms以内)")


def test_full_pipeline_performance():
    """フル・パイプラインのパフォーマンステスト"""
    print("\n=== フル・パイプライン・パフォーマンステスト ===")
    
    # メッシュと各コンポーネントを初期化
    mesh = create_test_mesh("medium")
    spatial_index = SpatialIndex(mesh, index_type=IndexType.BVH)
    searcher = CollisionSearcher(spatial_index)
    collision_tester = SphereTriangleCollision(mesh)
    event_queue = CollisionEventQueue()
    
    # 複数の手をシミュレート
    hands = [
        MockTrackedHand("left_hand", np.array([0.3, 0.3, 0.1]), np.array([0.05, 0.02, -0.01])),
        MockTrackedHand("right_hand", np.array([0.7, 0.7, 0.1]), np.array([-0.02, 0.05, -0.01])),
        MockTrackedHand("extra_hand_1", np.array([0.5, 0.2, 0.1]), np.array([0.01, 0.03, 0.0])),
        MockTrackedHand("extra_hand_2", np.array([0.2, 0.8, 0.1]), np.array([0.03, -0.01, 0.0])),
        MockTrackedHand("extra_hand_3", np.array([0.8, 0.3, 0.1]), np.array([-0.01, -0.02, 0.01]))
    ]
    
    sphere_radius = 0.05
    total_times = []
    total_events = 0
    
    print(f"\n  {len(hands)}個の手を処理:")
    
    for frame in range(3):  # 3フレーム処理
        frame_start = time.perf_counter()
        frame_events = 0
        
        for hand in hands:
            pipeline_start = time.perf_counter()
            
            # 1. 空間検索
            search_result = searcher.search_near_hand(hand, override_radius=0.15)
            
            # 2. 衝突判定
            collision_info = collision_tester.test_sphere_collision(
                hand.position, sphere_radius, search_result
            )
            
            # 3. イベント生成
            event = event_queue.create_event(
                collision_info, hand.hand_id, hand.position, hand.velocity
            )
            
            pipeline_time = (time.perf_counter() - pipeline_start) * 1000
            
            if event:
                frame_events += 1
            
            print(f"    {hand.hand_id}: {pipeline_time:.3f}ms")
        
        frame_time = (time.perf_counter() - frame_start) * 1000
        total_times.append(frame_time)
        total_events += frame_events
        
        print(f"  フレーム {frame+1}: {frame_time:.3f}ms, イベント: {frame_events}")
    
    avg_frame_time = np.mean(total_times)
    max_frame_time = np.max(total_times)
    
    print(f"\n  平均フレーム時間: {avg_frame_time:.3f}ms")
    print(f"  最大フレーム時間: {max_frame_time:.3f}ms")
    print(f"  総イベント数: {total_events}")
    print(f"  目標達成: {'✓' if max_frame_time < 5.0 else '✗'} (目標: 5ms以内)")
    
    # パフォーマンス統計の表示
    print(f"\n  コンポーネント統計:")
    search_stats = searcher.get_performance_stats()
    collision_stats = collision_tester.get_performance_stats()
    
    print(f"    検索: 平均 {search_stats.get('average_search_time_ms', 0):.3f}ms")
    print(f"    衝突: 平均 {collision_stats.get('average_test_time_ms', 0):.3f}ms")


def main():
    """メインテスト実行関数"""
    print("衝突検出フェーズ パフォーマンステスト開始")
    print("=" * 50)
    
    try:
        test_search_performance()
        test_collision_performance()
        test_event_performance()
        test_full_pipeline_performance()
        
        print("\n" + "=" * 50)
        print("パフォーマンステスト完了")
        return True
        
    except Exception as e:
        print(f"\nテスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 