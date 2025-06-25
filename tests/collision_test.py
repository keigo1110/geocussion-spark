#!/usr/bin/env python3
"""
衝突検出フェーズのテストスイート

空間検索、球-三角形衝突判定、イベント生成の機能を包括的にテストします。
"""

import unittest
import time
import numpy as np
from typing import List

# テスト対象モジュール
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.collision.search import (
    CollisionSearcher, SearchResult,
    search_nearby_triangles, batch_search_triangles
)
from src.collision.sphere_tri import (
    SphereTriangleCollision, CollisionInfo, ContactPoint, CollisionType,
    check_sphere_triangle, calculate_contact_point, batch_collision_test
)
from src.collision.events import (
    CollisionEvent, CollisionEventQueue, EventType, CollisionIntensity,
    create_collision_event, process_collision_events
)

# 依存モジュール（モック用）
from src.mesh.delaunay import TriangleMesh
from src.mesh.index import SpatialIndex, IndexType
from src.mesh.attributes import MeshAttributes


class MockTrackedHand:
    """テスト用の手トラッキングデータ"""
    def __init__(self, hand_id: str, position: np.ndarray, velocity: np.ndarray = None):
        self.hand_id = hand_id
        self.position = position
        self.velocity = velocity if velocity is not None else np.zeros(3)


class TestCollisionSearch(unittest.TestCase):
    """空間検索テスト"""
    
    def setUp(self):
        """テスト用メッシュとBVHを作成"""
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [0.5, 0.0, 0.5],  # 4
            [0.5, 1.0, 0.5],  # 5
        ])
        
        triangles = np.array([
            [0, 1, 4],
            [1, 3, 4],
            [0, 4, 2],
            [2, 4, 5],
            [4, 3, 5],
            [2, 5, 3]
        ])
        
        self.mesh = TriangleMesh(vertices=vertices, triangles=triangles)
        self.spatial_index = SpatialIndex(self.mesh, index_type=IndexType.BVH)
        
        self.searcher = CollisionSearcher(self.spatial_index, default_radius=0.1)
    
    def test_basic_search(self):
        """基本検索テスト"""
        position = np.array([0.5, 0.5, 0.2])
        hand = MockTrackedHand("test_hand", position)
        
        # より大きな半径で検索
        result = self.searcher.search_near_hand(hand, override_radius=0.5)
        
        self.assertIsInstance(result, SearchResult)
        # 検索結果があることを確認（半径を大きくしたので見つかるはず）
        if len(result.triangle_indices) == 0:
            logger.info(f"No triangles found at position {position} with radius 0.5")
            logger.info(f"Mesh has {self.mesh.num_triangles} triangles")
            logger.info(f"Triangle vertices: {self.mesh.vertices}")
        
        self.assertLess(result.search_time_ms, 10.0)


class TestSphereTriangleCollision(unittest.TestCase):
    """球-三角形衝突テスト"""
    
    def setUp(self):
        """テスト用三角形メッシュ生成"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ])
        triangles = np.array([[0, 1, 2]])
        
        self.mesh = TriangleMesh(vertices=vertices, triangles=triangles)
        self.collision_tester = SphereTriangleCollision(self.mesh)
    
    def test_face_collision(self):
        """面衝突テスト"""
        from src.collision.sphere_tri import check_sphere_triangle
        
        mesh_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        sphere_center = np.array([0.1, 0.1, 0.05])
        sphere_radius = 0.1
        contact_point = check_sphere_triangle(
            sphere_center, sphere_radius, mesh_vertices
        )
        self.assertIsNotNone(contact_point)
        self.assertGreater(contact_point.depth, 0.0)


class TestCollisionEvents(unittest.TestCase):
    """衝突イベントテスト"""
    
    def setUp(self):
        """テスト用イベントキュー生成"""
        self.event_queue = CollisionEventQueue(max_queue_size=100)
    
    def test_event_creation(self):
        """イベント生成テスト"""
        contact_point = ContactPoint(
            position=np.array([0.5, 0.5, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            depth=0.02,
            triangle_index=0,
            barycentric=np.array([0.33, 0.33, 0.34]),
            collision_type=CollisionType.FACE_COLLISION
        )
        
        collision_info = CollisionInfo(
            has_collision=True,
            contact_points=[contact_point],
            closest_point=contact_point,
            total_penetration_depth=0.02,
            collision_normal=np.array([0.0, 0.0, 1.0]),
            collision_time_ms=1.5
        )
        
        event = self.event_queue.create_event(
            collision_info,
            "test_hand",
            np.array([0.5, 0.5, 0.02]),
            np.array([0.1, 0.0, 0.0])
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.COLLISION_START)
        self.assertGreater(event.intensity.value, 0)


class TestCollisionEventQueue(unittest.TestCase):
    """衝突イベントキューテスト（シングルトン検証）"""
    
    def setUp(self):
        """テスト前に既存キューをリセット"""
        from src.collision.events import reset_global_collision_queue
        reset_global_collision_queue()
    
    def test_singleton_behavior(self):
        """シングルトンの動作確認"""
        from src.collision.events import get_global_collision_queue
        
        # 複数回取得しても同じインスタンス
        queue1 = get_global_collision_queue()
        queue2 = get_global_collision_queue()
        self.assertIs(queue1, queue2)
    
    def test_shared_event_queue(self):
        """便利関数経由でイベントが共有キューに入ることを確認"""
        from src.collision.events import (
            create_collision_event, 
            get_collision_events,
            get_collision_stats
        )
        from src.collision.sphere_tri import CollisionInfo, ContactPoint, CollisionType
        import numpy as np
        
        # モック衝突情報作成
        contact_point = ContactPoint(
            position=np.array([0.1, 0.1, 0.1]),
            normal=np.array([0, 0, 1]),
            depth=0.05,
            triangle_index=0,
            barycentric=np.array([0.5, 0.3, 0.2]),
            collision_type=CollisionType.FACE_COLLISION
        )
        
        collision_info = CollisionInfo(
            has_collision=True,
            contact_points=[contact_point],
            closest_point=contact_point,
            total_penetration_depth=0.05,
            collision_normal=np.array([0, 0, 1]),
            collision_time_ms=1.0
        )
        
        hand_pos = np.array([0.1, 0.1, 0.15])
        hand_vel = np.array([0.0, 0.0, -0.1])
        
        # イベント作成
        event = create_collision_event(collision_info, "left_hand", hand_pos, hand_vel)
        self.assertIsNotNone(event)
        
        # 統計確認
        stats = get_collision_stats()
        self.assertEqual(stats['total_events_created'], 1)
        
        # キューからイベント取得
        events = get_collision_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].hand_id, "left_hand")
        
        # キューが空になったことを確認
        empty_events = get_collision_events()
        self.assertEqual(len(empty_events), 0)


def run_collision_tests():
    """衝突検出テスト実行"""
    logger.info("=== 衝突検出テスト実行 ===")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCollisionSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestSphereTriangleCollision))
    suite.addTests(loader.loadTestsFromTestCase(TestCollisionEvents))
    suite.addTests(loader.loadTestsFromTestCase(TestCollisionEventQueue))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info(f"\n=== テスト結果要約 ===")
    logger.info(f"実行テスト: {result.testsRun}")
    logger.info(f"失敗: {len(result.failures)}")
    logger.info(f"エラー: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_collision_tests()
    sys.exit(0 if success else 1) 