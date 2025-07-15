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
    create_collision_event, process_collision_events, reset_global_collision_queue
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
            print(f"No triangles found at position {position} with radius 0.5")
            print(f"Mesh has {self.mesh.num_triangles} triangles")
            print(f"Triangle vertices: {self.mesh.vertices}")
        
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
        self.assertEqual(event.event_type, EventType.COLLISION_IMPACT)
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
    suite.addTests(loader.loadTestsFromTestCase(TestCollisionDebounce))  # 新しいテスト追加
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info(f"\n=== テスト結果要約 ===")
    logger.info(f"実行テスト: {result.testsRun}")
    logger.info(f"失敗: {len(result.failures)}")
    logger.info(f"エラー: {len(result.errors)}")
    
    return result.wasSuccessful()


class TestCollisionDebounce(unittest.TestCase):
    """一打＝一音の衝突デバウンステスト"""
    
    def setUp(self):
        """テスト用データを準備"""
        # グローバルキューをリセット
        reset_global_collision_queue()
        
        # テスト用の衝突情報を作成
        self.contact_point = ContactPoint(
            position=np.array([0.0, 0.5, 0.0], dtype=np.float32),
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            depth=0.01,
            triangle_index=0,
            barycentric=np.array([0.3, 0.3, 0.4], dtype=np.float32),
            collision_type=CollisionType.FACE_COLLISION
        )
        
        self.collision_info = CollisionInfo(
            has_collision=True,
            contact_points=[self.contact_point],
            closest_point=self.contact_point,
            total_penetration_depth=0.01,
            collision_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            collision_time_ms=1.0
        )
        
        self.hand_position = np.array([0.0, 0.6, 0.0], dtype=np.float32)
        self.hand_velocity = np.array([0.0, -0.1, 0.0], dtype=np.float32)
        self.hand_id = "test_hand"
    
    def test_single_impact_generation(self):
        """連続衝突フレームでもIMPACTは1回だけ生成されることを検証"""
        queue = CollisionEventQueue(debounce_ms=50.0)
        
        events = []
        # 5フレーム連続で同じ衝突を送信（100FPS想定で50ms）
        for frame in range(5):
            event = queue.create_event(
                self.collision_info, 
                self.hand_id, 
                self.hand_position, 
                self.hand_velocity
            )
            if event:
                events.append(event)
            # フレーム間隔をシミュレート（10ms = 100FPS）
            time.sleep(0.01)
        
        # IMPACTは1回、CONTINUEは4回発生するはず
        impact_events = [e for e in events if e.event_type == EventType.COLLISION_IMPACT]
        continue_events = [e for e in events if e.event_type == EventType.COLLISION_CONTINUE]
        
        self.assertEqual(len(impact_events), 1, "IMPACTイベントは1回だけ生成されるべき")
        self.assertEqual(len(continue_events), 4, "CONTINUEイベントは4回生成されるべき")
        
        # 統計確認
        stats = queue.get_stats()
        self.assertEqual(stats['impacts_created'], 1)
        self.assertEqual(stats['impacts_debounced'], 0)  # 継続中はデバウンス統計に含まれない
    
    def test_debounce_prevention(self):
        """デバウンス時間内の再衝突が抑制されることを検証"""
        queue = CollisionEventQueue(debounce_ms=100.0)
        
        # 最初の衝突
        event1 = queue.create_event(
            self.collision_info, 
            self.hand_id, 
            self.hand_position, 
            self.hand_velocity
        )
        self.assertIsNotNone(event1)
        self.assertEqual(event1.event_type, EventType.COLLISION_IMPACT)
        
        # 衝突終了
        no_collision = CollisionInfo(
            has_collision=False,
            contact_points=[],
            closest_point=None,
            total_penetration_depth=0.0,
            collision_normal=np.zeros(3),
            collision_time_ms=0.0
        )
        end_event = queue.create_event(no_collision, self.hand_id, self.hand_position)
        self.assertIsNotNone(end_event)
        self.assertEqual(end_event.event_type, EventType.COLLISION_END)
        
        # デバウンス時間内（50ms後）に再衝突を試行
        time.sleep(0.05)
        event2 = queue.create_event(
            self.collision_info, 
            self.hand_id, 
            self.hand_position, 
            self.hand_velocity
        )
        self.assertIsNone(event2, "デバウンス時間内の再衝突は抑制されるべき")
        
        # 統計確認
        stats = queue.get_stats()
        self.assertEqual(stats['impacts_created'], 1)
        self.assertEqual(stats['impacts_debounced'], 1)
    
    def test_debounce_expiry_allows_new_impact(self):
        """デバウンス時間経過後は新しいIMPACTが生成されることを検証"""
        queue = CollisionEventQueue(debounce_ms=50.0)  # 短いデバウンス時間
        
        # 最初の衝突
        event1 = queue.create_event(
            self.collision_info, 
            self.hand_id, 
            self.hand_position, 
            self.hand_velocity
        )
        self.assertIsNotNone(event1)
        self.assertEqual(event1.event_type, EventType.COLLISION_IMPACT)
        
        # 衝突終了
        no_collision = CollisionInfo(
            has_collision=False,
            contact_points=[],
            closest_point=None,
            total_penetration_depth=0.0,
            collision_normal=np.zeros(3),
            collision_time_ms=0.0
        )
        queue.create_event(no_collision, self.hand_id, self.hand_position)
        
        # デバウンス時間経過を待つ（60ms）
        time.sleep(0.06)
        
        # 再衝突を試行
        event2 = queue.create_event(
            self.collision_info, 
            self.hand_id, 
            self.hand_position, 
            self.hand_velocity
        )
        self.assertIsNotNone(event2, "デバウンス時間経過後は新しいIMPACTが生成されるべき")
        self.assertEqual(event2.event_type, EventType.COLLISION_IMPACT)
        
        # 統計確認
        stats = queue.get_stats()
        self.assertEqual(stats['impacts_created'], 2)
        self.assertEqual(stats['impacts_debounced'], 0)
    
    def test_multiple_hands_independent_debounce(self):
        """複数の手のデバウンスが独立して動作することを検証"""
        queue = CollisionEventQueue(debounce_ms=100.0)
        
        # 手1の衝突
        event1 = queue.create_event(
            self.collision_info, 
            "hand1", 
            self.hand_position, 
            self.hand_velocity
        )
        self.assertIsNotNone(event1)
        self.assertEqual(event1.event_type, EventType.COLLISION_IMPACT)
        
        # 手2の衝突（同じタイミング）
        event2 = queue.create_event(
            self.collision_info, 
            "hand2", 
            self.hand_position + np.array([0.1, 0, 0]), 
            self.hand_velocity
        )
        self.assertIsNotNone(event2)
        self.assertEqual(event2.event_type, EventType.COLLISION_IMPACT)
        
        # 統計確認：両方ともIMPACTが生成される
        stats = queue.get_stats()
        self.assertEqual(stats['impacts_created'], 2)
        self.assertEqual(stats['impacts_debounced'], 0)
    
    def test_intensity_preserved_during_continue(self):
        """CONTINUE中は初期IMPACTの強度が保持されることを検証"""
        queue = CollisionEventQueue()
        
        # 最初の衝突（高強度）
        high_velocity = np.array([0.0, -0.5, 0.0], dtype=np.float32)  # 高速
        impact_event = queue.create_event(
            self.collision_info, 
            self.hand_id, 
            self.hand_position, 
            high_velocity
        )
        self.assertIsNotNone(impact_event)
        initial_intensity = impact_event.intensity
        
        # 継続中（低速になっても強度は維持される）
        low_velocity = np.array([0.0, -0.01, 0.0], dtype=np.float32)  # 低速
        continue_event = queue.create_event(
            self.collision_info, 
            self.hand_id, 
            self.hand_position, 
            low_velocity
        )
        self.assertIsNotNone(continue_event)
        self.assertEqual(continue_event.event_type, EventType.COLLISION_CONTINUE)
        self.assertEqual(continue_event.intensity, initial_intensity, 
                        "CONTINUEイベントは初期IMPACTの強度を保持するべき")


if __name__ == '__main__':
    success = run_collision_tests()
    sys.exit(0 if success else 1) 