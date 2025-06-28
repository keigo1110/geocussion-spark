#!/usr/bin/env python3
"""
手検出フェーズの単体テスト
2D検出・3D投影・トラッキングの全機能テスト
"""

import unittest
import numpy as np
import time
import sys
import os

# テスト対象のモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.hands2d import (
    MediaPipeHandsWrapper, HandDetectionResult, HandLandmark, HandednessType,
    create_mock_hand_result, filter_hands_by_confidence
)
from src.detection.hands3d import (
    Hand3DProjector, Hand3DResult, Hand3DLandmark, DepthInterpolationMethod,
    create_mock_hand_3d_result
)
from src.detection.tracker import (
    Hand3DTracker, TrackedHand, TrackingState, KalmanFilterConfig,
    create_test_tracker, filter_stable_hands
)
from src.input.stream import CameraIntrinsics


class TestHandDetection2D(unittest.TestCase):
    """2D手検出のテスト"""
    
    def setUp(self):
        """テスト用の設定"""
        self.hands_wrapper = MediaPipeHandsWrapper(
            use_gpu=False,  # テスト環境ではCPU使用
            max_num_hands=2,
            min_detection_confidence=0.7
        )
    
    def test_wrapper_initialization(self):
        """ラッパーの初期化テスト"""
        self.assertIsNotNone(self.hands_wrapper)
        self.assertEqual(self.hands_wrapper.max_num_hands, 2)
        self.assertEqual(self.hands_wrapper.min_detection_confidence, 0.7)
        self.assertFalse(self.hands_wrapper.use_gpu)  # テスト環境
    
    def test_mock_hand_result(self):
        """モック手検出結果のテスト"""
        mock_result = create_mock_hand_result()
        
        self.assertIsInstance(mock_result, HandDetectionResult)
        self.assertEqual(len(mock_result.landmarks), 21)  # MediaPipeは21ランドマーク
        self.assertEqual(mock_result.handedness, HandednessType.RIGHT)
        self.assertGreater(mock_result.confidence, 0.9)
        
        # 中心点計算テスト
        center = mock_result.center_point
        self.assertEqual(len(center), 2)
        self.assertIsInstance(center[0], float)
        self.assertIsInstance(center[1], float)
        
        # 手のひら中心計算テスト
        palm_center = mock_result.palm_center
        self.assertEqual(len(palm_center), 2)
    
    def test_confidence_filtering(self):
        """信頼度フィルタリングテスト"""
        # 異なる信頼度の手検出結果を作成
        hands = []
        for conf in [0.5, 0.7, 0.9, 0.3]:
            mock_hand = create_mock_hand_result()
            mock_hand.confidence = conf
            hands.append(mock_hand)
        
        # 信頼度0.8以上でフィルタリング
        filtered = filter_hands_by_confidence(hands, min_confidence=0.8)
        
        self.assertEqual(len(filtered), 1)  # 0.9のみが残る
        self.assertEqual(filtered[0].confidence, 0.9)
    
    def test_performance_stats(self):
        """パフォーマンス統計テスト"""
        initial_stats = self.hands_wrapper.get_performance_stats()
        
        self.assertIn('total_frames', initial_stats)
        self.assertEqual(initial_stats['total_frames'], 0)
        
        # 統計リセットテスト
        self.hands_wrapper.reset_stats()
        reset_stats = self.hands_wrapper.get_performance_stats()
        self.assertEqual(reset_stats['total_frames'], 0)


class TestHand3DProjection(unittest.TestCase):
    """3D投影のテスト"""
    
    def setUp(self):
        """テスト用の設定"""
        self.intrinsics = CameraIntrinsics(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0,
            width=640, height=480
        )
        self.projector = Hand3DProjector(
            camera_intrinsics=self.intrinsics,
            interpolation_method=DepthInterpolationMethod.LINEAR
        )
    
    def test_projector_initialization(self):
        """プロジェクター初期化テスト"""
        self.assertIsNotNone(self.projector)
        self.assertEqual(self.projector.camera_intrinsics.width, 640)
        self.assertEqual(self.projector.camera_intrinsics.height, 480)
        self.assertEqual(self.projector.interpolation_method, DepthInterpolationMethod.LINEAR)
    
    def test_mock_3d_result(self):
        """モック3D結果のテスト"""
        mock_3d = create_mock_hand_3d_result()
        
        self.assertIsInstance(mock_3d, Hand3DResult)
        self.assertEqual(len(mock_3d.landmarks_3d), 21)
        self.assertGreater(mock_3d.confidence_3d, 0.8)
        
        # 手のひら法線ベクトルテスト（存在する場合のみ）
        if mock_3d.palm_normal is not None:
            normal = mock_3d.palm_normal
            self.assertEqual(len(normal), 3)
            self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=2)  # 正規化確認
    
    def test_depth_interpolation(self):
        """深度補間テスト"""
        # テスト用深度画像作成
        depth_image = np.full((480, 640), 1000, dtype=np.uint16)  # 1m
        
        # 中央部分に段差を作成
        depth_image[200:280, 280:360] = 800  # 0.8m
        
        # テスト用2D手検出結果
        mock_2d = create_mock_hand_result()
        
        # 3D投影実行
        result_3d = self.projector.project_hand_to_3d(mock_2d, depth_image)
        
        if result_3d:  # 投影が成功した場合
            self.assertIsInstance(result_3d, Hand3DResult)
            self.assertGreater(result_3d.confidence_3d, 0.0)
            
            # 深度値の範囲確認
            palm_z = result_3d.palm_center_3d[2]
            self.assertGreater(palm_z, 0.5)  # 0.5m以上
            self.assertLess(palm_z, 1.5)     # 1.5m以下
    
    def test_batch_projection(self):
        """バッチ投影テスト"""
        depth_image = np.full((480, 640), 1200, dtype=np.uint16)
        
        # 複数手の2D検出結果作成
        hands_2d = [create_mock_hand_result() for _ in range(3)]
        
        # バッチ投影実行
        results_3d = self.projector.project_hands_batch(hands_2d, depth_image)
        
        # 結果検証
        self.assertLessEqual(len(results_3d), 3)  # 最大3個
        for result in results_3d:
            self.assertIsInstance(result, Hand3DResult)
    
    def test_intrinsics_update(self):
        """内部パラメータ更新テスト"""
        new_intrinsics = CameraIntrinsics(
            fx=600.0, fy=600.0, cx=400.0, cy=300.0,
            width=800, height=600
        )
        
        self.projector.update_intrinsics(new_intrinsics)
        
        self.assertEqual(self.projector.camera_intrinsics.fx, 600.0)
        self.assertEqual(self.projector.camera_intrinsics.width, 800)


class TestHandTracking(unittest.TestCase):
    """手トラッキングのテスト"""
    
    def setUp(self):
        """テスト用の設定"""
        self.tracker = create_test_tracker()
    
    def test_tracker_initialization(self):
        """トラッカー初期化テスト"""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.tracked_hands), 0)
        self.assertIsNotNone(self.tracker.kalman_config)
    
    def test_new_track_creation(self):
        """新トラック作成テスト"""
        # 新しい3D手検出結果
        hands_3d = [create_mock_hand_3d_result()]
        
        # トラッキング更新
        tracked_hands = self.tracker.update(hands_3d)
        
        # 新トラックが作成されているか確認
        self.assertEqual(len(tracked_hands), 1)
        self.assertEqual(tracked_hands[0].state, TrackingState.INITIALIZING)
        self.assertIsNotNone(tracked_hands[0].id)
    
    def test_tracking_continuity(self):
        """トラッキング継続性テスト"""
        # 同じ位置の手を連続で検出
        base_position = (0.1, 0.0, 0.8)
        
        for i in range(6):  # 6フレーム連続
            mock_3d = create_mock_hand_3d_result()
            # 位置を少しずつ変化
            mock_3d.palm_center_3d = (
                base_position[0] + i * 0.01,
                base_position[1],
                base_position[2]
            )
            
            tracked_hands = self.tracker.update([mock_3d])
            
            # 最初のフレーム後は常に1つのトラックが存在
            if i > 0:
                self.assertEqual(len(tracked_hands), 1)
                
                # 十分な長さになったらTRACKING状態に
                if i >= 3:  # min_track_length = 3
                    self.assertEqual(tracked_hands[0].state, TrackingState.TRACKING)
    
    def test_kalman_filter_prediction(self):
        """カルマンフィルタ予測テスト"""
        # 等速運動する手をシミュレート
        positions = [
            (0.0, 0.0, 0.8),
            (0.05, 0.0, 0.8),
            (0.10, 0.0, 0.8),
            (0.15, 0.0, 0.8)
        ]
        
        track_id = None
        
        for i, pos in enumerate(positions):
            mock_3d = create_mock_hand_3d_result()
            mock_3d.palm_center_3d = pos
            
            tracked_hands = self.tracker.update([mock_3d])
            
            if tracked_hands:
                if track_id is None:
                    track_id = tracked_hands[0].id
                
                # トラックIDの継続性確認
                self.assertEqual(tracked_hands[0].id, track_id)
                
                # 速度推定の確認（3フレーム目以降）
                if i >= 2:
                    velocity = tracked_hands[0].velocity
                    # X方向の速度が正であることを確認
                    self.assertGreater(velocity[0], 0.0)
    
    def test_data_association(self):
        """データアソシエーションテスト"""
        # 2つの手を同時にトラッキング
        hand1_positions = [(0.0, 0.0, 0.8), (0.02, 0.0, 0.8), (0.04, 0.0, 0.8)]
        hand2_positions = [(0.3, 0.0, 0.8), (0.32, 0.0, 0.8), (0.34, 0.0, 0.8)]
        
        for i in range(3):
            hands_3d = []
            
            # 左手
            mock_3d_1 = create_mock_hand_3d_result()
            mock_3d_1.palm_center_3d = hand1_positions[i]
            mock_3d_1.handedness = HandednessType.LEFT
            hands_3d.append(mock_3d_1)
            
            # 右手
            mock_3d_2 = create_mock_hand_3d_result()
            mock_3d_2.palm_center_3d = hand2_positions[i]
            mock_3d_2.handedness = HandednessType.RIGHT
            hands_3d.append(mock_3d_2)
            
            tracked_hands = self.tracker.update(hands_3d)
            
            # 2つの手がトラッキングされているか確認
            if i >= 1:  # 2フレーム目以降
                self.assertEqual(len(tracked_hands), 2)
                
                # 手の左右が正しく割り当てられているか確認
                left_hands = [h for h in tracked_hands if h.handedness == HandednessType.LEFT]
                right_hands = [h for h in tracked_hands if h.handedness == HandednessType.RIGHT]
                
                self.assertEqual(len(left_hands), 1)
                self.assertEqual(len(right_hands), 1)
    
    def test_lost_track_handling(self):
        """消失トラック処理テスト"""
        # 手を検出後、消失させる
        mock_3d = create_mock_hand_3d_result()
        
        # 3フレーム検出してTRACKING状態にする
        tracked_hand_id = None
        for _ in range(3):
            tracked_hands = self.tracker.update([mock_3d])
            if tracked_hands:
                tracked_hand_id = tracked_hands[0].id
        
        # トラックが作成されていることを確認
        self.assertIsNotNone(tracked_hand_id)
        
        # 6フレーム消失（max_lost_frames=5を超える）
        for i in range(6):
            tracked_hands = self.tracker.update([])
            # アクティブなトラック（TRACKING, INITIALIZING状態）の数を確認
            # LOST状態のトラックは返されない
            if i >= 5:  # max_lost_framesを超えた場合
                self.assertEqual(len(tracked_hands), 0)
        
        # 内部のトラック状態を確認
        all_tracks = list(self.tracker.tracked_hands.values())
        
        # トラックが存在する場合、適切な状態になっているか確認
        if all_tracks:
            for track in all_tracks:
                # 長いトラック（min_track_length以上）はTERMINATED状態
                if track.track_length >= self.tracker.min_track_length:
                    self.assertEqual(track.state, TrackingState.TERMINATED)
                # 短いトラックは削除されているはず
                else:
                    self.fail(f"Short track should be deleted, but found: {track.state}")
        
        # アクティブなトラックが0個であることを最終確認
        active_tracks = [
            t for t in all_tracks 
            if t.state in [TrackingState.TRACKING, TrackingState.INITIALIZING]
        ]
        self.assertEqual(len(active_tracks), 0)
    
    def test_performance_measurement(self):
        """パフォーマンス計測テスト"""
        # 複数フレームの処理時間を計測
        mock_3d = create_mock_hand_3d_result()
        
        start_time = time.perf_counter()
        
        for _ in range(10):
            self.tracker.update([mock_3d])
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # 10フレームで100ms以下（目標：10ms/frame）
        self.assertLess(total_time, 100.0)
        
        # 統計情報確認
        stats = self.tracker.get_performance_stats()
        self.assertGreater(stats['total_updates'], 0)
        self.assertGreater(stats['avg_tracking_time_ms'], 0.0)


class TestDetectionIntegration(unittest.TestCase):
    """検出フェーズ統合テスト"""
    
    def setUp(self):
        """統合テスト用設定"""
        self.intrinsics = CameraIntrinsics(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0,
            width=640, height=480
        )
        self.hands_2d = MediaPipeHandsWrapper(use_gpu=False, max_num_hands=2)
        self.projector_3d = Hand3DProjector(self.intrinsics)
        self.tracker = create_test_tracker()
    
    def test_full_pipeline(self):
        """完全パイプラインテスト"""
        # テストデータ作成
        depth_image = np.full((480, 640), 1000, dtype=np.uint16)
        
        # パイプライン実行
        pipeline_start = time.perf_counter()
        
        # 1. 2D検出（モック）
        hands_2d = [create_mock_hand_result()]
        
        # 2. 3D投影
        hands_3d = self.projector_3d.project_hands_batch(hands_2d, depth_image)
        
        # 3. トラッキング
        tracked_hands = self.tracker.update(hands_3d)
        
        pipeline_time = (time.perf_counter() - pipeline_start) * 1000
        
        # パフォーマンス確認（目標：10ms以内）
        self.assertLess(pipeline_time, 15.0)  # 余裕を持って15ms
        
        print(f"Full pipeline time: {pipeline_time:.2f}ms")
        
        # 結果検証
        if hands_3d and tracked_hands:
            self.assertGreater(len(hands_3d), 0)
            self.assertGreater(len(tracked_hands), 0)
    
    def test_multi_frame_stability(self):
        """複数フレーム安定性テスト"""
        depth_image = np.full((480, 640), 1000, dtype=np.uint16)
        
        total_time = 0.0
        successful_frames = 0
        
        for frame_idx in range(20):  # 20フレーム処理
            frame_start = time.perf_counter()
            
            try:
                # 手の位置を少しずつ変化
                mock_2d = create_mock_hand_result()
                # 時間的変化をシミュレート
                for i, landmark in enumerate(mock_2d.landmarks):
                    landmark.x += 0.001 * frame_idx * np.sin(i * 0.1)
                    landmark.y += 0.001 * frame_idx * np.cos(i * 0.1)
                
                # パイプライン実行
                hands_3d = self.projector_3d.project_hands_batch([mock_2d], depth_image)
                tracked_hands = self.tracker.update(hands_3d)
                
                frame_time = (time.perf_counter() - frame_start) * 1000
                total_time += frame_time
                successful_frames += 1
                
            except Exception as e:
                print(f"Frame {frame_idx} failed: {e}")
        
        # 結果評価
        self.assertGreater(successful_frames, 15)  # 75%以上成功
        
        if successful_frames > 0:
            avg_frame_time = total_time / successful_frames
            self.assertLess(avg_frame_time, 12.0)  # 平均12ms以下
            
            print(f"Multi-frame test: {successful_frames}/{20} frames successful")
            print(f"Average frame time: {avg_frame_time:.2f}ms")
    
    def test_filter_integration(self):
        """フィルタリング統合テスト"""
        # 複数の手（信頼度、安定性が異なる）を作成
        tracked_hands = []
        
        # 高信頼度・安定した手
        stable_hand = TrackedHand(
            id="stable",
            handedness=HandednessType.RIGHT,
            state=TrackingState.TRACKING,
            position=np.array([0.1, 0.0, 0.8]),
            velocity=np.array([0.02, 0.0, 0.0]),  # 低速
            acceleration=np.zeros(3),
            confidence_2d=0.95,
            confidence_3d=0.90,
            confidence_tracking=0.92,
            last_seen_time=time.time(),
            track_length=10,
            lost_frames=0,
            hand_size=0.15
        )
        
        # 低信頼度・不安定な手
        unstable_hand = TrackedHand(
            id="unstable",
            handedness=HandednessType.LEFT,
            state=TrackingState.TRACKING,
            position=np.array([0.3, 0.0, 0.8]),
            velocity=np.array([0.5, 0.2, 0.1]),  # 高速
            acceleration=np.zeros(3),
            confidence_2d=0.60,
            confidence_3d=0.55,
            confidence_tracking=0.50,
            last_seen_time=time.time(),
            track_length=3,
            lost_frames=0,
            hand_size=0.12
        )
        
        tracked_hands = [stable_hand, unstable_hand]
        
        # 安定した手のみフィルタリング
        stable_hands = filter_stable_hands(
            tracked_hands,
            min_confidence=0.7,
            max_speed=1.0
        )
        
        # 安定した手のみが残っているか確認
        self.assertEqual(len(stable_hands), 1)
        self.assertEqual(stable_hands[0].id, "stable")


if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2) 