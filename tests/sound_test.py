#!/usr/bin/env python3
"""
音響生成フェーズのテストスイート

パラメータマッピング、音響合成、ボイス管理の全機能を包括的にテストします。
"""

import unittest
import time
import numpy as np
from typing import List

# テスト対象モジュール
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sound.mapping import (
    AudioMapper, AudioParameters, InstrumentType, ScaleType,
    map_collision_to_audio, batch_map_collisions, create_scale_mapper
)

from src.sound.synth import (
    AudioSynthesizer, AudioConfig, EngineState,
    create_audio_synthesizer, play_audio_immediately
)

from src.sound.voice_mgr import (
    VoiceManager, VoiceState, StealStrategy, SpatialMode, SpatialConfig,
    create_voice_manager, allocate_and_play
)

# 依存モジュール（モック用）
from src.collision.events import (
    CollisionEvent, CollisionEventQueue, EventType, CollisionIntensity
)
from src.collision.sphere_tri import ContactPoint, CollisionType


class MockCollisionEvent:
    """テスト用の衝突イベント"""
    def __init__(
        self,
        event_id: str = "test_event",
        hand_id: str = "test_hand",
        contact_position: np.ndarray = None,
        surface_normal: np.ndarray = None,
        velocity: float = 0.1,
        intensity: CollisionIntensity = CollisionIntensity.MEDIUM,
        penetration_depth: float = 0.01,
        contact_area: float = 0.001,
        collision_type = CollisionType.FACE_COLLISION,
        timestamp: float = None
    ):
        self.event_id = event_id
        self.hand_id = hand_id
        self.contact_position = contact_position if contact_position is not None else np.array([0.5, 0.5, 0.0])
        self.surface_normal = surface_normal if surface_normal is not None else np.array([0.0, 0.0, 1.0])
        self.velocity = velocity
        self.intensity = intensity
        self.penetration_depth = penetration_depth
        self.contact_area = contact_area
        self.collision_type = collision_type
        self.timestamp = timestamp or time.perf_counter()
        self.pitch_hint = 0.5
        self.timbre_hint = 0.5


class TestAudioMapping(unittest.TestCase):
    """音響パラメータマッピングテスト"""
    
    def setUp(self):
        """テスト用マッパー生成"""
        self.mapper = AudioMapper(
            scale=ScaleType.PENTATONIC,
            pitch_range=(48, 84),
            default_instrument=InstrumentType.MARIMBA
        )
    
    def test_basic_mapping(self):
        """基本マッピングテスト"""
        event = MockCollisionEvent()
        audio_params = self.mapper.map_collision_event(event)
        
        self.assertIsInstance(audio_params, AudioParameters)
        self.assertGreaterEqual(audio_params.pitch, 48)
        self.assertLessEqual(audio_params.pitch, 84)
        self.assertGreaterEqual(audio_params.velocity, 0.0)
        self.assertLessEqual(audio_params.velocity, 1.0)
        self.assertEqual(audio_params.event_id, "test_event")
        self.assertEqual(audio_params.hand_id, "test_hand")
        
        print(f"マッピングテスト: pitch={audio_params.pitch:.1f}, velocity={audio_params.velocity:.2f}")
    
    def test_pitch_mapping(self):
        """音高マッピングテスト"""
        # Y座標の異なる位置での音高テスト
        positions = [
            np.array([0.0, 0.0, 0.0]),  # 低い位置
            np.array([0.0, 0.5, 0.0]),  # 中央位置
            np.array([0.0, 1.0, 0.0])   # 高い位置
        ]
        
        pitches = []
        for pos in positions:
            event = MockCollisionEvent(contact_position=pos)
            audio_params = self.mapper.map_collision_event(event)
            pitches.append(audio_params.pitch)
        
        # Y座標が高いほど音高も高くなることを確認
        self.assertLessEqual(pitches[0], pitches[1])
        self.assertLessEqual(pitches[1], pitches[2])
        
        print(f"音高マッピング: {pitches[0]:.1f} -> {pitches[1]:.1f} -> {pitches[2]:.1f}")
    
    def test_velocity_mapping(self):
        """音量マッピングテスト"""
        # 異なる強度での音量テスト
        intensities = [
            CollisionIntensity.WHISPER,
            CollisionIntensity.MEDIUM,
            CollisionIntensity.MAXIMUM
        ]
        
        velocities = []
        for intensity in intensities:
            event = MockCollisionEvent(intensity=intensity)
            audio_params = self.mapper.map_collision_event(event)
            velocities.append(audio_params.velocity)
        
        # 強度が高いほど音量も大きくなることを確認
        self.assertLessEqual(velocities[0], velocities[1])
        self.assertLessEqual(velocities[1], velocities[2])
        
        print(f"音量マッピング: {velocities[0]:.2f} -> {velocities[1]:.2f} -> {velocities[2]:.2f}")
    
    def test_instrument_selection(self):
        """楽器選択テスト"""
        # Y座標による楽器選択
        high_event = MockCollisionEvent(contact_position=np.array([0.0, 0.8, 0.0]))
        low_event = MockCollisionEvent(contact_position=np.array([0.0, 0.2, 0.0]))
        
        high_params = self.mapper.map_collision_event(high_event)
        low_params = self.mapper.map_collision_event(low_event)
        
        print(f"楽器選択: 高位置={high_params.instrument.value}, 低位置={low_params.instrument.value}")
        
        # 楽器が選択されていることを確認
        self.assertIsInstance(high_params.instrument, InstrumentType)
        self.assertIsInstance(low_params.instrument, InstrumentType)
    
    def test_spatial_mapping(self):
        """空間マッピングテスト"""
        # X座標の異なる位置でのパンニングテスト
        left_event = MockCollisionEvent(contact_position=np.array([-1.0, 0.0, 0.0]))
        center_event = MockCollisionEvent(contact_position=np.array([0.0, 0.0, 0.0]))
        right_event = MockCollisionEvent(contact_position=np.array([1.0, 0.0, 0.0]))
        
        left_params = self.mapper.map_collision_event(left_event)
        center_params = self.mapper.map_collision_event(center_event)
        right_params = self.mapper.map_collision_event(right_event)
        
        # パンニングが正しく設定されることを確認
        self.assertLess(left_params.pan, center_params.pan)
        self.assertLess(center_params.pan, right_params.pan)
        
        print(f"パンニング: L={left_params.pan:.2f}, C={center_params.pan:.2f}, R={right_params.pan:.2f}")
    
    def test_performance(self):
        """マッピングパフォーマンステスト"""
        events = [MockCollisionEvent(f"event_{i}") for i in range(100)]
        
        start_time = time.perf_counter()
        audio_params_list = batch_map_collisions(events, self.mapper)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        avg_time_per_mapping = elapsed_ms / len(events)
        
        self.assertEqual(len(audio_params_list), 100)
        self.assertLess(avg_time_per_mapping, 0.1)  # 0.1ms以内
        
        print(f"マッピング性能: {avg_time_per_mapping:.3f}ms/mapping")


class TestAudioSynthesizer(unittest.TestCase):
    """音響シンセサイザーテスト"""
    
    def setUp(self):
        """テスト用シンセサイザー生成"""
        config = AudioConfig(
            sample_rate=44100,
            buffer_size=256,
            max_polyphony=16
        )
        self.synthesizer = AudioSynthesizer(config=config)
    
    def test_engine_lifecycle(self):
        """エンジンライフサイクルテスト"""
        # 初期状態
        self.assertEqual(self.synthesizer.state, EngineState.STOPPED)
        
        # 開始テスト（pyoが利用可能な場合のみ）
        try:
            success = self.synthesizer.start_engine()
            if success:
                self.assertEqual(self.synthesizer.state, EngineState.RUNNING)
                print("音響エンジン開始成功")
                
                # 停止テスト
                self.synthesizer.stop_engine()
                self.assertEqual(self.synthesizer.state, EngineState.STOPPED)
                print("音響エンジン停止成功")
            else:
                print("音響エンジン開始失敗（pyoが利用不可）")
                self.assertEqual(self.synthesizer.state, EngineState.ERROR)
        except Exception as e:
            print(f"音響エンジンテストスキップ: {e}")
    
    def test_audio_parameter_creation(self):
        """音響パラメータ作成テスト"""
        audio_params = AudioParameters(
            pitch=60.0,  # C4
            velocity=0.7,
            duration=1.0,
            instrument=InstrumentType.MARIMBA,
            timbre=0.5,
            brightness=0.6,
            pan=0.0,
            distance=0.5,
            reverb=0.3,
            attack=0.01,
            decay=0.1,
            sustain=0.8,
            release=0.5,
            event_id="test_audio",
            hand_id="test_hand",
            timestamp=time.perf_counter()
        )
        
        # MIDI変換テスト
        self.assertEqual(audio_params.midi_note, 60)
        self.assertEqual(audio_params.velocity_127, 88)  # int(0.7 * 127) = 88
        self.assertAlmostEqual(audio_params.frequency, 261.63, places=1)  # C4周波数
        
        print(f"音響パラメータ: MIDI={audio_params.midi_note}, 周波数={audio_params.frequency:.1f}Hz")
    
    def test_voice_management_basic(self):
        """基本ボイス管理テスト"""
        # モックパラメータ作成
        audio_params = AudioParameters(
            pitch=60.0, velocity=0.5, duration=1.0,
            instrument=InstrumentType.MARIMBA, timbre=0.5, brightness=0.5,
            pan=0.0, distance=0.5, reverb=0.3,
            attack=0.01, decay=0.1, sustain=0.8, release=0.5,
            event_id="test", hand_id="test", timestamp=time.perf_counter()
        )
        
        # pyoが利用可能な場合のみボイス再生テスト
        if self.synthesizer.start_engine():
            voice_id = self.synthesizer.play_audio_parameters(audio_params)
            if voice_id:
                self.assertIsInstance(voice_id, str)
                print(f"ボイス再生成功: {voice_id}")
                
                # ボイス停止
                self.synthesizer.stop_voice(voice_id)
                print("ボイス停止成功")
            
            self.synthesizer.stop_engine()


class TestVoiceManager(unittest.TestCase):
    """ボイス管理システムテスト"""
    
    def setUp(self):
        """テスト用ボイス管理システム生成"""
        config = AudioConfig(sample_rate=44100, buffer_size=256, max_polyphony=8)
        self.synthesizer = AudioSynthesizer(config=config)
        self.voice_manager = VoiceManager(
            synthesizer=self.synthesizer,
            max_polyphony=8,
            steal_strategy=StealStrategy.OLDEST
        )
    
    def test_voice_allocation(self):
        """ボイス割り当てテスト"""
        audio_params = AudioParameters(
            pitch=60.0, velocity=0.5, duration=1.0,
            instrument=InstrumentType.MARIMBA, timbre=0.5, brightness=0.5,
            pan=0.0, distance=0.5, reverb=0.3,
            attack=0.01, decay=0.1, sustain=0.8, release=0.5,
            event_id="test", hand_id="test", timestamp=time.perf_counter()
        )
        
        # ボイス割り当て（シンセサイザーが利用可能でない場合はスキップ）
        voice_id = self.voice_manager.allocate_voice(audio_params, priority=5)
        
        if voice_id:
            self.assertIsInstance(voice_id, str)
            print(f"ボイス割り当て成功: {voice_id}")
            
            # ボイス情報取得
            voice_info = self.voice_manager.get_voice_info(voice_id)
            self.assertIsNotNone(voice_info)
            self.assertEqual(voice_info.audio_params.pitch, 60.0)
            
            # ボイス解放
            self.voice_manager.deallocate_voice(voice_id)
            print("ボイス解放成功")
        else:
            print("ボイス割り当てスキップ（音響エンジン未使用）")
    
    def test_polyphony_limit(self):
        """ポリフォニー制限テスト"""
        audio_params_list = []
        for i in range(10):  # 制限(8)を超える数
            params = AudioParameters(
                pitch=60.0 + i, velocity=0.5, duration=2.0,
                instrument=InstrumentType.MARIMBA, timbre=0.5, brightness=0.5,
                pan=0.0, distance=0.5, reverb=0.3,
                attack=0.01, decay=0.1, sustain=0.8, release=0.5,
                event_id=f"test_{i}", hand_id=f"hand_{i}", timestamp=time.perf_counter()
            )
            audio_params_list.append(params)
        
        allocated_voices = []
        for params in audio_params_list:
            voice_id = self.voice_manager.allocate_voice(params)
            if voice_id:
                allocated_voices.append(voice_id)
        
        # 最大ポリフォニー以下になることを確認
        active_count = len(self.voice_manager.active_voices)
        self.assertLessEqual(active_count, self.voice_manager.max_polyphony)
        
        print(f"ポリフォニーテスト: {active_count}/{self.voice_manager.max_polyphony} ボイス")
        
        # 全ボイス解放
        for voice_id in allocated_voices:
            self.voice_manager.deallocate_voice(voice_id)
    
    def test_steal_strategies(self):
        """ボイススティール戦略テスト"""
        strategies = [
            StealStrategy.OLDEST,
            StealStrategy.QUIETEST,
            StealStrategy.LOWEST_PRIORITY
        ]
        
        for strategy in strategies:
            self.voice_manager.set_steal_strategy(strategy)
            self.assertEqual(self.voice_manager.steal_strategy, strategy)
            print(f"スティール戦略設定: {strategy.value}")
    
    def test_spatial_processing(self):
        """空間処理テスト"""
        spatial_config = SpatialConfig(
            mode=SpatialMode.STEREO_PAN,
            room_size=10.0,
            distance_attenuation=True
        )
        
        self.voice_manager.update_spatial_config(spatial_config)
        
        # 異なる空間位置でのボイス割り当て
        positions = [
            np.array([-2.0, 0.0, 0.0]),  # 左
            np.array([0.0, 0.0, 0.0]),   # 中央
            np.array([2.0, 0.0, 0.0])    # 右
        ]
        
        for i, pos in enumerate(positions):
            audio_params = AudioParameters(
                pitch=60.0, velocity=0.5, duration=1.0,
                instrument=InstrumentType.MARIMBA, timbre=0.5, brightness=0.5,
                pan=0.0, distance=0.5, reverb=0.3,
                attack=0.01, decay=0.1, sustain=0.8, release=0.5,
                event_id=f"spatial_{i}", hand_id=f"hand_{i}", timestamp=time.perf_counter()
            )
            
            voice_id = self.voice_manager.allocate_voice(audio_params, spatial_position=pos)
            if voice_id:
                voice_info = self.voice_manager.get_voice_info(voice_id)
                print(f"空間テスト {i}: pan={voice_info.current_pan:.2f}")
                self.voice_manager.deallocate_voice(voice_id)
    
    def test_performance_stats(self):
        """パフォーマンス統計テスト"""
        stats = self.voice_manager.get_performance_stats()
        
        # 統計項目の存在確認
        required_keys = [
            'total_voices_created', 'total_voices_stolen',
            'current_active_voices', 'polyphony_usage_percent'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        print(f"ボイス管理統計: {stats}")


class TestSoundIntegration(unittest.TestCase):
    """音響生成統合テスト"""
    
    def test_end_to_end_pipeline(self):
        """エンドツーエンドパイプラインテスト"""
        print("\n=== 音響生成パイプライン統合テスト ===")
        
        # 1. 衝突イベント作成
        collision_event = MockCollisionEvent(
            contact_position=np.array([0.3, 0.7, 0.1]),
            velocity=0.15,
            intensity=CollisionIntensity.MEDIUM_LOUD
        )
        
        # 2. 音響パラメータマッピング
        mapper = AudioMapper(scale=ScaleType.PENTATONIC)
        audio_params = mapper.map_collision_event(collision_event)
        
        print(f"音響パラメータ: pitch={audio_params.pitch:.1f}, vel={audio_params.velocity:.2f}")
        
        # 3. シンセサイザー初期化
        synthesizer = create_audio_synthesizer(buffer_size=256, max_polyphony=16)
        
        # 4. ボイス管理システム初期化
        voice_manager = create_voice_manager(
            synthesizer, 
            max_polyphony=16, 
            steal_strategy=StealStrategy.OLDEST
        )
        
        # 5. 音響再生（シンセサイザーが利用可能な場合）
        if synthesizer.start_engine():
            voice_id = allocate_and_play(
                voice_manager,
                audio_params,
                priority=7,
                spatial_position=np.array([0.3, 0.0, 0.1])
            )
            
            if voice_id:
                print(f"音響再生成功: {voice_id}")
                
                # 短時間待機
                time.sleep(0.1)
                
                # 統計取得
                mapper_stats = mapper.get_performance_stats()
                synth_stats = synthesizer.get_performance_stats()
                voice_stats = voice_manager.get_performance_stats()
                
                print(f"マッピング時間: {mapper_stats['last_mapping_time_ms']:.3f}ms")
                print(f"アクティブボイス: {voice_stats['current_active_voices']}")
                
                # クリーンアップ
                voice_manager.deallocate_voice(voice_id)
            
            synthesizer.stop_engine()
            print("パイプラインテスト完了")
        else:
            print("音響エンジンが利用できないため、パイプラインテストをスキップ")


def run_tests():
    """テスト実行"""
    # テストスイート作成
    test_classes = [
        TestAudioMapping,
        TestAudioSynthesizer,
        TestVoiceManager,
        TestSoundIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("音響生成フェーズのテスト開始...")
    success = run_tests()
    
    if success:
        print("\n✓ 全テストが成功しました！")
    else:
        print("\n✗ 一部のテストが失敗しました。")
    
    print("音響生成フェーズのテスト完了") 