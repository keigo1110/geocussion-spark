#!/usr/bin/env python3
"""
音響生成フェーズのテストスイート

パラメータマッピング、音響合成、ボイス管理の全機能を包括的にテストします。
"""

import unittest
import time
import numpy as np
from typing import List
import threading
from unittest.mock import MagicMock, patch

# テスト対象モジュール
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sound.mapping import (
    AudioMapper, AudioParameters,
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
    CollisionEvent, CollisionEventQueue, EventType, CollisionIntensity, CollisionType
)
from src.utils import config_manager


@patch('src.sound.mapping.config_manager', MagicMock())
@patch('src.sound.synth.config_manager', MagicMock())
class SoundTest(unittest.TestCase):

    def setUp(self):
        """テストごとのセットアップ"""
        self.config_data = {
            'audio': {
                'default_scale': 'PENTATONIC',
                'default_instrument': 'MARIMBA',
                'scales': {'PENTATONIC': [0, 2, 4, 7, 9]},
                'instruments': {
                    'MARIMBA': {'envelope': {'attack': 0.01, 'decay': 0.5, 'sustain': 0.2, 'release': 1.0}},
                    'SYNTH_PAD': {'envelope': {'attack': 0.1, 'decay': 0.8, 'sustain': 0.7, 'release': 1.5}},
                    'polyphony': 10,
                    'master_volume': 0.7
                }
            }
        }

        # config_manager.getをモック化
        self.patcher = patch('src.utils.config.config_manager.get')
        self.mock_config_get = self.patcher.start()
        self.mock_config_get.side_effect = lambda key, default=None: self.config_data.get(key, default) if key != 'audio' else self.config_data['audio']
        
        # モックのAudioSynthesizer
        self.synthesizer = MagicMock(spec=AudioSynthesizer)
        
        # テスト対象のVoiceManager
        self.voice_manager = VoiceManager(synthesizer=self.synthesizer, max_polyphony=10)
        
        # AudioMapperのインスタンス
        self.mapper = AudioMapper()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.patcher.stop()

    def test_audio_mapping_pitch(self):
        """音高マッピングテスト"""
        # Y座標が低い -> 音高が低い
        event_low = self.create_mock_event(y_pos=0.1)
        params_low = self.mapper.map_collision_event(event_low)
        # Y座標が高い -> 音高が高い
        event_high = self.create_mock_event(y_pos=0.9)
        params_high = self.mapper.map_collision_event(event_high)
        self.assertGreater(params_high.pitch, params_low.pitch)

    def test_audio_mapping_velocity(self):
        """音量マッピングテスト"""
        # 速度が遅い -> velocityが小さい
        event_slow = self.create_mock_event(velocity=0.2)
        params_slow = self.mapper.map_collision_event(event_slow)
        # 速度が速い -> velocityが大きい
        event_fast = self.create_mock_event(velocity=0.8)
        params_fast = self.mapper.map_collision_event(event_fast)
        self.assertGreater(params_fast.velocity, params_slow.velocity)

    def test_audio_mapping_instrument_selection(self):
        """楽器選択マッピングテスト"""
        # デフォルト楽器が設定されていること
        event = self.create_mock_event()
        params = self.mapper.map_collision_event(event)
        self.assertEqual(params.instrument, 'MARIMBA')

    def test_voice_allocation_and_deallocation_with_fadeout(self):
        """フェードアウト付きボイス割り当てと解放テスト"""
        params = self.mapper.map_collision_event(self.create_mock_event())
        voice_id = self.voice_manager.allocate_voice(params)
        
        self.assertIsNotNone(voice_id)
        self.assertIn(voice_id, self.voice_manager.active_voices)
        
        # フェードアウト付きでボイス解放
        if voice_id:
            self.voice_manager.deallocate_voice(voice_id, fade_out=True)
            self.synthesizer.stop_voice.assert_called_once()
            # stop_voiceのfadeout_sec引数が0より大きいことを確認
            args, kwargs = self.synthesizer.stop_voice.call_args
            self.assertGreater(kwargs.get('fadeout_sec', 0), 0)

    def test_voice_stealing_with_fadeout(self):
        """ボイススティール時のフェードアウトテスト"""
        # ポリフォニーを埋める
        for i in range(10):
            params = self.mapper.map_collision_event(self.create_mock_event(event_id=f"event_{i}"))
            self.voice_manager.allocate_voice(params)
        
        # 新しいボイスを割り当ててスティールを発生させる
        self.synthesizer.reset_mock() # 以前の呼び出しをリセット
        new_params = self.mapper.map_collision_event(self.create_mock_event(event_id="new_event"))
        self.voice_manager.allocate_voice(new_params)
        
        # stop_voiceがフェードアウト付きで呼ばれたことを確認
        self.synthesizer.stop_voice.assert_called_once()
        args, kwargs = self.synthesizer.stop_voice.call_args
        self.assertGreater(kwargs.get('fadeout_sec', 0), 0)

    def create_mock_event(self, event_id="test_event", hand_id="test_hand", event_type=EventType.COLLISION_START,
                          intensity=CollisionIntensity.MEDIUM, velocity=0.5, y_pos=0.5, x_pos=0.0) -> CollisionEvent:
        return CollisionEvent(event_id=event_id, event_type=event_type, timestamp=time.time(), duration_ms=0.0,
                              contact_position=np.array([x_pos, y_pos, 0.0]), hand_position=np.array([x_pos, y_pos, -0.1]),
                              surface_normal=np.array([0.0, 0.0, 1.0]), intensity=intensity, velocity=velocity,
                              penetration_depth=0.01, contact_area=0.001, pitch_hint=y_pos, timbre_hint=0.5,
                              spatial_position=np.array([x_pos, 0.0, 0.0]), triangle_index=0, hand_id=hand_id,
                              collision_type=CollisionType.FACE_COLLISION)

    @patch('src.utils.config.config_manager.get')
    def test_voicemanager_thread_safety(self, mock_config_get):
        """ボイスマネージャのスレッドセーフティテスト"""
        mock_config_get.return_value = self.config_data['audio']
        mapper = AudioMapper()
        vm = self.voice_manager
        num_threads = 8
        iterations = 20

        def worker(thread_id):
            for i in range(iterations):
                params = mapper.map_collision_event(
                    self.create_mock_event(event_id=f"event_{thread_id}_{i}")
                )
                voice_id = vm.allocate_voice(params)
                if voice_id:
                    vm.update_voice_parameters(voice_id, volume=0.6)
                vm.get_performance_stats()
                time.sleep(0.01)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

        stats = vm.get_performance_stats()
        total_created = stats['total_voices_created']
        self.assertGreater(total_created, 0)


@patch('src.sound.mapping.config_manager', MagicMock())
@patch('src.sound.synth.config_manager', MagicMock())
class TestSoundIntegration(unittest.TestCase):
    """音響生成統合テスト"""
    
    def setUp(self):
        """統合テストのセットアップ"""
        self.config_data = {
            'audio': {
                'default_scale': 'PENTATONIC',
                'default_instrument': 'MARIMBA',
                'scales': { 'PENTATONIC': [0, 2, 4, 7, 9] },
                'instruments': {
                    'MARIMBA': {
                        'pyo_object': 'PhyMarimba', 'type': 'physical_modeling',
                        'default_params': {'mul': 1.0},
                        'envelope': {'attack': 0.01, 'decay': 0.5, 'sustain': 0.2, 'release': 1.0}
                    }
                },
                'polyphony': 8,
                'master_volume': 0.7
            }
        }
        self.patcher = patch('src.utils.config.config_manager.get')
        self.mock_config_get = self.patcher.start()
        self.mock_config_get.side_effect = lambda key, default=None: self.config_data.get(key, default) if key != 'audio' else self.config_data['audio']

        self.mock_event_creator = SoundTest()

    def tearDown(self):
        self.patcher.stop()

    def test_full_pipeline_no_sound(self):
        """音響生成なしのパイプライン統合テスト"""
        mapper = AudioMapper()
        event = self.mock_event_creator.create_mock_event()
        audio_params = mapper.map_collision_event(event)
        self.assertIsInstance(audio_params, AudioParameters)
        print(f"統合テスト（音響なし）: pitch={audio_params.pitch:.1f}, vel={audio_params.velocity:.2f}")

    @unittest.skipIf(sys.platform == "darwin" and os.environ.get("CI"), "macOS CIではオーディオデバイスがなく失敗する")
    def test_full_pipeline_with_sound(self):
        """音響生成ありのパイプライン統合テスト"""
        mapper = AudioMapper()
        collision_event = self.mock_event_creator.create_mock_event()
        audio_params = mapper.map_collision_event(collision_event)
        
        synthesizer = AudioSynthesizer()
        voice_manager = VoiceManager(synthesizer=synthesizer, max_polyphony=8)
        
        if synthesizer.start_engine():
            try:
                voice_id = voice_manager.allocate_voice(audio_params)
                self.assertIsNotNone(voice_id)
                time.sleep(0.1)
                voice_manager.cleanup_finished_voices()
                
                mapper_stats = mapper.get_performance_stats()
                synth_stats = synthesizer.get_performance_stats()
                voice_stats = voice_manager.get_performance_stats()
                
                self.assertGreater(mapper_stats['total_mappings'], 0)
                self.assertGreater(synth_stats['total_notes_played'], 0)
                self.assertGreater(voice_stats['total_voices_created'], 0)
                
            finally:
                synthesizer.stop_engine()
        else:
            self.skipTest("Pyo audio engine could not be started.")


def run_tests():
    """テスト実行"""
    # テストスイート作成
    test_classes = [
        SoundTest,
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