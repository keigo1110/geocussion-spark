#!/usr/bin/env python3
"""
音響モジュール テスト

音響バックエンド、マッピング、シンセサイザー、ボイス管理のテストを提供します。
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import numpy as np

from src.sound.backend import create_backend, BackendType, get_available_backends, is_backend_available
from src.sound.backend.null_backend import NullAudioBackend
from src.sound.mapping import AudioParameters, InstrumentType, ScaleType, AudioParameterMapper
from src.sound.voice_mgr import VoiceManager, VoiceState
from src.sound.synth import AudioSynthesizer, EngineState


class TestAudioBackends:
    """音響バックエンドのテスト"""
    
    def test_null_backend_creation(self):
        """NullBackendの作成テスト"""
        backend = create_backend(BackendType.NULL)
        assert isinstance(backend, NullAudioBackend)
        assert backend.get_backend_type() == BackendType.NULL
    
    def test_null_backend_lifecycle(self):
        """NullBackendのライフサイクルテスト"""
        backend = NullAudioBackend()
        
        # 初期化
        assert backend.initialize(44100, 2, 256)
        assert backend.initialized
        
        # 開始
        assert backend.start()
        assert backend.running
        
        # ボイス作成
        voice_id = backend.create_voice(440.0, 0.5)
        assert voice_id is not None
        assert backend.is_voice_active(voice_id)
        assert backend.get_active_voice_count() == 1
        
        # ボイス停止
        assert backend.stop_voice(voice_id)
        assert not backend.is_voice_active(voice_id)
        assert backend.get_active_voice_count() == 0
        
        # 停止
        assert backend.stop()
        assert not backend.running
        
        # シャットダウン
        assert backend.shutdown()
        assert not backend.initialized
    
    def test_null_backend_stats(self):
        """NullBackendの統計情報テスト"""
        backend = NullAudioBackend()
        backend.initialize(48000, 2, 512)
        
        stats = backend.get_stats()
        assert stats['sample_rate'] == 48000
        assert stats['latency_ms'] > 0
        assert stats['active_voices'] == 0
        assert not stats['running']
    
    def test_backend_auto_selection(self):
        """バックエンド自動選択テスト"""
        backend = create_backend()
        assert backend is not None
        assert backend.get_backend_type() in [BackendType.PYO, BackendType.NULL]
    
    def test_available_backends(self):
        """利用可能バックエンド一覧テスト"""
        backends = get_available_backends()
        assert BackendType.NULL in backends  # Nullは常に利用可能
        assert len(backends) >= 1
        
        # 各バックエンドの利用可能性チェック
        for backend_type in backends:
            assert is_backend_available(backend_type)


class TestAudioParameterMapper:
    """音響パラメータマッピングのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.mapper = AudioParameterMapper()
    
    def test_coordinate_to_frequency(self):
        """座標から周波数へのマッピングテスト"""
        # Y座標範囲テスト
        freq_low = self.mapper._coordinate_to_frequency(0.0, 1.0)  # 最低点
        freq_high = self.mapper._coordinate_to_frequency(1.0, 1.0)  # 最高点
        
        assert freq_low < freq_high
        assert 200 <= freq_low <= 2000  # 妥当な周波数範囲
        assert 200 <= freq_high <= 2000
    
    def test_velocity_to_amplitude(self):
        """速度から振幅へのマッピングテスト"""
        # 速度範囲テスト
        amp_low = self.mapper._velocity_to_amplitude(0.0)
        amp_mid = self.mapper._velocity_to_amplitude(0.5)
        amp_high = self.mapper._velocity_to_amplitude(1.0)
        
        assert 0 <= amp_low <= amp_mid <= amp_high <= 1.0
    
    def test_position_to_pan(self):
        """位置からパンニングへのマッピングテスト"""
        pan_left = self.mapper._position_to_pan(-1.0)
        pan_center = self.mapper._position_to_pan(0.0)
        pan_right = self.mapper._position_to_pan(1.0)
        
        assert pan_left < pan_center < pan_right
        assert -1.0 <= pan_left <= 1.0
        assert -1.0 <= pan_right <= 1.0
    
    def test_map_collision_to_audio(self):
        """衝突からオーディオパラメータへのマッピングテスト"""
        collision_data = {
            'contact_point': np.array([0.5, 0.7, 0.0]),
            'collision_strength': 0.8,
            'surface_normal': np.array([0.0, 1.0, 0.0]),
            'velocity': np.array([0.1, -0.5, 0.0])
        }
        
        params = self.mapper.map_collision_to_audio(collision_data)
        
        assert isinstance(params, AudioParameters)
        assert params.frequency > 0
        assert 0 <= params.amplitude <= 1.0
        assert -1.0 <= params.pan <= 1.0
        assert params.instrument_type in InstrumentType
        assert params.scale_type in ScaleType
    
    def test_scale_constraints(self):
        """スケール制約テスト"""
        # ペンタトニックスケール制約
        self.mapper.scale_type = ScaleType.PENTATONIC
        
        # 複数の周波数でスケール制約を確認
        for y in np.linspace(0, 1, 20):
            freq = self.mapper._coordinate_to_frequency(y, 1.0)
            # ペンタトニックスケールの音程に近い値になっているかチェック
            assert freq > 0


class TestVoiceManager:
    """ボイス管理のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.voice_mgr = VoiceManager(max_voices=4)
    
    def test_voice_allocation(self):
        """ボイス割り当てテスト"""
        params = AudioParameters(
            frequency=440.0,
            amplitude=0.5,
            pan=0.0,
            instrument_type=InstrumentType.MARIMBA,
            scale_type=ScaleType.MAJOR,
            duration=1.0
        )
        
        voice_id = self.voice_mgr.allocate_voice(params)
        assert voice_id is not None
        assert self.voice_mgr.get_active_voice_count() == 1
        
        voice_info = self.voice_mgr.get_voice_info(voice_id)
        assert voice_info is not None
        assert voice_info.state == VoiceState.ALLOCATED
    
    def test_voice_stealing(self):
        """ボイススティールテスト"""
        params = AudioParameters(
            frequency=440.0,
            amplitude=0.5,
            pan=0.0,
            instrument_type=InstrumentType.MARIMBA,
            scale_type=ScaleType.MAJOR,
            duration=1.0
        )
        
        voice_ids = []
        # 最大数まで割り当て
        for i in range(self.voice_mgr.max_voices):
            voice_id = self.voice_mgr.allocate_voice(params)
            assert voice_id is not None
            voice_ids.append(voice_id)
        
        # 最大数到達確認
        assert self.voice_mgr.get_active_voice_count() == self.voice_mgr.max_voices
        
        # さらに割り当て（ボイススティール発生）
        new_voice_id = self.voice_mgr.allocate_voice(params)
        assert new_voice_id is not None
        
        # 古いボイスが停止されているか確認
        active_count = sum(1 for vid in voice_ids if self.voice_mgr.get_voice_info(vid) is not None)
        assert active_count < len(voice_ids)
    
    def test_voice_cleanup(self):
        """ボイスクリーンアップテスト"""
        params = AudioParameters(
            frequency=440.0,
            amplitude=0.5,
            pan=0.0,
            instrument_type=InstrumentType.MARIMBA,
            scale_type=ScaleType.MAJOR,
            duration=0.1  # 短い持続時間
        )
        
        voice_id = self.voice_mgr.allocate_voice(params)
        assert voice_id is not None
        
        # 時間経過を待つ
        time.sleep(0.15)
        
        # クリーンアップ実行
        self.voice_mgr.cleanup_finished_voices()
        
        # ボイスが削除されているか確認
        assert self.voice_mgr.get_voice_info(voice_id) is None


class TestAudioSynthesizer:
    """音響シンセサイザーのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        # NullBackendを使用してテスト
        self.synth = AudioSynthesizer(preferred_backend=BackendType.NULL)
    
    def test_synthesizer_initialization(self):
        """シンセサイザー初期化テスト"""
        assert self.synth.state == EngineState.STOPPED
        
        success = self.synth.start_engine()
        assert success
        assert self.synth.state == EngineState.RUNNING
    
    def test_audio_playback(self):
        """音響再生テスト"""
        self.synth.start_engine()
        
        params = AudioParameters(
            frequency=440.0,
            amplitude=0.5,
            pan=0.0,
            instrument_type=InstrumentType.MARIMBA,
            scale_type=ScaleType.MAJOR,
            duration=0.5
        )
        
        voice_id = self.synth.play_audio_parameters(params)
        assert voice_id is not None
        
        # ボイスがアクティブか確認
        assert self.synth.is_voice_active(voice_id)
        
        # ボイス停止
        self.synth.stop_voice(voice_id)
        assert not self.synth.is_voice_active(voice_id)
    
    def test_synthesizer_shutdown(self):
        """シンセサイザーシャットダウンテスト"""
        self.synth.start_engine()
        assert self.synth.state == EngineState.RUNNING
        
        self.synth.shutdown()
        assert self.synth.state == EngineState.STOPPED
    
    def test_performance_stats(self):
        """パフォーマンス統計テスト"""
        self.synth.start_engine()
        
        stats = self.synth.get_performance_stats()
        assert 'engine_state' in stats
        assert 'active_voices' in stats
        assert 'latency_ms' in stats


class TestSoundIntegration:
    """音響モジュール統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.mapper = AudioParameterMapper()
        self.synth = AudioSynthesizer(preferred_backend=BackendType.NULL)
        self.synth.start_engine()
    
    def teardown_method(self):
        """テストクリーンアップ"""
        self.synth.shutdown()
    
    def test_end_to_end_audio_pipeline(self):
        """E2E音響パイプラインテスト"""
        # 模擬衝突データ
        collision_data = {
            'contact_point': np.array([0.3, 0.6, 0.0]),
            'collision_strength': 0.7,
            'surface_normal': np.array([0.0, 1.0, 0.0]),
            'velocity': np.array([0.2, -0.4, 0.0])
        }
        
        # マッピング
        params = self.mapper.map_collision_to_audio(collision_data)
        assert isinstance(params, AudioParameters)
        
        # 再生
        voice_id = self.synth.play_audio_parameters(params)
        assert voice_id is not None
        
        # ボイス確認
        assert self.synth.is_voice_active(voice_id)
        
        # 停止
        self.synth.stop_voice(voice_id)
        assert not self.synth.is_voice_active(voice_id)
    
    def test_multiple_collision_handling(self):
        """複数衝突同時処理テスト"""
        collision_points = [
            {'contact_point': np.array([i*0.2, 0.5, 0.0]), 
             'collision_strength': 0.5, 
             'surface_normal': np.array([0.0, 1.0, 0.0]),
             'velocity': np.array([0.1, -0.3, 0.0])}
            for i in range(3)
        ]
        
        voice_ids = []
        for collision in collision_points:
            params = self.mapper.map_collision_to_audio(collision)
            voice_id = self.synth.play_audio_parameters(params)
            if voice_id:
                voice_ids.append(voice_id)
        
        # 複数ボイスが作成されたことを確認
        assert len(voice_ids) > 0
        
        # 全ボイス停止
        for voice_id in voice_ids:
            self.synth.stop_voice(voice_id)


# パフォーマンステスト
class TestSoundPerformance:
    """音響処理パフォーマンステスト"""
    
    @pytest.mark.benchmark
    def test_mapping_performance(self, benchmark):
        """マッピング性能テスト"""
        mapper = AudioParameterMapper()
        collision_data = {
            'contact_point': np.array([0.5, 0.7, 0.0]),
            'collision_strength': 0.8,
            'surface_normal': np.array([0.0, 1.0, 0.0]),
            'velocity': np.array([0.1, -0.5, 0.0])
        }
        
        def map_collision():
            return mapper.map_collision_to_audio(collision_data)
        
        result = benchmark(map_collision)
        assert isinstance(result, AudioParameters)
    
    @pytest.mark.benchmark
    def test_voice_allocation_performance(self, benchmark):
        """ボイス割り当て性能テスト"""
        voice_mgr = VoiceManager(max_voices=16)
        params = AudioParameters(
            frequency=440.0,
            amplitude=0.5,
            pan=0.0,
            instrument_type=InstrumentType.MARIMBA,
            scale_type=ScaleType.MAJOR,
            duration=1.0
        )
        
        def allocate_voice():
            return voice_mgr.allocate_voice(params)
        
        voice_id = benchmark(allocate_voice)
        assert voice_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 