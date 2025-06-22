#!/usr/bin/env python3
"""
音響生成 - シンセサイザーエンジン

pyoを使用したリアルタイム音響合成エンジン。
複数楽器の物理モデリングとサンプラー機能を提供します。
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np

try:
    import pyo
except ImportError:
    print("Warning: pyo not available. Audio synthesis will be disabled.")
    pyo = None

# 他フェーズとの連携
from .mapping import AudioParameters, InstrumentType


class EngineState(Enum):
    """エンジン状態の列挙"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AudioConfig:
    """音響設定データ"""
    sample_rate: int = 44100        # サンプリングレート
    buffer_size: int = 256          # バッファサイズ
    channels: int = 2               # チャンネル数（ステレオ）
    audio_driver: str = "portaudio"  # オーディオドライバ
    enable_duplex: bool = False     # 全二重通信
    max_polyphony: int = 32         # 最大ポリフォニー数
    master_volume: float = 0.7      # マスターボリューム
    reverb_level: float = 0.3       # リバーブレベル
    
    @property
    def latency_ms(self) -> float:
        """レイテンシーをミリ秒で取得"""
        return (self.buffer_size / self.sample_rate) * 1000


class AudioSynthesizer:
    """pyo音響合成エンジン"""
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        enable_physical_modeling: bool = True,
        enable_effects: bool = True,
        enable_spatial_audio: bool = True
    ):
        """
        初期化
        
        Args:
            config: 音響設定
            enable_physical_modeling: 物理モデリングを有効にするか
            enable_effects: エフェクトを有効にするか
            enable_spatial_audio: 空間音響を有効にするか
        """
        self.config = config or AudioConfig()
        self.enable_physical_modeling = enable_physical_modeling
        self.enable_effects = enable_effects
        self.enable_spatial_audio = enable_spatial_audio
        
        # pyo関連
        self.server: Optional[pyo.Server] = None
        self.state = EngineState.STOPPED
        
        # 楽器インスタンス
        self.instruments: Dict[InstrumentType, Any] = {}
        self.active_voices: Dict[str, Dict] = {}  # voice_id -> voice_data
        
        # エフェクト
        self.reverb: Optional[Callable[[Any], pyo.Freeverb]] = None
        
        # パフォーマンス統計
        self.stats = {
            'total_notes_played': 0,
            'active_voices_count': 0,
            'max_voices_used': 0,
            'audio_dropouts': 0,
            'average_latency_ms': 0.0,
            'cpu_usage': 0.0
        }
        
        # スレッド安全性
        self._lock = threading.Lock()
        
        print(f"AudioSynthesizer initialized with config: {self.config}")
    
    def start_engine(self) -> bool:
        """
        音響エンジンを開始
        
        Returns:
            成功したかどうか
        """
        if pyo is None:
            print("Error: pyo is not available")
            self.state = EngineState.ERROR
            return False
        
        try:
            self.state = EngineState.STARTING
            
            # pyoサーバー初期化
            self.server = pyo.Server(
                sr=self.config.sample_rate,
                nchnls=self.config.channels,
                buffersize=self.config.buffer_size,
                duplex=int(self.config.enable_duplex),
                audio=self.config.audio_driver
            )
            
            # サーバー起動順序修正: boot() -> start()
            self.server.boot()
            self.server.start()
            
            # 楽器とエフェクトの初期化
            self._initialize_instruments()
            self._initialize_effects()
            
            self.state = EngineState.RUNNING
            print(f"Audio engine started successfully. Latency: {self.config.latency_ms:.1f}ms")
            return True
            
        except Exception as e:
            print(f"Failed to start audio engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    def stop_engine(self):
        """音響エンジンを停止"""
        try:
            self.state = EngineState.STOPPING
            
            # 全ボイスを停止
            self.stop_all_voices()
            
            # サーバー停止
            if self.server:
                self.server.stop()
                self.server.shutdown()
                self.server = None
            
            self.state = EngineState.STOPPED
            print("Audio engine stopped")
            
        except Exception as e:
            print(f"Error stopping audio engine: {e}")
            self.state = EngineState.ERROR
    
    def play_audio_parameters(self, params: AudioParameters) -> Optional[str]:
        """
        音響パラメータを再生
        
        Args:
            params: 音響パラメータ
            
        Returns:
            ボイスID（再生に成功した場合）
        """
        if self.state != EngineState.RUNNING:
            return None
        
        with self._lock:
            try:
                # ボイスID生成
                voice_id = f"{params.event_id}_{int(time.time() * 1000000)}"
                
                # 楽器に応じたボイス生成
                voice = self._create_voice(params)
                if voice is None:
                    return None
                
                # ボイス管理に追加
                self.active_voices[voice_id] = {
                    'voice': voice,
                    'params': params,
                    'start_time': time.perf_counter(),
                    'duration': params.duration,
                    'instrument': params.instrument
                }
                
                # 統計更新
                self.stats['total_notes_played'] += 1
                self.stats['active_voices_count'] = len(self.active_voices)
                self.stats['max_voices_used'] = max(
                    self.stats['max_voices_used'],
                    self.stats['active_voices_count']
                )
                
                return voice_id
                
            except Exception as e:
                print(f"Error playing audio: {e}")
                return None
    
    def stop_voice(self, voice_id: str):
        """特定のボイスを停止"""
        with self._lock:
            if voice_id in self.active_voices:
                try:
                    voice_data = self.active_voices[voice_id]
                    voice = voice_data['voice']
                    
                    # ボイス停止
                    if hasattr(voice, 'stop'):
                        voice.stop()
                    
                    # 管理から削除
                    del self.active_voices[voice_id]
                    self.stats['active_voices_count'] = len(self.active_voices)
                    
                except Exception as e:
                    print(f"Error stopping voice {voice_id}: {e}")
    
    def stop_all_voices(self):
        """全ボイスを停止"""
        with self._lock:
            voice_ids = list(self.active_voices.keys())
            for voice_id in voice_ids:
                self.stop_voice(voice_id)
    
    def update_master_volume(self, volume: float):
        """マスターボリューム更新"""
        self.config.master_volume = max(0.0, min(1.0, volume))
        if self.server:
            self.server.setAmp(self.config.master_volume)
    
    def _initialize_instruments(self):
        """楽器を初期化"""
        if not self.server:
            return
        
        # 各楽器タイプのテンプレートを作成
        for instrument_type in InstrumentType:
            self.instruments[instrument_type] = self._create_instrument_template(instrument_type)
    
    def _initialize_effects(self):
        """エフェクトを初期化"""
        if not self.server or not self.enable_effects:
            return
        
        try:
            # リバーブエフェクト (後で入力を渡すファクトリとして保持)
            self.reverb = lambda inp: pyo.Freeverb(
                inp,
                size=0.8,
                damp=0.6,
                bal=self.config.reverb_level
            )
            
            # サーバーのマスターボリュームを設定
            self.server.setAmp(self.config.master_volume)
            
        except Exception as e:
            print(f"Error initializing effects: {e}")
    
    def _create_instrument_template(self, instrument_type: InstrumentType) -> Dict:
        """楽器テンプレートを作成"""
        
        templates = {
            InstrumentType.MARIMBA: {
                'oscillator_type': 'sine',
                'harmonics': [1.0, 0.3, 0.1, 0.05],
                'formant_freq': [440, 880, 1320],
                'resonance': 0.8,
                'decay_curve': 'exponential'
            },
            
            InstrumentType.SYNTH_PAD: {
                'oscillator_type': 'saw',
                'harmonics': [1.0, 0.5, 0.25, 0.125],
                'filter_cutoff': 2000,
                'filter_resonance': 0.3,
                'decay_curve': 'linear'
            },
            
            InstrumentType.BELL: {
                'oscillator_type': 'fm',
                'carrier_ratio': 1.0,
                'modulator_ratio': 3.14,
                'modulation_index': 2.0,
                'decay_curve': 'exponential'
            },
            
            InstrumentType.CRYSTAL: {
                'oscillator_type': 'sine',
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.2],
                'detune': 0.02,
                'chorus_depth': 0.3,
                'decay_curve': 'exponential'
            },
            
            InstrumentType.DRUM: {
                'oscillator_type': 'noise',
                'filter_type': 'bandpass',
                'filter_freq': 200,
                'filter_q': 2.0,
                'decay_curve': 'exponential'
            },
            
            InstrumentType.WATER_DROP: {
                'oscillator_type': 'sine',
                'frequency_sweep': True,
                'sweep_range': 0.5,
                'resonance': 0.9,
                'decay_curve': 'exponential'
            },
            
            InstrumentType.WIND: {
                'oscillator_type': 'noise',
                'filter_type': 'lowpass',
                'filter_cutoff': 1000,
                'amplitude_modulation': 0.2,
                'decay_curve': 'linear'
            },
            
            InstrumentType.STRING: {
                'oscillator_type': 'karplus_strong',
                'pluck_position': 0.3,
                'damping': 0.1,
                'string_tension': 0.8,
                'decay_curve': 'exponential'
            }
        }
        
        return templates.get(instrument_type, templates[InstrumentType.MARIMBA])
    
    def _create_voice(self, params: AudioParameters) -> Optional[Any]:
        """パラメータに基づいてボイスを作成"""
        if not self.server:
            return None
        
        try:
            instrument_template = self.instruments[params.instrument]
            
            # 基本周波数（型変換確実に）
            freq = float(params.frequency)
            
            # エンベロープ（型変換確実に）
            envelope = pyo.Adsr(
                attack=float(params.attack),
                decay=float(params.decay),
                sustain=float(params.sustain),
                release=float(params.release),
                dur=float(params.duration)
            )
            
            # 楽器に応じたオシレーター生成
            oscillator = self._create_oscillator(params, instrument_template, freq)
            
            # 音量適用（型変換確実に）
            voice = oscillator * envelope * float(params.velocity)
            
            # パンニング適用（型変換を確実に）
            if self.enable_spatial_audio and self.config.channels == 2:
                voice = pyo.Pan(voice, outs=2, pan=float(params.pan))
            
            # エフェクト適用（型変換確実に）
            if self.enable_effects and self.reverb:
                reverb_send = voice * float(params.reverb)
                reverb_out = self.reverb(reverb_send)
                voice = voice + reverb_out
            
            # 出力開始
            voice.out()
            envelope.play()
            
            return voice
            
        except Exception as e:
            print(f"Error creating voice: {e}")
            return None
    
    def _create_oscillator(self, params: AudioParameters, template: Dict, freq: float) -> Any:
        """楽器テンプレートに基づいてオシレーターを作成"""
        
        osc_type = template.get('oscillator_type', 'sine')
        
        if osc_type == 'sine':
            return pyo.Sine(freq=freq)
            
        elif osc_type == 'saw':
            return pyo.Saw(freq=freq)
            
        elif osc_type == 'fm':
            carrier_ratio = template.get('carrier_ratio', 1.0)
            modulator_ratio = template.get('modulator_ratio', 2.0)
            mod_index = template.get('modulation_index', 1.0)
            
            modulator = pyo.Sine(freq=freq * modulator_ratio)
            return pyo.Sine(freq=freq * carrier_ratio + modulator * mod_index)
            
        elif osc_type == 'noise':
            return pyo.Noise()
            
        elif osc_type == 'karplus_strong':
            return pyo.Pluck(
                freq=freq,
                dur=params.duration,
                damping=template.get('damping', 0.1)
            )
        
        else:
            # デフォルト
            return pyo.Sine(freq=freq)
    
    def cleanup_finished_voices(self):
        """終了したボイスをクリーンアップ"""
        with self._lock:
            current_time = time.perf_counter()
            finished_voices = []
            
            for voice_id, voice_data in self.active_voices.items():
                elapsed = current_time - voice_data['start_time']
                if elapsed >= voice_data['duration']:
                    finished_voices.append(voice_id)
            
            for voice_id in finished_voices:
                self.stop_voice(voice_id)
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        stats['engine_state'] = self.state.value
        stats['config'] = {
            'sample_rate': self.config.sample_rate,
            'buffer_size': self.config.buffer_size,
            'latency_ms': self.config.latency_ms,
            'max_polyphony': self.config.max_polyphony
        }
        return stats
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_notes_played': 0,
            'active_voices_count': len(self.active_voices),
            'max_voices_used': 0,
            'audio_dropouts': 0,
            'average_latency_ms': 0.0,
            'cpu_usage': 0.0
        }


# 便利関数

def create_audio_synthesizer(
    sample_rate: int = 44100,
    buffer_size: int = 256,
    max_polyphony: int = 32
) -> AudioSynthesizer:
    """
    音響シンセサイザーを作成（簡単なインターフェース）
    
    Args:
        sample_rate: サンプリングレート
        buffer_size: バッファサイズ
        max_polyphony: 最大ポリフォニー
        
    Returns:
        設定されたシンセサイザー
    """
    config = AudioConfig(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        max_polyphony=max_polyphony
    )
    
    return AudioSynthesizer(config=config)


def play_audio_immediately(
    params: AudioParameters,
    synthesizer: Optional[AudioSynthesizer] = None
) -> Optional[str]:
    """
    音響パラメータを即座に再生
    
    Args:
        params: 音響パラメータ
        synthesizer: 使用するシンセサイザー（Noneの場合は新規作成）
        
    Returns:
        ボイスID
    """
    if synthesizer is None:
        synthesizer = create_audio_synthesizer()
        if not synthesizer.start_engine():
            return None
    
    return synthesizer.play_audio_parameters(params) 