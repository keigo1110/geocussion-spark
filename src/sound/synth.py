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

# 他フェーズとの連携
from .mapping import AudioParameters, InstrumentType
from ..config import get_config, AudioConfig as GlobalAudioConfig
from ..resource_manager import ManagedResource, get_resource_manager
from .. import get_logger

logger = get_logger(__name__)

try:
    import pyo
    from pyo import Biquadx, ButLP
except ImportError:
    logger.warning("pyo not available. Audio synthesis will be disabled.")
    pyo = None


class EngineState(Enum):
    """エンジン状態の列挙"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AudioConfig:
    """音響設定データ（後方互換性のため）"""
    sample_rate: int = 44100        # サンプリングレート
    buffer_size: int = 256          # バッファサイズ
    channels: int = 2               # チャンネル数（ステレオ）
    audio_driver: str = "portaudio"  # オーディオドライバ
    enable_duplex: bool = False     # 全二重通信
    max_polyphony: int = 32         # 最大ポリフォニー数
    master_volume: float = 0.7      # マスターボリューム
    reverb_level: float = 0.3       # リバーブレベル
    
    @classmethod
    def from_global_config(cls, global_config: Optional[GlobalAudioConfig] = None) -> 'AudioConfig':
        """グローバル設定から音響設定を作成"""
        if global_config is None:
            global_config = get_config().audio
        
        return cls(
            sample_rate=global_config.sample_rate,
            buffer_size=global_config.buffer_size,
            channels=global_config.channels,
            audio_driver=global_config.audio_driver,
            enable_duplex=global_config.enable_duplex,
            max_polyphony=global_config.max_polyphony,
            master_volume=global_config.master_volume,
            reverb_level=global_config.reverb_level
        )
    
    @property
    def latency_ms(self) -> float:
        """レイテンシーをミリ秒で取得"""
        return (self.buffer_size / self.sample_rate) * 1000


class AudioSynthesizer(ManagedResource):
    """pyo音響合成エンジン（リソース管理対応）"""
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        enable_physical_modeling: bool = True,
        enable_effects: bool = True,
        enable_spatial_audio: bool = True,
        resource_id: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            config: 音響設定
            enable_physical_modeling: 物理モデリングを有効にするか
            enable_effects: エフェクトを有効にするか
            enable_spatial_audio: 空間音響を有効にするか
            resource_id: リソースID（自動生成される場合はNone）
        """
        # ManagedResourceの初期化
        resource_id = resource_id or f"audio_synth_{int(time.time() * 1000000)}"
        super().__init__(resource_id)
        
        self.config = config or AudioConfig.from_global_config()
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
        
        logger.info(f"AudioSynthesizer initialized: {self.resource_id}, config: {self.config}")
        
        # リソースマネージャーに自動登録
        manager = get_resource_manager()
        manager.register_resource(self, memory_estimate=50 * 1024 * 1024)  # 50MB推定
    
    def initialize(self) -> bool:
        """リソース初期化（ManagedResourceインターフェース）"""
        return self.start_engine()
    
    def start_engine(self) -> bool:
        """
        音響エンジンを開始
        
        Returns:
            成功したかどうか
        """
        if pyo is None:
            logger.error("pyo is not available")
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
            logger.info(f"Audio engine started successfully. Latency: {self.config.latency_ms:.1f}ms")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to start audio engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    def cleanup(self) -> bool:
        """リソースクリーンアップ（ManagedResourceインターフェース）"""
        try:
            self.stop_engine()
            return True
        except Exception as e:
            logger.error(f"Error in cleanup for {self.resource_id}: {e}")
            return False
    
    def stop_engine(self):
        """音響エンジンを停止"""
        try:
            self.state = EngineState.STOPPING
            
            # 全ボイスを停止
            self.stop_all_voices()
            
            # pyoサーバーの完全なクリーンアップ
            if self.server:
                try:
                    # サーバーからすべてのオブジェクトを削除
                    if hasattr(self.server, 'getStreams'):
                        for stream in self.server.getStreams():
                            try:
                                stream.stop()
                            except Exception:
                                pass
                    
                    # サーバー停止と完全シャットダウン
                    self.server.stop()
                    self.server.shutdown()
                    
                    # インスタンス削除
                    del self.server
                    self.server = None
                    
                except Exception as e:
                    logger.error(f"Error during pyo server shutdown: {e}")
            
            # 楽器インスタンスのクリーンアップ
            for instrument_type, instrument in self.instruments.items():
                try:
                    if instrument and hasattr(instrument, 'stop'):
                        instrument.stop()
                except Exception as e:
                    logger.error(f"Error stopping instrument {instrument_type}: {e}")
            self.instruments.clear()
            
            # エフェクトのクリーンアップ
            self.reverb = None
            
            # アクティブボイスの強制クリア
            self.active_voices.clear()
            
            self.state = EngineState.STOPPED
            logger.info("Audio engine stopped and cleaned up")
            
        except Exception as e:
            logger.error(f"Error stopping audio engine: {e}")
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
                logger.exception(f"Error playing audio: {e}")
                return None
    
    def stop_voice(self, voice_id: str):
        """特定のボイスを停止（安全版）"""
        with self._lock:
            if voice_id not in self.active_voices:
                return
                
            try:
                voice_data = self.active_voices[voice_id]
                voice = voice_data.get('voice')
                
                # pyoボイスの安全な停止
                if voice is not None:
                    try:
                        # pyoオブジェクトの停止（複数の方法を試行）
                        if hasattr(voice, 'stop'):
                            voice.stop()
                        if hasattr(voice, 'out'):
                            voice.out(0)  # 出力を停止
                    except Exception as e:
                        logger.error(f"Error stopping pyo voice: {e}")
                
                # 管理から削除
                del self.active_voices[voice_id]
                self.stats['active_voices_count'] = len(self.active_voices)
                
            except Exception as e:
                logger.exception(f"Error stopping voice {voice_id}: {e}")
                # エラーでも削除を試行
                try:
                    if voice_id in self.active_voices:
                        del self.active_voices[voice_id]
                        self.stats['active_voices_count'] = len(self.active_voices)
                except Exception:
                    pass
    
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
            # Pre-EQ (high-shelf −6 dB @ 8 kHz) then Freeverb
            def _reverb_chain(inp):
                try:
                    eq = pyo.Biquadx(inp, freq=8000, q=0.707, type=5, gain=-6)
                except Exception:
                    eq = inp
                return pyo.Freeverb(eq, size=0.8, damp=0.6, bal=self.config.reverb_level)

            self.reverb = _reverb_chain
            
            # サーバーのマスターボリュームを設定
            self.server.setAmp(self.config.master_volume)
            
        except Exception as e:
            logger.exception(f"Error initializing effects: {e}")
    
    def _create_instrument_template(self, instrument_type: InstrumentType) -> Dict:
        """楽器テンプレートを作成"""
        
        templates = {
            InstrumentType.MARIMBA: {
                'oscillator_type': 'fm',
                'carrier_ratio': 1.0,
                'modulator_ratio': 4.0,
                'modulation_index': 1.2,
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
                'modulation_index': 1.0,
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
            
            # 音量適用 + 基準ゲイン
            voice = oscillator * envelope * float(params.velocity) * float(getattr(params, "gain", 1.0))
            
            # パンニング適用（型変換を確実に）
            if self.enable_spatial_audio and self.config.channels == 2:
                voice = pyo.Pan(voice, outs=2, pan=float(params.pan))
            
            # エフェクト適用（型変換確実に）
            if self.enable_effects and self.reverb:
                reverb_send = voice * float(params.reverb)

                # Alias reduction: low-pass reverb send at 10 kHz
                try:
                    reverb_send = pyo.Biquadx(reverb_send, freq=10000, q=0.707, type=0, stages=2)
                except Exception:
                    pass

                reverb_out = self.reverb(reverb_send)
                voice = voice + reverb_out
            
            # 出力開始
            voice.out()
            envelope.play()
            
            # --- Global Limiter (simple hard clip) ------------------------
            try:
                if not hasattr(self, "_global_limiter"):
                    # Create a compressor/limiter on server out when first voice created
                    self._global_limiter = pyo.Clip(
                        self.server.getOutput(), min=-0.95, max=0.95
                    ).out()
            except Exception:
                pass
            
            return voice
            
        except Exception as e:
            logger.exception(f"Error creating voice: {e}")
            return None
    
    def _create_oscillator(self, params: AudioParameters, template: Dict, freq: float) -> Any:
        """楽器テンプレートに基づいてオシレーターを作成"""
        
        osc_type = template.get('oscillator_type', 'sine')
        
        if osc_type == 'sine':
            return pyo.Sine(freq=freq)
            
        elif osc_type == 'saw':
            saw = pyo.Saw(freq=freq)
            return ButLP(saw, freq=8000)
            
        elif osc_type == 'fm':
            carrier_ratio = template.get('carrier_ratio', 1.0)
            modulator_ratio = template.get('modulator_ratio', 2.0)
            mod_index = template.get('modulation_index', 1.0)
            
            modulator = pyo.Sine(freq=freq * modulator_ratio)
            fm = pyo.Sine(freq=freq * carrier_ratio + modulator * mod_index)
            # Anti-alias: low-pass at 8 kHz
            return Biquadx(fm, freq=8000, q=0.707, type=0)
            
        elif osc_type == 'noise':
            noi = pyo.Noise()
            return ButLP(noi, freq=8000)
            
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
        """終了したボイスをクリーンアップ（安全版）"""
        if self.state != EngineState.RUNNING:
            return
            
        finished_voices = []
        
        # 1. ロック内で終了ボイスを特定
        with self._lock:
            try:
                current_time = time.perf_counter()
                for voice_id, voice_data in list(self.active_voices.items()):
                    if voice_data is None:
                        finished_voices.append(voice_id)
                        continue
                    
                    try:
                        elapsed = current_time - voice_data['start_time']
                        duration = voice_data.get('duration', 1.0)
                        if elapsed >= duration + 0.1:  # 100ms余裕
                            finished_voices.append(voice_id)
                    except Exception:
                        # 時間計算エラーの場合も削除
                        finished_voices.append(voice_id)
            except Exception as e:
                logger.debug(f"Error during voice scan: {e}")
                voices_to_remove = []
                for voice_id, voice_data in list(self.active_voices.items()):
                    if voice_data is None:
                        voices_to_remove.append(voice_id)
                        continue
                    
                    try:
                        elapsed = time.perf_counter() - voice_data['start_time']
                        duration = voice_data.get('duration', 1.0)
                        if elapsed >= duration + 0.1:  # 100ms余裕
                            voices_to_remove.append(voice_id)
                    except Exception:
                        # 時間計算エラーの場合も削除
                        voices_to_remove.append(voice_id)
        
        # 問題のあるボイスを削除
        for voice_id in voices_to_remove:
            try:
                self.stop_voice(voice_id)
            except Exception as e:
                logger.error(f"Error stopping voice {voice_id}: {e}")
    
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
    
    @property
    def resource_type(self) -> str:
        """リソースタイプ（ManagedResourceインターフェース）"""
        return "audio_synthesizer"
    
    def get_memory_usage(self) -> int:
        """メモリ使用量を取得（ManagedResourceインターフェース）"""
        # 概算値の計算
        base_memory = 10 * 1024 * 1024  # 10MB基本
        voice_memory = len(self.active_voices) * 1024 * 1024  # 1MB/ボイス
        instrument_memory = len(self.instruments) * 5 * 1024 * 1024  # 5MB/楽器
        return base_memory + voice_memory + instrument_memory


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