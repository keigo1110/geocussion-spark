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
import copy

try:
    import pyo
except ImportError:
    print("Warning: pyo not available. Audio synthesis will be disabled.")
    pyo = None

# 他フェーズとの連携
from .mapping import AudioParameters
from ..utils import config_manager
from ..utils.config import settings


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
        enable_physical_modeling: bool = True,
        enable_effects: bool = True,
        enable_spatial_audio: bool = True
    ):
        """
        初期化
        
        Args:
            enable_physical_modeling: 物理モデリングを有効にするか
            enable_effects: エフェクトを有効にするか
            enable_spatial_audio: 空間音響を有効にするか
        """
        self.audio_config = settings.get('audio', {})
        
        self.internal_config = AudioConfig(
            master_volume=self.audio_config.get('master_volume', 0.7),
            max_polyphony=self.audio_config.get('polyphony', 16)
        )
        
        self.enable_physical_modeling = enable_physical_modeling
        self.enable_effects = enable_effects
        self.enable_spatial_audio = enable_spatial_audio
        
        # pyo関連
        self.server: Optional['pyo.Server'] = None
        self.state = EngineState.STOPPED
        
        # 楽器インスタンス
        self.instrument_templates: Dict[str, Dict] = {}
        self.active_voices: Dict[str, Dict] = {}  # voice_id -> voice_data
        
        # エフェクト
        self.reverb: Optional['pyo.Freeverb'] = None
        
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
        
        if not pyo:
            self.state = EngineState.ERROR
            print("AudioSynthesizer disabled because pyo is not available.")
        else:
            # Pyoサーバーが起動する前にテンプレートを初期化
            self._initialize_instrument_templates()
            print(f"AudioSynthesizer initialized with config: {self.internal_config}")
    
    def start_engine(self) -> bool:
        """
        音響エンジンを開始
        
        Returns:
            成功したかどうか
        """
        if not pyo or self.server is not None:
            self.state = EngineState.ERROR
            return False
        
        try:
            self.state = EngineState.STARTING
            
            # pyoサーバー初期化
            self.server = pyo.Server(
                sr=self.internal_config.sample_rate,
                nchnls=self.internal_config.channels,
                buffersize=self.internal_config.buffer_size,
                duplex=int(self.internal_config.enable_duplex),
                audio=self.internal_config.audio_driver
            )
            
            # サーバー起動順序修正: boot() -> start()
            self.server.boot()
            self.server.start()
            
            # エフェクトの初期化 (サーバー起動後)
            self._initialize_effects()
            
            self.state = EngineState.RUNNING
            print(f"Audio engine started successfully. Latency: {self.internal_config.latency_ms:.1f}ms")
            if self.server:
                self.server.setAmp(self.internal_config.master_volume)
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
        if not pyo or self.state != EngineState.RUNNING:
            return None
        
        with self._lock:
            try:
                # ボイスID生成
                voice_id = f"{params.event_id}_{int(time.time() * 1000000)}"
                
                # 楽器テンプレートを検索
                instrument_name = params.instrument.upper()
                template = self.instrument_templates.get(instrument_name)

                if template is None:
                    print(f"Warning: Instrument template '{instrument_name}' not found.")
                    return None

                # 楽器に応じたボイス生成
                voice = self._create_voice(params, template)
                if voice is None:
                    return None
                
                # ボイス管理に追加
                self.active_voices[voice_id] = voice
                
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
    
    def stop_voice(self, voice_id: str, fadeout_sec: float = 0.0):
        """
        特定のボイスを停止

        Args:
            voice_id (str): 停止するボイスのID
            fadeout_sec (float, optional): フェードアウト時間(秒). Defaults to 0.0.
        """
        with self._lock:
            if voice_id in self.active_voices:
                try:
                    voice_data = self.active_voices.pop(voice_id) # 先に削除
                    voice = voice_data['source']
                    
                    if fadeout_sec > 0:
                        # PyoObjectのout()メソッドで滑らかに停止
                        if hasattr(voice, 'out'):
                            voice.out(dur=fadeout_sec)
                    else:
                        if hasattr(voice, 'stop'):
                            voice.stop()
                    
                    self.stats['active_voices_count'] = len(self.active_voices)
                    
                except Exception as e:
                    print(f"Error stopping voice {voice_id}: {e}")
    
    def stop_all_voices(self, fadeout_sec: float = 0.05):
        """
        全ボイスを停止

        Args:
            fadeout_sec (float, optional): フェードアウト時間(秒). Defaults to 0.05.
        """
        with self._lock:
            voice_ids = list(self.active_voices.keys())
            for voice_id in voice_ids:
                self.stop_voice(voice_id, fadeout_sec=fadeout_sec)
    
    def update_master_volume(self, volume: float):
        """マスターボリューム更新"""
        self.internal_config.master_volume = max(0.0, min(1.0, volume))
        if self.server:
            self.server.setAmp(self.internal_config.master_volume)
    
    def _initialize_instrument_templates(self):
        """設定ファイルから楽器テンプレートを初期化"""
        if 'instruments' in self.audio_config:
            self.instrument_templates = self.audio_config['instruments']
            print(f"Loaded {len(self.instrument_templates)} instrument templates from config.")
        else:
            print("Warning: No instrument definitions found in config.yaml")
    
    def _initialize_effects(self):
        """エフェクトを初期化"""
        if not self.server or not pyo:
            return
        
        try:
            # マスターリバーブ
            reverb_level = self.audio_config.get('reverb_level', 0.2)
            self.reverb = pyo.Freeverb(
                self.server.getBootedSources(),
                size=0.8, damp=0.5, bal=reverb_level
            ).out()
            
            # サーバーのマスターボリュームを設定
            self.server.setAmp(self.internal_config.master_volume)
            
        except Exception as e:
            print(f"Error initializing effects: {e}")
    
    def _create_voice(self, params: AudioParameters, template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """オーディオパラメータに基づいてボイスを生成"""
        if not pyo:
            return None

        # pyoオブジェクト名を取得
        pyo_object_name = template.get("pyo_object")
        if not pyo_object_name or not hasattr(pyo, pyo_object_name):
            print(f"Error: pyo object '{pyo_object_name}' not found.")
            return None
        
        pyo_class = getattr(pyo, pyo_object_name)

        # パラメータを構築
        voice_params = copy.deepcopy(template.get("default_params", {}))
        voice_params['freq'] = params.frequency
        voice_params['mul'] = params.velocity

        # ADSRエンベロープを作成
        envelope = pyo.Adsr(
            attack=params.attack,
            decay=params.decay,
            sustain=params.sustain,
            release=params.release,
            dur=params.duration,
            mul=voice_params.get('mul', params.velocity)
        )
        # 再生終了後に自身をクリーンアップするコールバックを設定
        envelope.setCallback(lambda: self.cleanup_voice(params.event_id))

        voice_params['mul'] = envelope

        # Pyoオブジェクトをインスタンス化
        try:
            sound_source = pyo_class(**voice_params)
        except Exception as e:
            print(f"Error creating pyo object '{pyo_object_name}' with params {voice_params}: {e}")
            return None

        # 空間配置
        if self.enable_spatial_audio:
            panner = pyo.Pan(sound_source, outs=2, pan=params.pan, spread=0.5)
        else:
            panner = sound_source.mix(2)

        # リバーブ処理
        if self.enable_effects and self.reverb:
            # Pyoオブジェクトを直接リバーブに渡す
            final_output = self.reverb(panner).mix(2)
        else:
            final_output = panner

        final_output.out()

        return {
            "source": sound_source,
            "envelope": envelope,
            "panner": panner,
            "output": final_output,
            "timestamp": time.perf_counter(),
            "event_id": params.event_id
        }
    
    def cleanup_finished_voices(self):
        """再生が完了したボイスをクリーンアップ"""
        if not pyo:
            return
        # このメソッドはAdsrのコールバックから直接呼ばれるようになったため、
        # 定期的な呼び出しは不要になった
        pass

    def cleanup_voice(self, event_id: str):
        """特定のイベントIDに紐づくボイスをクリーンアップ"""
        with self._lock:
            # event_idに部分一致するボイスを探す
            voices_to_remove = [
                vid for vid in self.active_voices 
                if self.active_voices[vid].get("event_id") == event_id
            ]
            
            for voice_id in voices_to_remove:
                if voice_id in self.active_voices:
                    # stop()は不要、ADSRが完了しているため
                    del self.active_voices[voice_id]
                    self.stats['active_voices_count'] = len(self.active_voices)
    
    def update_voice_parameters(self, voice_id: str, volume: Optional[float], pan: Optional[float], reverb: Optional[float]):
        """再生中のボイスのパラメータを更新"""
        if not pyo:
            return
            
        with self._lock:
            voice_data = self.active_voices.get(voice_id)
            if not voice_data:
                return

            if volume is not None and 'envelope' in voice_data:
                voice_data['envelope'].mul = volume
            
            if pan is not None and 'panner' in voice_data:
                voice_data['panner'].pan = pan
            
            # リバーブの動的変更は複雑なため、ここでは未実装
            # if reverb is not None and self.reverb:
            #     pass

    def get_performance_stats(self) -> dict:
        """パフォーマンス統計を取得"""
        with self._lock:
            stats = self.stats.copy()
            stats['engine_state'] = self.state.value
            stats['config'] = {
                'sample_rate': self.internal_config.sample_rate,
                'buffer_size': self.internal_config.buffer_size,
                'latency_ms': self.internal_config.latency_ms,
                'max_polyphony': self.internal_config.max_polyphony
            }
            stats['cpu_usage'] = self.server.getCPU() if self.server else 0.0
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
    enable_physical_modeling: bool = True,
    enable_effects: bool = True,
    enable_spatial_audio: bool = True,
) -> AudioSynthesizer:
    """
    AudioSynthesizerのファクトリ関数
    """
    return AudioSynthesizer(
        enable_physical_modeling=enable_physical_modeling,
        enable_effects=enable_effects,
        enable_spatial_audio=enable_spatial_audio
    )


def play_audio_immediately(
    params: AudioParameters,
    synthesizer: Optional[AudioSynthesizer] = None
) -> Optional[str]:
    """
    音響パラメータを即時再生するヘルパー関数
    """
    if synthesizer is None:
        synthesizer = create_audio_synthesizer()
        if not synthesizer.start_engine():
            return None
    
    return synthesizer.play_audio_parameters(params) 