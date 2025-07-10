#!/usr/bin/env python3
"""
音響生成 - pygame ベースシンセサイザーエンジン

pyoの代わりにpygameを使用したリアルタイム音響合成エンジン。
instruments.pyのInstrumentSynthesizerを活用して複数楽器の確実な音色を提供します。
"""

import time
import threading
import math
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
import numpy as np

# 他フェーズとの連携
from .mapping import AudioParameters, InstrumentType
from .. import get_logger

# pygame音響システム
try:
    import pygame
    import pygame.mixer
    pygame_available = True
except ImportError:
    logger = get_logger(__name__)
    logger.warning("pygame not available. Audio synthesis will be disabled.")
    pygame_available = False

# ロギング設定
logger = get_logger(__name__)


class EngineState(Enum):
    """エンジン状態"""
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class AudioConfig:
    """音響設定データ（pygame版）"""
    sample_rate: int = 44100        # サンプリングレート
    buffer_size: int = 512          # バッファサイズ
    channels: int = 2               # チャンネル数（ステレオ）
    master_volume: float = 0.7      # マスターボリューム
    max_channels: int = 32          # 最大同時チャンネル数


class InstrumentSynthesizer:
    """楽器シンセサイザー（instruments.pyから移植・簡略化）"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def generate_wave(self, instrument: InstrumentType, frequency: float, 
                     duration: float = 2.0, velocity: float = 0.7) -> np.ndarray:
        """指定された楽器の波形を生成"""
        
        # 時間軸
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # 楽器ごとの波形生成
        if instrument == InstrumentType.MARIMBA:
            wave = self._marimba(t, frequency)
        elif instrument == InstrumentType.SYNTH_PAD:
            wave = self._synth_pad(t, frequency)
        elif instrument == InstrumentType.BELL:
            wave = self._bell(t, frequency)
        elif instrument == InstrumentType.CRYSTAL:
            wave = self._crystal(t, frequency)
        elif instrument == InstrumentType.DRUM:
            wave = self._drum(t, frequency)
        elif instrument == InstrumentType.WATER_DROP:
            wave = self._water_drop(t, frequency)
        elif instrument == InstrumentType.WIND:
            wave = self._wind(t, frequency)
        elif instrument == InstrumentType.STRING:
            wave = self._string(t, frequency)
        else:
            # デフォルト：サイン波
            wave = np.sin(2 * np.pi * frequency * t)
        
        # 音量調整
        wave = wave * velocity
        
        # クリッピング防止
        wave = np.clip(wave, -1.0, 1.0)
        
        # 16ビット整数に変換
        wave = (wave * 32767 * 0.8).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo
    
    def _marimba(self, t, freq):
        """マリンバ：木製の温かい音"""
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.4 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.15 * np.sin(2 * np.pi * freq * 3 * t)
        wave = fundamental + h2 + h3
        
        # エンベロープ（木琴特有の減衰）
        env = np.exp(-3 * t) * (1 - np.exp(-500 * t))
        return wave * env
    
    def _synth_pad(self, t, freq):
        """シンセパッド：厚みのある持続音"""
        # デチューンした複数のオシレーター
        osc1 = np.sin(2 * np.pi * freq * t)
        osc2 = np.sin(2 * np.pi * freq * 1.01 * t)
        osc3 = np.sin(2 * np.pi * freq * 0.99 * t)
        
        wave = (osc1 + 0.7 * osc2 + 0.7 * osc3) / 2.4
        
        # ゆっくりしたエンベロープ
        env = (1 - np.exp(-2 * t)) * np.exp(-0.5 * t)
        return wave * env
    
    def _bell(self, t, freq):
        """ベル：FM合成風"""
        mod_freq = freq * 3.5
        mod_amp = freq * 0.5 * np.exp(-2 * t)
        modulator = mod_amp * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * (freq + modulator) * t)
        
        # 倍音
        h2 = 0.3 * np.sin(2 * np.pi * freq * 2.1 * t)
        wave = carrier + h2
        
        env = np.exp(-1.5 * t) * (1 - np.exp(-100 * t))
        return wave * env
    
    def _crystal(self, t, freq):
        """クリスタル：澄んだ音"""
        # 純正律の倍音
        h1 = np.sin(2 * np.pi * freq * t)
        h2 = 0.7 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.5 * np.sin(2 * np.pi * freq * 3 * t)
        h4 = 0.3 * np.sin(2 * np.pi * freq * 4 * t)
        h5 = 0.2 * np.sin(2 * np.pi * freq * 5 * t)
        
        wave = h1 + h2 + h3 + h4 + h5
        
        # キラキラ感
        shimmer = 1 + 0.05 * np.sin(2 * np.pi * 7 * t)
        wave *= shimmer
        
        env = np.exp(-2 * t) * (1 - np.exp(-300 * t))
        return wave * env * 0.5
    
    def _drum(self, t, freq):
        """ドラム：打撃音"""
        # トーン成分（低音）
        tone = np.sin(2 * np.pi * 60 * t)
        
        # ノイズ成分
        noise = np.random.normal(0, 1, len(t))
        
        # ピッチベンド
        pitch_env = np.exp(-20 * t)
        bent_tone = np.sin(2 * np.pi * 60 * (1 + 2 * pitch_env) * t)
        
        wave = 0.5 * bent_tone + 0.5 * noise
        
        # パンチのあるエンベロープ
        env = np.exp(-15 * t)
        return wave * env
    
    def _water_drop(self, t, freq):
        """水滴：ピッチベンド"""
        # 周波数が下がる
        bent_freq = freq * (1 + 0.5 * np.exp(-10 * t))
        wave = np.sin(2 * np.pi * bent_freq * t)
        
        # 水の響き
        resonance = 0.3 * np.sin(2 * np.pi * bent_freq * 2.1 * t)
        wave += resonance
        
        env = np.exp(-5 * t) * (1 - np.exp(-200 * t))
        return wave * env
    
    def _wind(self, t, freq):
        """風：フィルターノイズ"""
        # ホワイトノイズ
        noise = np.random.normal(0, 1, len(t))
        
        # 簡易バンドパスフィルター（周波数に依存）
        cutoff = 0.1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        filtered = noise * cutoff
        
        # トーン成分を少し追加
        tone = 0.1 * np.sin(2 * np.pi * freq * t)
        
        wave = filtered + tone
        env = (1 - np.exp(-1 * t)) * np.exp(-0.5 * t)
        return wave * env * 0.3
    
    def _string(self, t, freq):
        """弦楽器：豊かな倍音"""
        # 複数の倍音
        harmonics = []
        for i in range(1, 8):
            amp = 1.0 / i
            harmonics.append(amp * np.sin(2 * np.pi * freq * i * t))
        
        wave = sum(harmonics)
        
        # ボウイング（弓）効果
        env = (1 - np.exp(-5 * t)) * np.exp(-0.5 * t)
        return wave * env * 0.3


class PygameAudioSynthesizer:
    """pygame ベースの音響合成エンジン"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        初期化
        
        Args:
            config: 音響設定
        """
        self.config = config or AudioConfig()
        self.state = EngineState.STOPPED
        self.active_voices: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._voice_counter = 0
        
        # インストゥルメント・シンセサイザー
        self.instrument_synth = InstrumentSynthesizer(self.config.sample_rate)
        
        # 音源キャッシュ（事前生成による高速化）
        self.sound_cache: Dict[str, Any] = {}
        
        logger.info(f"PygameAudioSynthesizer initialized with config: {self.config}")
    
    def start_engine(self) -> bool:
        """音響エンジンを開始"""
        if not pygame_available:
            logger.error("pygame is not available")
            self.state = EngineState.ERROR
            return False
        
        if self.state == EngineState.RUNNING:
            logger.warning("Engine already running")
            return True
        
        try:
            # pygame.mixer初期化
            pygame.mixer.pre_init(
                frequency=self.config.sample_rate,
                size=-16,  # 16ビット signed
                channels=self.config.channels,
                buffer=self.config.buffer_size
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(self.config.max_channels)
            
            self.state = EngineState.RUNNING
            logger.info("pygame audio engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pygame audio engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    def stop_engine(self):
        """音響エンジンを停止"""
        if self.state != EngineState.RUNNING:
            return
        
        try:
            # 全ボイスを停止
            self._stop_all_voices()
            
            # pygame.mixer停止
            pygame.mixer.stop()
            pygame.mixer.quit()
            
            # キャッシュクリア
            self.sound_cache.clear()
            
            self.state = EngineState.STOPPED
            logger.info("pygame audio engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pygame audio engine: {e}")
    
    # ------------------------------------------------------------------
    # Public API (backward-compat)
    # ------------------------------------------------------------------

    def play_audio_parameters(self, params: AudioParameters) -> Optional[str]:
        """Legacy alias – thin wrapper around play_note."""
        return self.play_note(params)

    def play_note(self, params: AudioParameters) -> Optional[str]:
        """
        単音を再生
        
        Args:
            params: 音響パラメータ
            
        Returns:
            ボイスID（成功した場合）
        """
        if self.state != EngineState.RUNNING:
            logger.warning("Engine not running")
            return None
        
        with self._lock:
            try:
                # ボイスID生成
                self._voice_counter += 1
                voice_id = f"voice_{self._voice_counter}"
                
                # 音源生成またはキャッシュから取得
                sound = self._get_or_create_sound(params)
                if sound is None:
                    return None
                
                # 空いているチャンネルを探す
                channel = pygame.mixer.find_channel()
                if channel is None:
                    logger.warning("No available channels")
                    return None
                
                # 音量とパンニング設定
                volume = params.velocity * self.config.master_volume
                channel.set_volume(volume)
                
                # 再生開始
                channel.play(sound)
                
                # 管理辞書に追加
                self.active_voices[voice_id] = {
                    'channel': channel,
                    'start_time': time.time(),
                    'duration': params.duration,
                    'params': params
                }
                
                # 自動停止スケジュール
                self._schedule_voice_stop(voice_id, params.duration)
                
                logger.debug(f"Playing note: {params.instrument} @ {params.frequency}Hz")
                return voice_id
                
            except Exception as e:
                logger.error(f"Error playing note: {e}")
                return None
    
    def stop_voice(self, voice_id: str):
        """特定のボイスを停止"""
        with self._lock:
            if voice_id not in self.active_voices:
                return
            
            try:
                voice_data = self.active_voices[voice_id]
                channel = voice_data['channel']
                
                # チャンネル停止
                if channel:
                    channel.stop()
                
                # 管理辞書から削除
                del self.active_voices[voice_id]
                
            except Exception as e:
                logger.error(f"Error stopping voice {voice_id}: {e}")
                # エラーでも削除
                self.active_voices.pop(voice_id, None)
    
    def _stop_all_voices(self):
        """全ボイスを停止"""
        voice_ids = list(self.active_voices.keys())
        for voice_id in voice_ids:
            self.stop_voice(voice_id)
    
    def _get_or_create_sound(self, params: AudioParameters) -> Optional[Any]:
        """音源を取得または生成（キャッシュ機能付き）"""
        # キャッシュキー生成（楽器、周波数、音長で識別）
        cache_key = f"{params.instrument.value}_{params.frequency:.1f}_{params.duration:.2f}"
        
        if cache_key in self.sound_cache:
            return self.sound_cache[cache_key]
        
        try:
            # 新しい音源を生成
            wave_data = self.instrument_synth.generate_wave(
                instrument=params.instrument,
                frequency=params.frequency,
                duration=params.duration,
                velocity=1.0  # ここでは1.0に固定、音量は再生時に調整
            )
            
            # pygame.Soundオブジェクト作成
            sound = pygame.sndarray.make_sound(wave_data)
            
            # キャッシュに保存（メモリ制限のため最大100個まで）
            if len(self.sound_cache) < 100:
                self.sound_cache[cache_key] = sound
            
            return sound
            
        except Exception as e:
            logger.error(f"Error creating sound: {e}")
            return None
    
    def _schedule_voice_stop(self, voice_id: str, duration: float):
        """指定時間後にボイスを停止"""
        def stop_after_delay():
            time.sleep(duration + 0.1)  # 少し余裕を持たせる
            self.stop_voice(voice_id)
        
        # 別スレッドで実行
        thread = threading.Thread(target=stop_after_delay, daemon=True)
        thread.start()
    
    # ------------------------------------------------------------------
    # Extra helper methods required by other modules / tests
    # ------------------------------------------------------------------

    def is_voice_active(self, voice_id: str) -> bool:
        """Return True if the given voice is still active."""
        with self._lock:
            if voice_id not in self.active_voices:
                return False
            voice_data = self.active_voices[voice_id]
            channel = voice_data['channel']
            return channel and channel.get_busy()

    def update_master_volume(self, volume: float) -> None:
        """Update master output gain."""
        self.config.master_volume = max(0.0, min(1.0, float(volume)))
        logger.debug(f"Master volume updated to {self.config.master_volume}")

    def cleanup_finished_voices(self) -> None:
        """終了したボイスをクリーンアップ"""
        with self._lock:
            finished_voices = []
            for voice_id, voice_data in self.active_voices.items():
                channel = voice_data['channel']
                if not channel or not channel.get_busy():
                    finished_voices.append(voice_id)
            
            for voice_id in finished_voices:
                self.active_voices.pop(voice_id, None)

    def get_active_voice_count(self) -> int:
        """アクティブなボイス数を取得"""
        with self._lock:
            return len(self.active_voices)
    
    def is_running(self) -> bool:
        """エンジンが動作中かどうか"""
        return self.state == EngineState.RUNNING


# ----------------------------------------------------------------------
# Legacy aliases – keep other modules working without modifications
# ----------------------------------------------------------------------

# Export alias so that `from src.sound.synth import AudioSynthesizer` keeps
# working after the refactor.
AudioSynthesizer = PygameAudioSynthesizer


def create_audio_synthesizer(
    sample_rate: int = 44100,
    buffer_size: int = 512,
    max_polyphony: int = 32,
) -> "AudioSynthesizer":
    """Legacy factory that mirrors former signature but returns the new engine."""
    
    config = AudioConfig(
        sample_rate=sample_rate, 
        buffer_size=buffer_size,
        max_channels=max_polyphony
    )
    return AudioSynthesizer(config=config)


def create_simple_synthesizer() -> AudioSynthesizer:
    """シンプルなシンセサイザーを作成"""
    return AudioSynthesizer()


def play_audio_immediately(
    params: 'AudioParameters',
    synthesizer: Optional['AudioSynthesizer'] = None,
) -> Optional[str]:
    """Play the given AudioParameters right away, creating a temporary synthesizer if needed."""
    
    # Ensure we have an engine
    engine = synthesizer or create_audio_synthesizer()

    if not engine.is_running():
        if not engine.start_engine():
            return None

    return engine.play_audio_parameters(params)


def play_single_note(
    instrument: InstrumentType,
    frequency: float,
    duration: float = 1.0,
    velocity: float = 0.7
) -> Optional[AudioSynthesizer]:
    """
    単音を即座に再生する最もシンプルな関数
    
    Args:
        instrument: 楽器タイプ
        frequency: 周波数（Hz）
        duration: 音の長さ（秒）
        velocity: 音量（0.0-1.0）
        
    Returns:
        使用したシンセサイザー（後続の制御用）
    """
    # --- Fill full AudioParameters dataclass ---------------------------
    # AudioParameters requires many fields. Map the bare-minimum for CLI
    # usage with sensible defaults so that downstream utilities work.

    pitch_val = 69.0 + 12.0 * math.log2(max(1e-6, frequency) / 440.0)

    params = AudioParameters(
        # 基本
        pitch=pitch_val,
        velocity=velocity,
        duration=duration,
        # 楽器・音色
        instrument=instrument,
        timbre=0.5,
        brightness=0.5,
        # 空間
        pan=0.0,
        distance=0.5,
        reverb=0.0,
        # エンベロープ
        attack=0.01,
        decay=0.1,
        sustain=0.7,
        release=0.2,
        # メタ
        event_id="quick_play",
        hand_id="cli",
        timestamp=time.time(),
        gain=1.0,
    )
    
    # シンセサイザー作成・起動
    synth = create_simple_synthesizer()
    if not synth.start_engine():
        logger.error("Failed to start synthesizer")
        return None
    
    # 音を再生
    voice_id = synth.play_note(params)
    if voice_id:
        logger.info(f"Playing {instrument.value} at {frequency}Hz for {duration}s")
        return synth
    else:
        synth.stop_engine()
        return None


# 使用例
if __name__ == "__main__":
    # マリンバでC4（261.63Hz）を2秒間再生
    synth = play_single_note(
        instrument=InstrumentType.MARIMBA,
        frequency=261.63,
        duration=2.0,
        velocity=0.8
    )
    
    if synth:
        # 音が鳴り終わるまで待機
        time.sleep(2.5)
        synth.stop_engine()