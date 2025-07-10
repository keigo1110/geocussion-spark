#!/usr/bin/env python3
"""
シンプルなpygame音響合成システム

s_test.pyで動作確認済みのパターンをベースにした、
確実に音が鳴るシンプルな実装です。
"""

import time
import threading
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 他フェーズとの連携
from .mapping import AudioParameters, InstrumentType
from .. import get_logger

# pygame音響システム
try:
    import pygame
    import pygame.mixer
    pygame_available = True
except ImportError:
    pygame_available = False

# ロギング設定
logger = get_logger(__name__)

# 音階の周波数定義（s_test.pyから移植）
NOTES_FREQUENCIES = {
    # 第3オクターブ
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
    'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
    # 第4オクターブ
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
    # 第5オクターブ
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
    'G5': 783.99, 'A5': 880.00, 'B5': 987.77, 'C6': 1046.50,
}


class SimpleAudioState(Enum):
    """シンプル音響システムの状態"""
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class SimpleAudioConfig:
    """シンプル音響設定"""
    sample_rate: int = 44100
    buffer_size: int = 512
    channels: int = 2
    master_volume: float = 0.7
    max_channels: int = 32


class SimpleSoundBank:
    """音源バンク（s_test.pyから移植・改良）"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sounds: Dict[str, Any] = {}
        self._generate_all_instruments()
    
    def _generate_marimba_wave(self, frequency: float, duration: float = 2.0) -> np.ndarray:
        """マリンバ波形を生成"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # マリンバの音色（基音＋倍音）
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonic2 = 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        harmonic3 = 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
        harmonic4 = 0.05 * np.sin(2 * np.pi * frequency * 4 * t)
        
        wave = fundamental + harmonic2 + harmonic3 + harmonic4
        
        # エンベロープ（マリンバ特有の減衰）
        attack_time = 0.005
        decay_time = 0.15
        
        envelope = np.ones_like(t)
        attack_samples = int(attack_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        
        # アタック
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # ディケイ〜サステイン
        decay_end = attack_samples + decay_samples
        if decay_end < len(envelope):
            envelope[attack_samples:decay_end] = np.linspace(1, 0.2, decay_samples)
            # 指数減衰
            remaining = len(envelope) - decay_end
            if remaining > 0:
                envelope[decay_end:] = 0.2 * np.exp(-2 * np.linspace(0, 4, remaining))
        
        # 適用
        wave = wave * envelope * 0.3
        
        # 16ビット整数に変換
        wave = (wave * 32767).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo
    
    def _generate_bell_wave(self, frequency: float, duration: float = 2.0) -> np.ndarray:
        """ベル波形を生成"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # FM合成風
        mod_freq = frequency * 3.5
        mod_amp = frequency * 0.5 * np.exp(-2 * t)
        modulator = mod_amp * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * (frequency + modulator) * t)
        
        # 倍音
        h2 = 0.3 * np.sin(2 * np.pi * frequency * 2.1 * t)
        wave = carrier + h2
        
        # エンベロープ
        env = np.exp(-1.5 * t) * (1 - np.exp(-100 * t))
        wave = wave * env * 0.4
        
        # 16ビット整数に変換
        wave = (wave * 32767).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo
    
    def _generate_pad_wave(self, frequency: float, duration: float = 2.0) -> np.ndarray:
        """シンセパッド波形を生成"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # デチューンした複数のオシレーター
        osc1 = np.sin(2 * np.pi * frequency * t)
        osc2 = np.sin(2 * np.pi * frequency * 1.01 * t)
        osc3 = np.sin(2 * np.pi * frequency * 0.99 * t)
        
        wave = (osc1 + 0.7 * osc2 + 0.7 * osc3) / 2.4
        
        # ゆっくりしたエンベロープ
        env = (1 - np.exp(-2 * t)) * np.exp(-0.5 * t)
        wave = wave * env * 0.5
        
        # 16ビット整数に変換
        wave = (wave * 32767).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo
    
    def _generate_drum_wave(self, frequency: float, duration: float = 0.5) -> np.ndarray:
        """ドラム波形を生成"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # ノイズベース（キックドラム風）
        noise = np.random.uniform(-1, 1, len(t))
        
        # 低周波成分（キック）
        kick = np.sin(2 * np.pi * frequency * 0.5 * t) 
        kick += 0.3 * np.sin(2 * np.pi * frequency * 0.3 * t)
        
        # 高周波成分（スネア風）
        snare = noise * 0.6
        
        # 組み合わせ
        wave = kick * 0.7 + snare * 0.3
        
        # 急激な減衰エンベロープ（ドラム特有）
        env = np.exp(-8 * t) * (1 - np.exp(-50 * t))
        wave = wave * env * 0.6
        
        # 16ビット整数に変換
        wave = (wave * 32767).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo

    def _generate_all_instruments(self):
        """全楽器の音源を生成"""
        logger.info("Generating simple sound bank...")
        
        # 基本的な周波数セット
        frequencies = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,  # C3-B3
                      261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,  # C4-B4
                      523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,  # C5-B5
                      1046.50]  # C6
        
        for freq in frequencies:
            # マリンバ
            marimba_key = f"MARIMBA_{freq:.0f}"
            self.sounds[marimba_key] = pygame.sndarray.make_sound(
                self._generate_marimba_wave(freq, duration=1.5)
            )
            
            # ベル
            bell_key = f"BELL_{freq:.0f}"
            self.sounds[bell_key] = pygame.sndarray.make_sound(
                self._generate_bell_wave(freq, duration=2.0)
            )
            
            # シンセパッド
            pad_key = f"SYNTH_PAD_{freq:.0f}"
            self.sounds[pad_key] = pygame.sndarray.make_sound(
                self._generate_pad_wave(freq, duration=2.5)
            )
            
            # ドラム
            drum_key = f"DRUM_{freq:.0f}"
            self.sounds[drum_key] = pygame.sndarray.make_sound(
                self._generate_drum_wave(freq, duration=0.8)
            )
        
        logger.info(f"Generated {len(self.sounds)} sounds in bank")
    
    def get_sound(self, instrument: InstrumentType, frequency: float) -> Optional[Any]:
        """音源を取得（フォールバック機能付き）"""
        key = f"{instrument.value.upper()}_{frequency:.0f}"
        sound = self.sounds.get(key)
        
        # 見つからない場合は近い周波数で検索
        if sound is None:
            # まず、同じ楽器で近い周波数を探す
            instrument_prefix = f"{instrument.value.upper()}_"
            available_keys = [k for k in self.sounds.keys() if k.startswith(instrument_prefix)]
            
            if available_keys:
                # 最も近い周波数を探す
                target_freq = frequency
                best_key = min(available_keys, 
                             key=lambda k: abs(float(k.split('_')[1]) - target_freq))
                sound = self.sounds.get(best_key)
                logger.debug(f"Using fallback sound {best_key} for {key}")
            
            # それでも見つからない場合はマリンバにフォールバック
            if sound is None:
                fallback_key = f"MARIMBA_{frequency:.0f}"
                sound = self.sounds.get(fallback_key)
                if sound is None:
                    # 最も近いマリンバ音を探す
                    marimba_keys = [k for k in self.sounds.keys() if k.startswith("MARIMBA_")]
                    if marimba_keys:
                        best_key = min(marimba_keys, 
                                     key=lambda k: abs(float(k.split('_')[1]) - frequency))
                        sound = self.sounds.get(best_key)
                        logger.debug(f"Using marimba fallback {best_key} for {key}")
        
        return sound


class SimpleAudioSynthesizer:
    """シンプルなpygame音響合成システム"""
    
    def __init__(self, config: Optional[SimpleAudioConfig] = None):
        """初期化"""
        self.config = config or SimpleAudioConfig()
        self.state = SimpleAudioState.STOPPED
        self.sound_bank: Optional[SimpleSoundBank] = None
        self.active_channels: Dict[str, Any] = {}
        self._voice_counter = 0
        self._lock = threading.Lock()
        
        logger.info(f"SimpleAudioSynthesizer initialized with {self.config}")
    
    def start_engine(self) -> bool:
        """音響エンジンを開始"""
        if not pygame_available:
            logger.error("pygame is not available")
            self.state = SimpleAudioState.ERROR
            return False
        
        if self.state == SimpleAudioState.RUNNING:
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
            
            # 音源バンク初期化
            self.sound_bank = SimpleSoundBank(self.config.sample_rate)
            
            self.state = SimpleAudioState.RUNNING
            logger.info("Simple pygame audio engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start simple pygame audio engine: {e}")
            self.state = SimpleAudioState.ERROR
            return False
    
    def stop_engine(self):
        """音響エンジンを停止"""
        if self.state != SimpleAudioState.RUNNING:
            return
        
        try:
            # 全チャンネル停止
            pygame.mixer.stop()
            pygame.mixer.quit()
            
            # リソースクリア
            self.sound_bank = None
            self.active_channels.clear()
            
            self.state = SimpleAudioState.STOPPED
            logger.info("Simple pygame audio engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping simple pygame audio engine: {e}")
    
    def play_audio_parameters(self, params: AudioParameters) -> Optional[str]:
        """音響パラメータから音を再生"""
        if self.state != SimpleAudioState.RUNNING or self.sound_bank is None:
            logger.warning(f"Engine not ready: state={self.state}, sound_bank={self.sound_bank is not None}")
            return None
        
        with self._lock:
            try:
                # ボイスID生成
                self._voice_counter += 1
                voice_id = f"simple_voice_{self._voice_counter}"
                
                # 音源取得
                sound = self.sound_bank.get_sound(params.instrument, params.frequency)
                if sound is None:
                    logger.warning(f"No sound found for {params.instrument} @ {params.frequency:.1f}Hz")
                    return None
                
                # 空いているチャンネルを探す
                channel = pygame.mixer.find_channel()
                if channel is None:
                    logger.warning("No available channels")
                    return None
                
                # 音量設定
                volume = params.velocity * self.config.master_volume
                channel.set_volume(min(1.0, max(0.0, volume)))
                
                # 再生開始
                channel.play(sound)
                
                # アクティブチャンネル管理
                self.active_channels[voice_id] = {
                    'channel': channel,
                    'start_time': time.time(),
                    'params': params
                }
                
                logger.debug(f"Simple audio: {params.instrument} @ {params.frequency:.1f}Hz, vel={params.velocity:.2f}")
                return voice_id
                
            except Exception as e:
                logger.error(f"Error playing simple audio: {e}")
                return None
    
    def stop_voice(self, voice_id: str):
        """特定のボイスを停止"""
        with self._lock:
            if voice_id not in self.active_channels:
                return
            
            try:
                channel_data = self.active_channels[voice_id]
                channel = channel_data['channel']
                
                if channel:
                    channel.stop()
                
                del self.active_channels[voice_id]
                
            except Exception as e:
                logger.error(f"Error stopping simple voice {voice_id}: {e}")
                self.active_channels.pop(voice_id, None)
    
    def cleanup_finished_voices(self):
        """完了したボイスをクリーンアップ"""
        with self._lock:
            finished_voices = []
            for voice_id, channel_data in self.active_channels.items():
                channel = channel_data['channel']
                if channel and not channel.get_busy():
                    finished_voices.append(voice_id)
            
            for voice_id in finished_voices:
                self.active_channels.pop(voice_id, None)
    
    def update_master_volume(self, volume: float):
        """マスターボリューム更新"""
        self.config.master_volume = min(1.0, max(0.0, volume))
        logger.debug(f"Master volume updated to {self.config.master_volume:.2f}")
    
    def get_active_voice_count(self) -> int:
        """アクティブボイス数を取得"""
        return len(self.active_channels)
    
    def is_running(self) -> bool:
        """エンジンが動作中か"""
        return self.state == SimpleAudioState.RUNNING


def create_simple_audio_synthesizer(
    sample_rate: int = 44100,
    buffer_size: int = 512,
    max_polyphony: int = 32
) -> SimpleAudioSynthesizer:
    """シンプル音響合成器を作成"""
    config = SimpleAudioConfig(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        max_channels=max_polyphony
    )
    return SimpleAudioSynthesizer(config)


def test_simple_synthesizer():
    """シンプル音響合成器のテスト"""
    from .mapping import AudioParameters, InstrumentType
    
    print("Testing SimpleAudioSynthesizer...")
    
    # 合成器作成
    synth = create_simple_audio_synthesizer()
    
    # エンジン開始
    if not synth.start_engine():
        print("Failed to start engine")
        return False
    
    # テスト音の再生
    test_params = AudioParameters(
        pitch=60.0,  # C4
        velocity=0.7,
        duration=1.0,
        instrument=InstrumentType.MARIMBA,
        timbre=0.5,
        brightness=0.5,
        pan=0.0,
        distance=0.5,
        reverb=0.3,
        attack=0.01,
        decay=0.1,
        sustain=0.7,
        release=0.5,
        event_id="test_event",
        hand_id="test_hand",
        timestamp=time.time()
    )
    
    voice_id = synth.play_audio_parameters(test_params)
    if voice_id:
        print(f"Test audio played successfully with voice_id: {voice_id}")
        time.sleep(2.0)  # 音が鳴るまで待つ
        synth.stop_engine()
        return True
    else:
        print("Failed to play test audio")
        synth.stop_engine()
        return False


if __name__ == "__main__":
    test_simple_synthesizer() 