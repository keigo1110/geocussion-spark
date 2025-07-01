#!/usr/bin/env python3
"""
Null音響バックエンド

音響処理を行わないダミーバックエンド。
CI環境やテスト環境でpyoが利用できない場合に使用します。
"""

import time
import threading
from typing import Dict, Optional, Set
from . import IAudioBackend, BackendType
from ... import get_logger

logger = get_logger(__name__)


class NullAudioBackend(IAudioBackend):
    """音響処理を行わないダミーバックエンド"""
    
    def __init__(self):
        self.initialized = False
        self.running = False
        self.sample_rate = 44100
        self.channels = 2
        self.buffer_size = 256
        
        # ダミーボイス管理
        self.active_voices: Set[str] = set()
        self.voice_counter = 0
        self._lock = threading.Lock()
        
        logger.debug("NullAudioBackend initialized")
    
    def initialize(self, sample_rate: int, channels: int, buffer_size: int, **kwargs) -> bool:
        """バックエンドを初期化"""
        try:
            self.sample_rate = sample_rate
            self.channels = channels
            self.buffer_size = buffer_size
            self.initialized = True
            logger.info(f"NullBackend initialized: {sample_rate}Hz, {channels}ch, {buffer_size} samples")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NullBackend: {e}")
            return False
    
    def start(self) -> bool:
        """音響処理を開始"""
        if not self.initialized:
            logger.error("Backend not initialized")
            return False
        
        self.running = True
        logger.info("NullBackend started (no actual audio processing)")
        return True
    
    def stop(self) -> bool:
        """音響処理を停止"""
        self.running = False
        with self._lock:
            self.active_voices.clear()
        logger.info("NullBackend stopped")
        return True
    
    def shutdown(self) -> bool:
        """バックエンドをシャットダウン"""
        try:
            self.stop()
            self.initialized = False
            logger.info("NullBackend shutdown")
            return True
        except Exception as e:
            logger.error(f"Error shutting down NullBackend: {e}")
            return False
    
    def create_voice(self, frequency: float, amplitude: float, **params) -> Optional[str]:
        """ボイスを作成し、ボイスIDを返す"""
        if not self.running:
            return None
        
        with self._lock:
            self.voice_counter += 1
            voice_id = f"null_voice_{self.voice_counter:06d}"
            self.active_voices.add(voice_id)
        
        logger.debug(f"Created null voice: {voice_id} (freq={frequency:.1f}Hz, amp={amplitude:.3f})")
        return voice_id
    
    def stop_voice(self, voice_id: str) -> bool:
        """指定されたボイスを停止"""
        with self._lock:
            if voice_id in self.active_voices:
                self.active_voices.remove(voice_id)
                logger.debug(f"Stopped null voice: {voice_id}")
                return True
        return False
    
    def is_voice_active(self, voice_id: str) -> bool:
        """ボイスがアクティブかどうか"""
        with self._lock:
            return voice_id in self.active_voices
    
    def get_latency_ms(self) -> float:
        """レイテンシーをミリ秒で取得"""
        return (self.buffer_size / self.sample_rate) * 1000
    
    def get_backend_type(self) -> BackendType:
        """バックエンドタイプを取得"""
        return BackendType.NULL
    
    def get_active_voice_count(self) -> int:
        """アクティブボイス数を取得"""
        with self._lock:
            return len(self.active_voices)
    
    def get_stats(self) -> Dict[str, float]:
        """統計情報を取得"""
        return {
            'active_voices': self.get_active_voice_count(),
            'sample_rate': self.sample_rate,
            'latency_ms': self.get_latency_ms(),
            'running': self.running
        } 