#!/usr/bin/env python3
"""
Pyo音響バックエンド

pyoライブラリのラッパー実装。
リアルタイム音響処理を提供します。
"""

import time
import threading
from typing import Dict, Optional, Set, Any
from . import IAudioBackend, BackendType
from ... import get_logger

logger = get_logger(__name__)

try:
    import pyo
    HAS_PYO = True
except ImportError:
    HAS_PYO = False
    pyo = None
    logger.warning("pyo not available. PyoBackend will not function.")


class PyoAudioBackend(IAudioBackend):
    """pyo音響処理バックエンド"""
    
    def __init__(self):
        if not HAS_PYO:
            raise RuntimeError("pyo library is not available")
        
        self.server: Optional[pyo.Server] = None
        self.initialized = False
        self.running = False
        
        # ボイス管理
        self.active_voices: Dict[str, Any] = {}
        self.voice_counter = 0
        self._lock = threading.Lock()
        
        logger.debug("PyoAudioBackend initialized")
    
    def initialize(self, sample_rate: int, channels: int, buffer_size: int, **kwargs) -> bool:
        """バックエンドを初期化"""
        if not HAS_PYO:
            logger.error("pyo is not available")
            return False
        
        try:
            audio_driver = kwargs.get('audio_driver', 'portaudio')
            duplex = kwargs.get('duplex', False)
            
            self.server = pyo.Server(
                sr=sample_rate,
                nchnls=channels,
                buffersize=buffer_size,
                duplex=int(duplex),
                audio=audio_driver
            )
            
            self.server.boot()
            self.initialized = True
            logger.info(f"PyoBackend initialized: {sample_rate}Hz, {channels}ch, {buffer_size} samples")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize PyoBackend: {e}")
            return False
    
    def start(self) -> bool:
        """音響処理を開始"""
        if not self.initialized or not self.server:
            logger.error("Backend not initialized")
            return False
        
        try:
            self.server.start()
            self.running = True
            logger.info("PyoBackend started")
            return True
        except Exception as e:
            logger.exception(f"Failed to start PyoBackend: {e}")
            return False
    
    def stop(self) -> bool:
        """音響処理を停止"""
        try:
            if self.server:
                # すべてのボイスを停止
                with self._lock:
                    for voice_id, voice_obj in list(self.active_voices.items()):
                        try:
                            if hasattr(voice_obj, 'stop'):
                                voice_obj.stop()
                        except Exception as e:
                            logger.error(f"Error stopping voice {voice_id}: {e}")
                    self.active_voices.clear()
                
                self.server.stop()
            
            self.running = False
            logger.info("PyoBackend stopped")
            return True
            
        except Exception as e:
            logger.exception(f"Error stopping PyoBackend: {e}")
            return False
    
    def shutdown(self) -> bool:
        """バックエンドをシャットダウン"""
        try:
            self.stop()
            
            if self.server:
                self.server.shutdown()
                self.server = None
            
            self.initialized = False
            logger.info("PyoBackend shutdown")
            return True
            
        except Exception as e:
            logger.exception(f"Error shutting down PyoBackend: {e}")
            return False
    
    def create_voice(self, frequency: float, amplitude: float, **params) -> Optional[str]:
        """ボイスを作成し、ボイスIDを返す"""
        if not self.running or not HAS_PYO:
            return None
        
        try:
            with self._lock:
                self.voice_counter += 1
                voice_id = f"pyo_voice_{self.voice_counter:06d}"
                
                # シンプルなサイン波オシレーター
                waveform = params.get('waveform', 'sine')
                if waveform == 'sine':
                    voice_obj = pyo.Sine(freq=frequency, mul=amplitude)
                elif waveform == 'square':
                    voice_obj = pyo.Square(freq=frequency, mul=amplitude)
                else:
                    voice_obj = pyo.Sine(freq=frequency, mul=amplitude)
                
                # パンニング
                pan = params.get('pan', 0.0)
                if pan != 0.0:
                    voice_obj = pyo.Pan(voice_obj, outs=2, pan=pan)
                
                voice_obj.out()
                self.active_voices[voice_id] = voice_obj
            
            logger.debug(f"Created pyo voice: {voice_id} (freq={frequency:.1f}Hz, amp={amplitude:.3f})")
            return voice_id
            
        except Exception as e:
            logger.exception(f"Error creating voice: {e}")
            return None
    
    def stop_voice(self, voice_id: str) -> bool:
        """指定されたボイスを停止"""
        with self._lock:
            if voice_id in self.active_voices:
                try:
                    voice_obj = self.active_voices[voice_id]
                    if hasattr(voice_obj, 'stop'):
                        voice_obj.stop()
                    del self.active_voices[voice_id]
                    logger.debug(f"Stopped pyo voice: {voice_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error stopping voice {voice_id}: {e}")
        return False
    
    def is_voice_active(self, voice_id: str) -> bool:
        """ボイスがアクティブかどうか"""
        with self._lock:
            return voice_id in self.active_voices
    
    def get_latency_ms(self) -> float:
        """レイテンシーをミリ秒で取得"""
        if self.server:
            return (self.server.getBufferSize() / self.server.getSamplingRate()) * 1000
        return 0.0
    
    def get_backend_type(self) -> BackendType:
        """バックエンドタイプを取得"""
        return BackendType.PYO
    
    def get_active_voice_count(self) -> int:
        """アクティブボイス数を取得"""
        with self._lock:
            return len(self.active_voices)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        stats = {
            'active_voices': self.get_active_voice_count(),
            'latency_ms': self.get_latency_ms(),
            'running': self.running
        }
        
        if self.server:
            stats.update({
                'sample_rate': self.server.getSamplingRate(),
                'buffer_size': self.server.getBufferSize(),
                'channels': self.server.getNchnls()
            })
        
        return stats 