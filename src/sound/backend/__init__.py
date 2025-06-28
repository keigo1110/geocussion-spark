#!/usr/bin/env python3
"""
音響バックエンド抽象化層

異なる音響ライブラリ（pyo, null, mock）を統一インターフェースで利用できます。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import numpy as np


class BackendType(Enum):
    """バックエンドタイプの列挙"""
    PYO = "pyo"
    NULL = "null"
    MOCK = "mock"


class IAudioBackend(ABC):
    """音響バックエンドインターフェース"""
    
    @abstractmethod
    def initialize(self, sample_rate: int, channels: int, buffer_size: int, **kwargs) -> bool:
        """バックエンドを初期化"""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """音響処理を開始"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """音響処理を停止"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """バックエンドをシャットダウン"""
        pass
    
    @abstractmethod
    def create_voice(self, frequency: float, amplitude: float, **params) -> Optional[str]:
        """ボイスを作成し、ボイスIDを返す"""
        pass
    
    @abstractmethod
    def stop_voice(self, voice_id: str) -> bool:
        """指定されたボイスを停止"""
        pass
    
    @abstractmethod
    def is_voice_active(self, voice_id: str) -> bool:
        """ボイスがアクティブかどうか"""
        pass
    
    @abstractmethod
    def get_latency_ms(self) -> float:
        """レイテンシーをミリ秒で取得"""
        pass
    
    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """バックエンドタイプを取得"""
        pass


# ファクトリ関数のインポート
from .factory import create_backend, get_available_backends, is_backend_available

__all__ = [
    'IAudioBackend',
    'BackendType',
    'create_backend',
    'get_available_backends',
    'is_backend_available'
] 