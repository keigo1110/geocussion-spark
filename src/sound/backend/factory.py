#!/usr/bin/env python3
"""
音響バックエンドファクトリ

利用可能なライブラリに応じて適切な音響バックエンドを作成します。
"""

from typing import Optional
from . import IAudioBackend, BackendType
from .null_backend import NullAudioBackend
from ... import get_logger

logger = get_logger(__name__)

# pyoバックエンドは遅延インポート
_pyo_backend_available = None


def _check_pyo_available() -> bool:
    """pyoの利用可能性をチェック（キャッシュ）"""
    global _pyo_backend_available
    if _pyo_backend_available is None:
        try:
            from .pyo_backend import PyoAudioBackend
            _pyo_backend_available = True
        except (ImportError, RuntimeError):
            _pyo_backend_available = False
    return _pyo_backend_available


def create_backend(preferred_type: Optional[BackendType] = None) -> IAudioBackend:
    """
    音響バックエンドを作成
    
    Args:
        preferred_type: 優先するバックエンドタイプ。Noneの場合は自動選択。
    
    Returns:
        作成されたバックエンドインスタンス
    """
    
    # 明示的にNullが指定された場合
    if preferred_type == BackendType.NULL:
        logger.info("Creating NullAudioBackend (explicitly requested)")
        return NullAudioBackend()
    
    # Pyoが明示的に指定された場合
    if preferred_type == BackendType.PYO:
        if _check_pyo_available():
            from .pyo_backend import PyoAudioBackend
            logger.info("Creating PyoAudioBackend (explicitly requested)")
            return PyoAudioBackend()
        else:
            logger.warning("Pyo backend requested but not available, falling back to Null")
            return NullAudioBackend()
    
    # 自動選択: Pyoが利用可能ならPyo、そうでなければNull
    if _check_pyo_available():
        from .pyo_backend import PyoAudioBackend
        logger.info("Creating PyoAudioBackend (auto-selected)")
        return PyoAudioBackend()
    else:
        logger.info("Creating NullAudioBackend (pyo not available)")
        return NullAudioBackend()


def get_available_backends() -> list[BackendType]:
    """利用可能なバックエンドタイプのリストを取得"""
    available = [BackendType.NULL]  # Nullは常に利用可能
    
    if _check_pyo_available():
        available.append(BackendType.PYO)
    
    return available


def is_backend_available(backend_type: BackendType) -> bool:
    """指定されたバックエンドが利用可能かチェック"""
    if backend_type == BackendType.NULL:
        return True
    elif backend_type == BackendType.PYO:
        return _check_pyo_available()
    else:
        return False 