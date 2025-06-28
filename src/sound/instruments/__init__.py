#!/usr/bin/env python3
"""
楽器Strategy パターン

楽器ごとの音響生成ロジックを分離し、拡張可能な設計を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import numpy as np

# 逆順インポート回避のため
from ..mapping import InstrumentType
from ... import get_logger

logger = get_logger(__name__)


class IInstrument(ABC):
    """楽器インターフェース"""
    
    @abstractmethod
    def get_instrument_type(self) -> InstrumentType:
        """楽器タイプを取得"""
        pass
    
    @abstractmethod
    def create_voice_parameters(self, frequency: float, amplitude: float, **params) -> Dict[str, Any]:
        """ボイス生成パラメータを作成"""
        pass
    
    @abstractmethod
    def get_default_settings(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        pass


class BaseInstrument(IInstrument):
    """楽器ベースクラス"""
    
    def __init__(self, instrument_type: InstrumentType):
        self.instrument_type = instrument_type
        self.default_settings = self._initialize_default_settings()
    
    def get_instrument_type(self) -> InstrumentType:
        return self.instrument_type
    
    def get_default_settings(self) -> Dict[str, Any]:
        return self.default_settings.copy()
    
    @abstractmethod
    def _initialize_default_settings(self) -> Dict[str, Any]:
        """デフォルト設定の初期化（サブクラスで実装）"""
        pass
    
    def create_voice_parameters(self, frequency: float, amplitude: float, **params) -> Dict[str, Any]:
        """ボイス生成パラメータを作成"""
        # ベース設定をコピー
        voice_params = self.get_default_settings()
        
        # 基本パラメータを設定
        voice_params.update({
            'frequency': frequency,
            'amplitude': amplitude,
            'waveform': params.get('waveform', voice_params.get('waveform', 'sine')),
            'pan': params.get('pan', 0.0),
            'duration': params.get('duration', 1.0)
        })
        
        # 楽器固有の処理
        self._apply_instrument_specific_params(voice_params, **params)
        
        return voice_params
    
    def _apply_instrument_specific_params(self, voice_params: Dict[str, Any], **params):
        """楽器固有のパラメータ適用（サブクラスでオーバーライド）"""
        pass


# ファクトリ関数
def create_instrument(instrument_type: InstrumentType) -> IInstrument:
    """楽器インスタンスを作成"""
    from .marimba import MarimbaInstrument
    from .synth_pad import SynthPadInstrument
    from .percussion import PercussionInstrument
    
    instrument_map = {
        InstrumentType.MARIMBA: MarimbaInstrument,
        InstrumentType.SYNTH_PAD: SynthPadInstrument,
        InstrumentType.PERCUSSION: PercussionInstrument,
    }
    
    instrument_class = instrument_map.get(instrument_type)
    if instrument_class is None:
        logger.warning(f"Unknown instrument type: {instrument_type}, falling back to MARIMBA")
        instrument_class = MarimbaInstrument
    
    return instrument_class()


__all__ = [
    'IInstrument',
    'BaseInstrument', 
    'create_instrument'
] 