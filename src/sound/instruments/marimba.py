#!/usr/bin/env python3
"""
マリンバ楽器

木質打楽器の音響特性を模倣します。
"""

from typing import Dict, Any
from . import BaseInstrument
from ..mapping import InstrumentType
from ... import get_logger

logger = get_logger(__name__)


class MarimbaInstrument(BaseInstrument):
    """マリンバ楽器クラス"""
    
    def __init__(self):
        super().__init__(InstrumentType.MARIMBA)
        logger.debug("MarimbaInstrument initialized")
    
    def _initialize_default_settings(self) -> Dict[str, Any]:
        """マリンバのデフォルト設定"""
        return {
            'waveform': 'sine',
            'attack': 0.01,      # 速いアタック（打撃楽器）
            'decay': 0.3,        # 比較的速い減衰
            'sustain': 0.4,      # 中程度の持続
            'release': 0.8,      # ゆるやかなリリース
            'brightness': 0.6,   # 明るめの音色
            'resonance': 0.5,    # 木の共鳴特性
            'harmonics': [1.0, 0.6, 0.3, 0.15, 0.08],  # 倍音構成
            'frequency_multipliers': [1.0, 3.2, 6.8, 10.5]  # マリンバ特有の倍音比
        }
    
    def _apply_instrument_specific_params(self, voice_params: Dict[str, Any], **params):
        """マリンバ固有のパラメータ処理"""
        # 高い音ほど減衰が速い（実楽器の物理特性）
        frequency = voice_params['frequency']
        if frequency > 800:  # 高音域
            voice_params['decay'] *= 0.7
            voice_params['release'] *= 0.6
        elif frequency < 200:  # 低音域
            voice_params['decay'] *= 1.3
            voice_params['release'] *= 1.4
        
        # 振幅に応じた音色変化
        amplitude = voice_params['amplitude']
        if amplitude > 0.7:  # 強打
            voice_params['brightness'] *= 1.2
            voice_params['attack'] *= 0.8  # より速いアタック
        elif amplitude < 0.3:  # 弱打
            voice_params['brightness'] *= 0.8
            voice_params['attack'] *= 1.2  # よりソフトなアタック
        
        # 共鳴効果の調整
        resonance_factor = params.get('surface_hardness', 0.5)
        voice_params['resonance'] *= (0.5 + resonance_factor * 0.5)
        
        logger.debug(f"Marimba voice created: freq={frequency:.1f}Hz, "
                    f"decay={voice_params['decay']:.2f}s, "
                    f"brightness={voice_params['brightness']:.2f}")
    
    def get_frequency_adjustment(self, base_frequency: float) -> float:
        """マリンバ特有の周波数調整"""
        # マリンバは少し低めに調律されることが多い
        return base_frequency * 0.98 