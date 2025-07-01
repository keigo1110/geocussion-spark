#!/usr/bin/env python3
"""
パーカッション楽器

打楽器全般の音響特性を提供します。
"""

from typing import Dict, Any
from . import BaseInstrument
from ..mapping import InstrumentType
from ... import get_logger

logger = get_logger(__name__)


class PercussionInstrument(BaseInstrument):
    """パーカッション楽器クラス"""
    
    def __init__(self):
        super().__init__(InstrumentType.PERCUSSION)
        logger.debug("PercussionInstrument initialized")
    
    def _initialize_default_settings(self) -> Dict[str, Any]:
        """パーカッションのデフォルト設定"""
        return {
            'waveform': 'noise',   # ノイズベースの音色
            'attack': 0.001,       # 瞬間的なアタック
            'decay': 0.15,         # 速い減衰
            'sustain': 0.1,        # 低い持続レベル
            'release': 0.2,        # 短いリリース
            'brightness': 0.8,     # 明るい音色
            'resonance': 0.3,      # 控えめな共鳴
            'noise_amount': 0.7,   # ノイズの量
            'pitch_bend': 0.1,     # 瞬間的なピッチベンド
            'harmonics': [1.0, 0.4, 0.2, 0.1, 0.05],  # 非調和倍音
            'filter_type': 'bandpass'  # バンドパスフィルター
        }
    
    def _apply_instrument_specific_params(self, voice_params: Dict[str, Any], **params):
        """パーカッション固有のパラメータ処理"""
        # 金属的 vs 木質的な音色の調整
        surface_type = params.get('surface_type', 'medium')  # hard, medium, soft
        
        if surface_type == 'hard':  # 金属的
            voice_params['brightness'] *= 1.4
            voice_params['resonance'] *= 1.5
            voice_params['decay'] *= 1.3
            voice_params['noise_amount'] *= 0.8
        elif surface_type == 'soft':  # 木質的/皮革的
            voice_params['brightness'] *= 0.7
            voice_params['resonance'] *= 0.6
            voice_params['decay'] *= 0.7
            voice_params['noise_amount'] *= 1.2
        
        # 振幅による音色変化（打撃の強さ）
        amplitude = voice_params['amplitude']
        if amplitude > 0.8:  # 強打
            voice_params['pitch_bend'] *= 1.5  # より大きなピッチベンド
            voice_params['noise_amount'] *= 1.2
            voice_params['brightness'] *= 1.1
        elif amplitude < 0.3:  # 弱打
            voice_params['pitch_bend'] *= 0.5
            voice_params['noise_amount'] *= 0.8
            voice_params['brightness'] *= 0.9
        
        # 周波数帯域による特性調整
        frequency = voice_params['frequency']
        if frequency > 1000:  # 高音域（シンバル系）
            voice_params['decay'] *= 2.0
            voice_params['brightness'] *= 1.2
            voice_params['noise_amount'] *= 1.3
        elif frequency < 200:  # 低音域（ドラム系）
            voice_params['decay'] *= 0.8
            voice_params['resonance'] *= 1.3
            voice_params['pitch_bend'] *= 2.0
        
        # 衝突の種類による調整
        collision_type = params.get('collision_type', 'normal')
        if collision_type == 'scrape':  # 擦過音
            voice_params['attack'] *= 3.0
            voice_params['noise_amount'] *= 1.5
            voice_params['decay'] *= 2.0
        elif collision_type == 'roll':  # ロール奏法
            voice_params['attack'] *= 0.5
            voice_params['sustain'] *= 3.0
            voice_params['noise_amount'] *= 0.7
        
        logger.debug(f"Percussion voice created: freq={frequency:.1f}Hz, "
                    f"surface={surface_type}, "
                    f"noise_amount={voice_params['noise_amount']:.2f}")
    
    def get_noise_characteristics(self, frequency: float) -> Dict[str, float]:
        """周波数に応じたノイズ特性"""
        if frequency > 2000:  # 超高音域
            return {
                'noise_type': 'white',
                'filter_freq': frequency * 1.5,
                'filter_q': 2.0
            }
        elif frequency > 500:  # 高音域
            return {
                'noise_type': 'pink',
                'filter_freq': frequency * 1.2,
                'filter_q': 1.5
            }
        else:  # 中低音域
            return {
                'noise_type': 'brown',
                'filter_freq': frequency * 0.8,
                'filter_q': 1.0
            } 