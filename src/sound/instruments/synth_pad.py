#!/usr/bin/env python3
"""
シンセサイザーパッド楽器

電子的な持続音の音響特性を提供します。
"""

from typing import Dict, Any
from . import BaseInstrument
from ..mapping import InstrumentType
from ... import get_logger

logger = get_logger(__name__)


class SynthPadInstrument(BaseInstrument):
    """シンセサイザーパッド楽器クラス"""
    
    def __init__(self):
        super().__init__(InstrumentType.SYNTH_PAD)
        logger.debug("SynthPadInstrument initialized")
    
    def _initialize_default_settings(self) -> Dict[str, Any]:
        """シンセパッドのデフォルト設定"""
        return {
            'waveform': 'square',  # 方形波でリッチな倍音
            'attack': 0.5,         # ゆっくりとしたアタック
            'decay': 0.2,          # 短い減衰
            'sustain': 0.8,        # 高い持続レベル
            'release': 2.0,        # 長いリリース
            'brightness': 0.4,     # やわらかい音色
            'resonance': 0.7,      # 電子的な共鳴
            'filter_cutoff': 0.6,  # ローパスフィルター
            'chorus_depth': 0.3,   # コーラス効果
            'reverb': 0.4,         # リバーブ効果
            'harmonics': [1.0, 0.3, 0.8, 0.2, 0.5, 0.1, 0.3]  # 方形波的な倍音
        }
    
    def _apply_instrument_specific_params(self, voice_params: Dict[str, Any], **params):
        """シンセパッド固有のパラメータ処理"""
        # 低音域では温かみのある音色に
        frequency = voice_params['frequency']
        if frequency < 300:  # 低音域
            voice_params['filter_cutoff'] *= 0.8
            voice_params['brightness'] *= 1.1
            voice_params['attack'] *= 1.2  # よりゆっくりなアタック
        elif frequency > 1000:  # 高音域
            voice_params['filter_cutoff'] *= 1.2
            voice_params['brightness'] *= 0.9
            voice_params['attack'] *= 0.8  # 少し速めのアタック
        
        # 振幅に応じた表現力の変化
        amplitude = voice_params['amplitude']
        if amplitude > 0.8:  # 強い表現
            voice_params['chorus_depth'] *= 1.3
            voice_params['resonance'] *= 1.1
        elif amplitude < 0.2:  # 弱い表現
            voice_params['chorus_depth'] *= 0.7
            voice_params['attack'] *= 1.5  # よりソフトなアタック
        
        # 持続時間による効果調整
        duration = voice_params.get('duration', 1.0)
        if duration > 3.0:  # 長い音符
            voice_params['reverb'] *= 1.2
            voice_params['chorus_depth'] *= 1.1
        
        # 空間的な位置による音色変化
        pan = voice_params.get('pan', 0.0)
        if abs(pan) > 0.5:  # 端に位置する場合
            voice_params['reverb'] *= 1.1
        
        logger.debug(f"SynthPad voice created: freq={frequency:.1f}Hz, "
                    f"attack={voice_params['attack']:.2f}s, "
                    f"filter_cutoff={voice_params['filter_cutoff']:.2f}")
    
    def get_modulation_params(self, velocity: float) -> Dict[str, float]:
        """速度に応じたモジュレーションパラメータ"""
        return {
            'lfo_rate': 2.0 + velocity * 3.0,      # 0.5-5.5 Hz
            'lfo_depth': 0.1 + velocity * 0.3,     # 0.1-0.4
            'vibrato_rate': 4.5 + velocity * 1.5,  # 4.5-6.0 Hz
            'vibrato_depth': 0.05 + velocity * 0.1  # 0.05-0.15
        } 