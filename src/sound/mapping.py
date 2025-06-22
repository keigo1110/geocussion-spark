#!/usr/bin/env python3
"""
音響生成 - パラメータマッピング

衝突イベントから音響パラメータ（音高、音量、楽器、空間配置）への
マッピングを行う機能を提供します。MIDI互換の標準化されたフォーマットを使用します。
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

# 他フェーズとの連携
from ..collision.events import CollisionEvent, CollisionIntensity, EventType
from ..utils import config_manager
from ..utils.config import settings


@dataclass
class AudioParameters:
    """音響パラメータの統合データ"""
    # 基本パラメータ
    pitch: float                  # 音高（MIDI note number）
    velocity: float               # 音量（0.0-1.0）
    duration: float               # 持続時間（秒）
    
    # 楽器・音色
    instrument: str    # 楽器タイプ (文字列に変更)
    timbre: float                 # 音色変化（0.0-1.0）
    brightness: float             # 明度（0.0-1.0）
    
    # 空間配置
    pan: float                    # パンニング（-1.0 左 ～ 1.0 右）
    distance: float               # 距離感（0.0 近 ～ 1.0 遠）
    reverb: float                 # リバーブ量（0.0-1.0）
    
    # エンベロープ
    attack: float                 # アタック時間（秒）
    decay: float                  # ディケイ時間（秒）
    sustain: float                # サステイン レベル（0.0-1.0）
    release: float                # リリース時間（秒）
    
    # メタデータ
    event_id: str                 # イベントID
    hand_id: str                  # 手のID
    timestamp: float              # タイムスタンプ
    
    @property
    def midi_note(self) -> int:
        """MIDI note numberとして取得"""
        return max(0, min(127, int(round(self.pitch))))
    
    @property
    def frequency(self) -> float:
        """周波数（Hz）として取得"""
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))
    
    @property
    def velocity_127(self) -> int:
        """MIDI velocity (0-127)として取得"""
        return max(0, min(127, int(self.velocity * 127)))
    
    @property
    def pan_degrees(self) -> float:
        """パンニング角度（度）として取得"""
        return self.pan * 90.0  # -90度～+90度


class AudioMapper:
    """音響パラメータマッピングクラス"""
    
    def __init__(
        self,
        base_octave: int = 4,
        pitch_range: Tuple[int, int] = (48, 84),
        volume_curve: str = "logarithmic",
        spatial_range: float = 2.0,
        enable_adaptive_mapping: bool = True
    ):
        """
        初期化
        """
        self.audio_config = settings.get('audio', {})
        
        # configから直接読み込む
        self.scale_patterns = self._load_scales_from_config()
        self.instrument_envelopes = self._load_instrument_envelopes_from_config()
        
        self.scale = self.audio_config.get('default_scale', 'PENTATONIC').upper()
        if self.scale not in self.scale_patterns:
            self.scale = next(iter(self.scale_patterns)) # フォールバック

        self.default_instrument = self.audio_config.get('default_instrument', 'MARIMBA').upper()
        if self.default_instrument not in self.instrument_envelopes:
            self.default_instrument = next(iter(self.instrument_envelopes)) # フォールバック

        self.base_octave = base_octave
        self.pitch_range = pitch_range
        self.volume_curve = volume_curve
        self.spatial_range = spatial_range
        self.enable_adaptive_mapping = enable_adaptive_mapping
        
        self.adaptive_params = {
            'y_range': [0.0, 1.0],
            'velocity_range': [0.0, 1.0],
            'x_range': [-1.0, 1.0],
            'samples_count': 0,
            'update_interval': 100
        }
        
        self.stats = {
            'total_mappings': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_mapping_time_ms': 0.0,
            'instrument_usage': {inst: 0 for inst in self.instrument_envelopes.keys()},
            'note_distribution': np.zeros(128)
        }
    
    def map_collision_event(self, event: CollisionEvent) -> AudioParameters:
        """
        衝突イベントを音響パラメータにマッピング
        
        Args:
            event: 衝突イベント
            
        Returns:
            音響パラメータ
        """
        start_time = time.perf_counter()
        
        # 基本パラメータマッピング
        pitch = self._map_pitch(event)
        velocity = self._map_velocity(event)
        duration = self._map_duration(event)
        
        # 楽器選択
        instrument = self._select_instrument(event)
        
        # 音色・明度
        timbre = self._map_timbre(event)
        brightness = self._map_brightness(event)
        
        # 空間配置
        pan = self._map_pan(event)
        distance = self._map_distance(event)
        reverb = self._map_reverb(event)
        
        # エンベロープ
        attack, decay, sustain, release = self._map_envelope(event, instrument)
        
        # 適応的パラメータ更新
        if self.enable_adaptive_mapping:
            self._update_adaptive_params(event)
        
        audio_params = AudioParameters(
            pitch=pitch,
            velocity=velocity,
            duration=duration,
            instrument=instrument,
            timbre=timbre,
            brightness=brightness,
            pan=pan,
            distance=distance,
            reverb=reverb,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            event_id=event.event_id,
            hand_id=event.hand_id,
            timestamp=event.timestamp
        )
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, audio_params)
        
        return audio_params

    def _map_pitch(self, event: CollisionEvent) -> float:
        """Y座標を音高にマッピング"""
        # Y座標を正規化（適応的範囲使用）
        y_min, y_max = self.adaptive_params['y_range']
        y_pos = event.contact_position[1]
        
        if y_max > y_min:
            normalized_y = (y_pos - y_min) / (y_max - y_min)
        else:
            normalized_y = 0.5
        
        normalized_y = max(0.0, min(1.0, normalized_y))
        
        # 音階に応じて量子化
        scale_key = self.scale
        scale_notes = self.scale_patterns.get(scale_key, [0, 2, 4, 7, 9])
        min_note, max_note = self.pitch_range
        
        # スケール内の音程を計算
        octave_range = (max_note - min_note) // 12
        note_in_scale = int(normalized_y * len(scale_notes) * octave_range)
        
        octave = note_in_scale // len(scale_notes)
        scale_degree = note_in_scale % len(scale_notes)
        
        # MIDI note number計算
        base_note = min_note + octave * 12
        scale_offset = scale_notes[scale_degree]
        final_note = base_note + scale_offset
        
        return max(min_note, min(max_note, final_note))
    
    def _map_velocity(self, event: CollisionEvent) -> float:
        """衝突イベントの速度と強度を音量にマッピング"""
        # 基本速度
        base_velocity = np.clip(event.velocity * 2.0, 0.1, 1.0) # 0.5m/sで最大に
        
        # 侵入深度による加算
        depth_factor = np.clip(event.penetration_depth * 10.0, 0.0, 0.2)
        
        # 強度による乗算
        intensity_factor = (event.intensity.value / CollisionIntensity.MAXIMUM.value)
        
        final_velocity = base_velocity * (1.0 + depth_factor) * (0.5 + intensity_factor * 0.5)

        # 対数カーブ
        if self.volume_curve == "logarithmic":
            # ゼロや負の値を回避
            safe_velocity = max(1e-4, final_velocity)
            # 1.0以上にならないようにクリップし、floatに変換
            return float(np.clip(np.log1p(safe_velocity * 9) / np.log1p(9), 0.0, 1.0))

        return float(np.clip(final_velocity, 0.0, 1.0))
    
    def _map_duration(self, event: CollisionEvent) -> float:
        """持続時間をマッピング"""
        # 継続時間があればそれを優先
        if event.event_type == EventType.COLLISION_CONTINUE and event.duration_ms > 0:
            return event.duration_ms / 1000.0
        
        # 強度と速度に基づいて基本時間を設定
        base_duration = 0.1 + (event.intensity.value / CollisionIntensity.MAXIMUM.value) * 1.0
        velocity_factor = 1.0 + np.clip(event.velocity, 0, 1.0)
        
        duration = base_duration * velocity_factor
        return max(0.1, min(5.0, duration))
    
    def _select_instrument(self, event: CollisionEvent) -> str:
        """衝突イベントに基づいて楽器を選択"""
        # Y座標（高さ）に基づいて楽器を切り替える
        y_pos = event.contact_position[1]
        
        if y_pos > 0.8:
            return "CRYSTAL"
        elif y_pos > 0.5:
            return "BELL"
        elif y_pos > 0.2:
            return "MARIMBA"
        else:
            # 低い位置はパーカッシブな音
            return self.audio_config.get('percussion_instrument', 'DRUM').upper()
    
    def _map_timbre(self, event: CollisionEvent) -> float:
        """音色をマッピング"""
        # 表面法線の傾きから音色を計算
        normal = event.surface_normal
        tilt = np.linalg.norm(normal[:2])  # XY平面での傾き
        
        # 侵入深度による音色変化
        depth_factor = min(event.penetration_depth * 50, 1.0)
        
        # 組み合わせ
        timbre = tilt * 0.6 + depth_factor * 0.4
        return max(0.0, min(1.0, timbre))
    
    def _map_brightness(self, event: CollisionEvent) -> float:
        """明度をマッピング"""
        # 速度が速いほど明るい音
        velocity_factor = min(event.velocity * 10, 1.0)
        
        # 強度が高いほど明るい音
        intensity_factor = event.intensity.value / CollisionIntensity.MAXIMUM.value
        
        brightness = velocity_factor * 0.7 + intensity_factor * 0.3
        return max(0.1, min(1.0, brightness))
    
    def _map_pan(self, event: CollisionEvent) -> float:
        """X座標をステレオパンニングにマッピング"""
        x_min, x_max = self.adaptive_params['x_range']
        x_pos = event.contact_position[0]
        
        if x_max > x_min:
            normalized_x = (x_pos - x_min) / (x_max - x_min)
        else:
            normalized_x = 0.5
        
        # -1.0（左）～ 1.0（右）にマッピング
        pan = (normalized_x - 0.5) * 2.0
        return max(-1.0, min(1.0, pan))
    
    def _map_distance(self, event: CollisionEvent) -> float:
        """距離感をマッピング"""
        # Z座標を距離として使用
        z_pos = event.contact_position[2]
        
        # 正規化（近い=0.0、遠い=1.0）
        distance = max(0.0, min(self.spatial_range, abs(z_pos))) / self.spatial_range
        return distance
    
    def _map_reverb(self, event: CollisionEvent) -> float:
        """リバーブ量をマッピング"""
        # 距離が遠いほどリバーブを多く
        distance_factor = self._map_distance(event)
        
        # 接触面積が大きいほどリバーブを多く
        area_factor = min(event.contact_area * 500, 1.0)
        
        reverb = distance_factor * 0.6 + area_factor * 0.4
        return max(0.0, min(0.8, reverb))  # 最大80%
    
    def _map_envelope(self, event: CollisionEvent, instrument: str) -> Tuple[float, float, float, float]:
        """イベントと楽器に基づいてエンベロープをマッピング"""
        envelope = self.instrument_envelopes.get(instrument.upper())
        if envelope:
            # 衝突速度に応じてアタックを少し変える
            attack_mod = 1.0 - np.clip(event.velocity * 0.5, 0, 0.5)
            return (
                envelope['attack'] * attack_mod,
                envelope['decay'],
                envelope['sustain'],
                envelope['release']
            )
        # デフォルトエンベロープ
        return (0.01, 0.3, 0.5, 0.5)

    def _load_scales_from_config(self) -> Dict[str, List[int]]:
        """config.yamlから音階定義を読み込む"""
        scales_config = self.audio_config.get('scales', {})
        return {name.upper(): pattern for name, pattern in scales_config.items()}

    def _load_instrument_envelopes_from_config(self) -> Dict[str, Dict[str, float]]:
        """config.yamlから楽器のエンベロープ定義を読み込む"""
        instruments_config = self.audio_config.get('instruments', {})
        envelopes = {}
        for name, details in instruments_config.items():
            if 'envelope' in details:
                envelopes[name.upper()] = details['envelope']
        return envelopes

    def _update_adaptive_params(self, event: CollisionEvent):
        """適応的パラメータ（Y座標、速度、X座標の範囲）を更新"""
        self.adaptive_params['samples_count'] += 1
        
        if self.adaptive_params['samples_count'] % self.adaptive_params['update_interval'] == 0:
            # 範囲をわずかにリセットして外れ値の影響を徐々に減らす
            self.adaptive_params['y_range'][0] *= 1.05
            self.adaptive_params['y_range'][1] *= 0.95
            self.adaptive_params['velocity_range'][1] *= 0.95

    def _update_stats(self, elapsed_ms: float, audio_params: AudioParameters):
        """パフォーマンス統計を更新"""
        self.stats['total_mappings'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_mappings']
        self.stats['last_mapping_time_ms'] = elapsed_ms
        
        # 楽器使用状況
        inst_key = audio_params.instrument.upper()
        self.stats['instrument_usage'][inst_key] += 1
        
        # 音高分布
        self.stats['note_distribution'][audio_params.midi_note] += 1
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        stats['note_distribution'] = self.stats['note_distribution'].tolist()
        stats['adaptive_params'] = self.adaptive_params.copy()
        return stats
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_mappings': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_mapping_time_ms': 0.0,
            'instrument_usage': {inst: 0 for inst in self.instrument_envelopes.keys()},
            'note_distribution': np.zeros(128)
        }
        
        self.adaptive_params['samples_count'] = 0
    
    def set_scale(self, scale_name: str):
        """使用する音階を設定"""
        scale_name_upper = scale_name.upper()
        if scale_name_upper in self.scale_patterns:
            self.scale = scale_name_upper
        else:
            raise ValueError(f"Scale '{scale_name}' not found in config.")
    
    def set_pitch_range(self, min_note: int, max_note: int):
        """音域を変更"""
        self.pitch_range = (min_note, max_note)
    
    def set_spatial_range(self, range_meters: float):
        """空間範囲を変更"""
        self.spatial_range = range_meters


# 便利関数

def map_collision_to_audio(
    event: CollisionEvent,
    scale: Optional[str] = None,
    instrument: Optional[str] = None
) -> AudioParameters:
    """
    衝突イベントを音響パラメータにマッピングする簡易関数
    
    Args:
        event: 衝突イベント
        scale: 音階（指定されない場合はコンフィグのデフォルト値を使用）
        instrument: 楽器（指定されない場合はコンフィグのデフォルト値を使用）
        
    Returns:
        音響パラメータ
    """
    mapper = AudioMapper()
    if scale:
        mapper.set_scale(scale)
    if instrument:
        mapper.default_instrument = instrument
    
    return mapper.map_collision_event(event)


def batch_map_collisions(
    events: List[CollisionEvent],
    mapper: Optional[AudioMapper] = None
) -> List[AudioParameters]:
    """
    複数の衝突イベントを一括でマッピング
    
    Args:
        events: 衝突イベントのリスト
        mapper: 使用するAudioMapperインスタンス（指定されない場合は新規作成）
        
    Returns:
        音響パラメータのリスト
    """
    if mapper is None:
        mapper = AudioMapper()
    
    return [mapper.map_collision_event(event) for event in events]


def create_scale_mapper(
    scale_name: str,
    root_note: int = 60  # C4
) -> AudioMapper:
    """特定の音階に設定されたマッパーを生成"""
    mapper = AudioMapper()
    mapper.set_scale(scale_name)
    # 必要に応じて他のパラメータも設定
    return mapper
