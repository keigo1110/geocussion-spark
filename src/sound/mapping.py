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


class InstrumentType(Enum):
    """楽器タイプの列挙"""
    MARIMBA = "marimba"           # マリンバ（木琴）
    SYNTH_PAD = "synth_pad"       # シンセパッド
    BELL = "bell"                 # ベル・チャイム
    CRYSTAL = "crystal"           # クリスタル
    DRUM = "drum"                 # ドラム・パーカッション
    WATER_DROP = "water_drop"     # 水滴音
    WIND = "wind"                 # 風音
    STRING = "string"             # 弦楽器


class ScaleType(Enum):
    """音階タイプの列挙"""
    CHROMATIC = "chromatic"       # 半音階
    MAJOR = "major"               # 長調
    MINOR = "minor"               # 短調  
    PENTATONIC = "pentatonic"     # ペンタトニック
    BLUES = "blues"               # ブルース
    WHOLE_TONE = "whole_tone"     # 全音階
    JAPANESE = "japanese"         # 日本音階


@dataclass
class AudioParameters:
    """音響パラメータの統合データ"""
    # 基本パラメータ
    pitch: float                  # 音高（MIDI note number）
    velocity: float               # 音量（0.0-1.0）
    duration: float               # 持続時間（秒）
    
    # 楽器・音色
    instrument: InstrumentType    # 楽器タイプ
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
    
    # ゲイン（振幅スケール）
    gain: float = 1.0             # 0.0-1.0, 音色別に基準音量を調整
    
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
        scale: ScaleType = ScaleType.PENTATONIC,
        base_octave: int = 4,                    # 基準オクターブ（C4 = 60）
        pitch_range: Tuple[int, int] = (48, 84), # 音域（C3～C6）
        volume_curve: str = "logarithmic",       # 音量カーブ（linear/logarithmic/exponential）
        spatial_range: float = 2.0,             # 空間範囲（メートル）
        default_instrument: InstrumentType = InstrumentType.MARIMBA,
        enable_adaptive_mapping: bool = True     # 適応的マッピング
    ):
        """
        初期化
        
        Args:
            scale: 使用する音階
            base_octave: 基準オクターブ
            pitch_range: 音域範囲 (min_note, max_note)
            volume_curve: 音量カーブタイプ
            spatial_range: 空間マッピング範囲
            default_instrument: デフォルト楽器
            enable_adaptive_mapping: 適応的マッピングを有効にするか
        """
        self.scale = scale
        self.base_octave = base_octave
        self.pitch_range = pitch_range
        self.volume_curve = volume_curve
        self.spatial_range = spatial_range
        self.default_instrument = default_instrument
        self.enable_adaptive_mapping = enable_adaptive_mapping
        
        # 音階パターンを初期化
        self.scale_patterns = self._initialize_scale_patterns()
        
        # 楽器マッピングテーブル
        self.instrument_mappings = self._initialize_instrument_mappings()
        
        # 適応的パラメータ
        self.adaptive_params = {
            'y_range': [0.0, 1.0],           # Y座標の実測範囲
            'velocity_range': [0.0, 1.0],    # 速度の実測範囲
            'x_range': [-1.0, 1.0],          # X座標の実測範囲
            'samples_count': 0,
            'update_interval': 100           # 適応更新間隔
        }
        
        # パフォーマンス統計
        self.stats = {
            'total_mappings': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_mapping_time_ms': 0.0,
            'instrument_usage': {inst: 0 for inst in InstrumentType},
            'note_distribution': np.zeros(128)  # MIDI note distribution
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
        
        # ゲイン（音色ごとの基準値）
        gain = self._map_gain(instrument)
        
        # 適応的パラメータ更新
        if self.enable_adaptive_mapping:
            self._update_adaptive_params(event)
        
        audio_params = AudioParameters(
            pitch=float(pitch),
            velocity=float(velocity),
            duration=float(duration),
            instrument=instrument,
            timbre=float(timbre),
            brightness=float(brightness),
            pan=float(pan),
            distance=float(distance),
            reverb=float(reverb),
            attack=float(attack),
            decay=float(decay),
            sustain=float(sustain),
            release=float(release),
            gain=float(gain),
            event_id=event.event_id,
            hand_id=event.hand_id,
            timestamp=float(event.timestamp)
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
        scale_notes = self.scale_patterns[self.scale]
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
        """衝突速度・強度を音量にマッピング"""
        # 速度成分（適応的範囲使用）
        vel_min, vel_max = self.adaptive_params['velocity_range']
        if vel_max > vel_min:
            velocity_component = (event.velocity - vel_min) / (vel_max - vel_min)
        else:
            velocity_component = 0.5
        
        velocity_component = max(0.0, min(1.0, velocity_component))
        
        # 強度成分（8段階）
        intensity_component = event.intensity.value / CollisionIntensity.MAXIMUM.value
        
        # 侵入深度成分
        depth_component = min(event.penetration_depth * 100, 1.0)
        
        # 重み付き結合
        combined_velocity = (
            velocity_component * 0.4 + 
            intensity_component * 0.4 + 
            depth_component * 0.2
        )
        
        # 音量カーブ適用
        if self.volume_curve == "logarithmic":
            # 対数カーブ（より自然な音量変化）
            mapped_velocity = np.power(combined_velocity, 0.5)
        elif self.volume_curve == "exponential":
            # 指数カーブ（急激な変化）
            mapped_velocity = np.power(combined_velocity, 2.0)
        else:  # linear
            mapped_velocity = combined_velocity
        
        return max(0.01, min(1.0, mapped_velocity))  # 完全無音は避ける
    
    def _map_duration(self, event: CollisionEvent) -> float:
        """持続時間をマッピング

        打楽器的な衝突は短いパルスが望ましいため、これまでよりも短い基本値を採用。
        面積・強度による伸び幅も抑え、全体として 0.15-0.8 秒程度に収まるようにする。
        """

        # --- 1. 基本持続時間を短縮 ---------------------------------------
        base_duration = 0.25  # 250 ms – percussive hit

        # --- 2. 面積による微調整 ----------------------------------------
        #   最大 40 % まで延長
        area_factor = 1.0 + min(event.contact_area * 400, 0.4)

        # --- 3. 衝突強度による微調整 ------------------------------------
        intensity_factor = 0.8 + (event.intensity.value / CollisionIntensity.MAXIMUM.value) * 0.3

        duration = base_duration * area_factor * intensity_factor

        # --- 4. 安全クランプ -------------------------------------------
        return max(0.15, min(0.8, duration))
    
    def _select_instrument(self, event: CollisionEvent) -> InstrumentType:
        """楽器を選択"""
        # デフォルト楽器から開始
        instrument = self.default_instrument
        
        # 衝突タイプによる楽器選択
        if hasattr(event, 'collision_type'):
            collision_type_str = str(event.collision_type)
            if "FACE_COLLISION" in collision_type_str:
                instrument = InstrumentType.MARIMBA
            elif "EDGE_COLLISION" in collision_type_str:
                instrument = InstrumentType.BELL
            elif "VERTEX_COLLISION" in collision_type_str:
                instrument = InstrumentType.CRYSTAL
        
        # 高度（Y座標）による楽器選択
        y_pos = event.contact_position[1]
        if y_pos > 0.7:
            instrument = InstrumentType.BELL
        elif y_pos < 0.3:
            instrument = InstrumentType.DRUM
        
        # 音色ヒントによる調整
        if hasattr(event, 'timbre_hint'):
            if event.timbre_hint > 0.8:
                instrument = InstrumentType.CRYSTAL
            elif event.timbre_hint < 0.2:
                instrument = InstrumentType.WATER_DROP
        
        return instrument
    
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
        x_pos = float(event.contact_position[0])  # numpy.float64 → float変換
        
        if x_max > x_min:
            normalized_x = (x_pos - x_min) / (x_max - x_min)
        else:
            normalized_x = 0.5
        
        # -1.0（左）～ 1.0（右）にマッピング
        pan = (normalized_x - 0.5) * 2.0
        return float(max(-1.0, min(1.0, pan)))  # 確実にPython floatで返す
    
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
    
    def _map_envelope(self, event: CollisionEvent, instrument: InstrumentType) -> Tuple[float, float, float, float]:
        """楽器タイプに応じたエンベロープをマッピング"""
        
        # 楽器ごとのデフォルトエンベロープ（打楽器らしく短く調整）
        envelopes = {
            InstrumentType.MARIMBA: (0.005, 0.15, 0.0, 0.8),     # 即座にアタック、sustainなし、自然減衰
            InstrumentType.SYNTH_PAD: (0.2, 0.3, 0.9, 1.0),
            InstrumentType.BELL: (0.01, 0.3, 0.0, 1.5),          # ベルも短いsustain
            InstrumentType.CRYSTAL: (0.003, 0.1, 0.0, 1.2),      # クリスタルも打楽器的
            InstrumentType.DRUM: (0.001, 0.05, 0.0, 0.15),       # ドラムは非常に短い
            InstrumentType.WATER_DROP: (0.02, 0.08, 0.0, 0.25),  # 水滴も短く
            InstrumentType.WIND: (0.5, 0.5, 0.9, 1.5),
            InstrumentType.STRING: (0.1, 0.2, 0.8, 0.6)
        }
        
        attack, decay, sustain, release = envelopes.get(
            instrument, (0.05, 0.2, 0.8, 0.5)
        )
        
        # 強度による調整
        intensity_factor = event.intensity.value / CollisionIntensity.MAXIMUM.value
        
        # アタックは強度が高いほど短く
        attack *= (1.0 - intensity_factor * 0.5)
        
        # リリースは接触面積が大きいほど長く
        area_factor = min(event.contact_area * 1000, 2.0)
        release *= area_factor
        
        # Sustain=0.0 を許可して完全なワンショットに
        return (
            max(0.001, attack),
            max(0.01, decay),
            max(0.0, min(1.0, sustain)),  # ← 0.0 OK
            max(0.05, release)
        )

    def _map_gain(self, instrument: InstrumentType) -> float:
        """音色別に基準ゲインを設定（FM や Noise 系は出力が大きい）"""
        loud_instruments = {
            InstrumentType.MARIMBA,
            InstrumentType.BELL,
            InstrumentType.DRUM,
            InstrumentType.CRYSTAL,
        }

        if instrument in loud_instruments:
            return 0.35  # -9 dBFS 相当
        elif instrument == InstrumentType.SYNTH_PAD:
            return 0.6
        else:
            return 0.5

    def _initialize_scale_patterns(self) -> Dict[ScaleType, List[int]]:
        """音階パターンを初期化"""
        return {
            ScaleType.CHROMATIC: list(range(12)),
            ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
            ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.PENTATONIC: [0, 2, 4, 7, 9],
            ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
            ScaleType.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
            ScaleType.JAPANESE: [0, 1, 5, 7, 8]  # 雅楽音階
        }
    
    def _initialize_instrument_mappings(self) -> Dict[InstrumentType, Dict[str, float]]:
        """楽器特性マッピングを初期化"""
        return {
            InstrumentType.MARIMBA: {
                'brightness_base': 0.6,
                'decay_multiplier': 1.0,
                'reverb_affinity': 0.3
            },
            InstrumentType.SYNTH_PAD: {
                'brightness_base': 0.4,
                'decay_multiplier': 3.0,
                'reverb_affinity': 0.8
            },
            InstrumentType.BELL: {
                'brightness_base': 0.8,
                'decay_multiplier': 2.0,
                'reverb_affinity': 0.6
            },
            InstrumentType.CRYSTAL: {
                'brightness_base': 0.9,
                'decay_multiplier': 1.5,
                'reverb_affinity': 0.4
            },
            InstrumentType.DRUM: {
                'brightness_base': 0.5,
                'decay_multiplier': 0.3,
                'reverb_affinity': 0.2
            },
            InstrumentType.WATER_DROP: {
                'brightness_base': 0.7,
                'decay_multiplier': 0.5,
                'reverb_affinity': 0.5
            },
            InstrumentType.WIND: {
                'brightness_base': 0.3,
                'decay_multiplier': 4.0,
                'reverb_affinity': 0.9
            },
            InstrumentType.STRING: {
                'brightness_base': 0.6,
                'decay_multiplier': 2.0,
                'reverb_affinity': 0.5
            }
        }
    
    def _update_adaptive_params(self, event: CollisionEvent):
        """適応的パラメータを更新"""
        self.adaptive_params['samples_count'] += 1
        
        if self.adaptive_params['samples_count'] % self.adaptive_params['update_interval'] == 0:
            # Y座標範囲の適応更新
            y_pos = event.contact_position[1]
            y_min, y_max = self.adaptive_params['y_range']
            self.adaptive_params['y_range'] = [
                min(y_min, y_pos),
                max(y_max, y_pos)
            ]
            
            # 速度範囲の適応更新
            vel_min, vel_max = self.adaptive_params['velocity_range']
            self.adaptive_params['velocity_range'] = [
                min(vel_min, event.velocity),
                max(vel_max, event.velocity)
            ]
            
            # X座標範囲の適応更新
            x_pos = event.contact_position[0]
            x_min, x_max = self.adaptive_params['x_range']
            self.adaptive_params['x_range'] = [
                min(x_min, x_pos),
                max(x_max, x_pos)
            ]
    
    def _update_stats(self, elapsed_ms: float, audio_params: AudioParameters):
        """パフォーマンス統計更新"""
        self.stats['total_mappings'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_mappings']
        self.stats['last_mapping_time_ms'] = elapsed_ms
        
        # 楽器使用統計
        self.stats['instrument_usage'][audio_params.instrument] += 1
        
        # 音高分布統計
        midi_note = audio_params.midi_note
        if 0 <= midi_note < 128:
            self.stats['note_distribution'][midi_note] += 1
    
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
            'instrument_usage': {inst: 0 for inst in InstrumentType},
            'note_distribution': np.zeros(128)
        }
        
        self.adaptive_params['samples_count'] = 0
    
    def set_scale(self, scale: ScaleType):
        """音階を変更"""
        self.scale = scale
    
    def set_pitch_range(self, min_note: int, max_note: int):
        """音域を変更"""
        self.pitch_range = (min_note, max_note)
    
    def set_spatial_range(self, range_meters: float):
        """空間範囲を変更"""
        self.spatial_range = range_meters


# 便利関数

def map_collision_to_audio(
    event: CollisionEvent,
    scale: ScaleType = ScaleType.PENTATONIC,
    instrument: InstrumentType = InstrumentType.MARIMBA
) -> AudioParameters:
    """
    衝突イベントを音響パラメータにマッピング（簡単なインターフェース）
    
    Args:
        event: 衝突イベント
        scale: 使用する音階
        instrument: デフォルト楽器
        
    Returns:
        音響パラメータ
    """
    mapper = AudioMapper(scale=scale, default_instrument=instrument)
    return mapper.map_collision_event(event)


def batch_map_collisions(
    events: List[CollisionEvent],
    mapper: Optional[AudioMapper] = None
) -> List[AudioParameters]:
    """
    複数の衝突イベントを一括マッピング
    
    Args:
        events: 衝突イベントのリスト
        mapper: 使用するマッパー（Noneの場合はデフォルト作成）
        
    Returns:
        音響パラメータのリスト
    """
    if mapper is None:
        mapper = AudioMapper()
    
    return [mapper.map_collision_event(event) for event in events]


def create_scale_mapper(
    scale: ScaleType,
    root_note: int = 60  # C4
) -> AudioMapper:
    """
    指定音階のマッパーを作成
    
    Args:
        scale: 音階タイプ
        root_note: ルート音のMIDIノート番号
        
    Returns:
        設定されたマッパー
    """
    base_octave = root_note // 12
    pitch_range = (root_note - 12, root_note + 24)  # ±1オクターブ + 1オクターブ
    
    return AudioMapper(
        scale=scale,
        base_octave=base_octave,
        pitch_range=pitch_range
    )
