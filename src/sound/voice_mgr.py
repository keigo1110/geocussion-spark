#!/usr/bin/env python3
"""
音響生成 - ボイス管理システム

ポリフォニー制御、ボイススティール戦略、ステレオ空間配置を行う
高度なボイス管理システムを提供します。
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, IntEnum
from queue import PriorityQueue
import numpy as np

# 他フェーズとの連携
from .mapping import AudioParameters, InstrumentType
from .synth import AudioSynthesizer


class VoiceState(Enum):
    """ボイス状態の列挙"""
    INACTIVE = "inactive"
    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"
    FINISHED = "finished"


class StealStrategy(Enum):
    """ボイススティール戦略の列挙"""
    OLDEST = "oldest"           # 最も古いボイスを停止
    QUIETEST = "quietest"       # 最も音量の小さいボイスを停止
    LOWEST_PRIORITY = "lowest_priority"  # 最も優先度の低いボイスを停止
    SAME_INSTRUMENT = "same_instrument"  # 同じ楽器のボイスを停止
    NEAREST_PITCH = "nearest_pitch"      # 最も近い音高のボイスを停止


class SpatialMode(Enum):
    """空間音響モードの列挙"""
    STEREO_PAN = "stereo_pan"   # ステレオパンニング
    BINAURAL = "binaural"       # バイノーラル
    SURROUND = "surround"       # サラウンド
    AMBISONICS = "ambisonics"   # アンビソニックス


@dataclass
class VoiceInfo:
    """ボイス情報データ"""
    voice_id: str
    audio_params: AudioParameters
    synthesizer_voice_id: Optional[str]
    
    start_time: float
    state: VoiceState = VoiceState.INACTIVE
    priority: int = 5  # 1(低) - 10(高)
    
    # 空間情報
    spatial_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    distance: float = 1.0
    
    # 動的パラメータ
    current_volume: float = 1.0
    current_pan: float = 0.0
    current_reverb: float = 0.3
    
    # 統計情報
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    @property
    def age(self) -> float:
        """ボイスの経過時間（秒）"""
        return time.perf_counter() - self.start_time
    
    @property
    def is_active(self) -> bool:
        """アクティブかどうか"""
        return self.state not in [VoiceState.INACTIVE, VoiceState.FINISHED]
    
    @property
    def estimated_remaining_time(self) -> float:
        """推定残り時間（秒）"""
        total_duration = self.audio_params.duration
        return max(0.0, total_duration - self.age)


@dataclass
class SpatialConfig:
    """空間音響設定"""
    mode: SpatialMode = SpatialMode.STEREO_PAN
    listener_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    listener_orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    room_size: float = 10.0           # 部屋のサイズ（メートル）
    reverb_decay: float = 2.0         # リバーブ減衰時間（秒）
    doppler_factor: float = 1.0       # ドップラー効果の係数
    distance_attenuation: bool = True  # 距離減衰を有効にするか
    air_absorption: float = 0.01      # 空気吸収係数


class VoiceManager:
    """ボイス管理システム"""
    
    def __init__(
        self,
        synthesizer: AudioSynthesizer,
        max_polyphony: int = 32,
        steal_strategy: StealStrategy = StealStrategy.OLDEST,
        enable_priority_system: bool = True,
        spatial_config: Optional[SpatialConfig] = None
    ):
        """
        初期化
        
        Args:
            synthesizer: 音響シンセサイザー
            max_polyphony: 最大ポリフォニー数
            steal_strategy: ボイススティール戦略
            enable_priority_system: 優先度システムを有効にするか
            spatial_config: 空間音響設定
        """
        self.synthesizer = synthesizer
        self.max_polyphony = max_polyphony
        self.steal_strategy = steal_strategy
        self.enable_priority_system = enable_priority_system
        self.spatial_config = spatial_config or SpatialConfig()
        
        # ボイス管理
        self.active_voices: Dict[str, VoiceInfo] = {}
        self.voice_queue = PriorityQueue()  # 優先度付きキュー
        self.voice_counter = 0
        
        # 楽器ごとのボイス管理
        self.instrument_voices: Dict[InstrumentType, List[str]] = {
            instrument: [] for instrument in InstrumentType
        }
        
        # パフォーマンス統計
        self.stats = {
            'total_voices_created': 0,
            'total_voices_stolen': 0,
            'max_simultaneous_voices': 0,
            'average_polyphony': 0.0,
            'steal_strategy_usage': {strategy: 0 for strategy in StealStrategy},
            'instrument_usage': {instrument: 0 for instrument in InstrumentType},
            'spatial_processing_time_ms': 0.0
        }
        
        # スレッド安全性
        self._lock = threading.Lock()
        
        print(f"VoiceManager initialized: max_polyphony={max_polyphony}, strategy={steal_strategy.value}")
    
    def allocate_voice(
        self,
        audio_params: AudioParameters,
        priority: int = 5,
        spatial_position: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        ボイスを割り当て
        
        Args:
            audio_params: 音響パラメータ
            priority: 優先度 (1-10)
            spatial_position: 空間位置
            
        Returns:
            ボイスID（成功した場合）
        """
        with self._lock:
            start_time = time.perf_counter()
            
            try:
                # ボイスID生成
                self.voice_counter += 1
                voice_id = f"voice_{self.voice_counter:06d}_{audio_params.hand_id}"
                
                # 空間配置パラメータ適用
                if spatial_position is not None:
                    audio_params = self._apply_spatial_processing(audio_params, spatial_position)
                
                # ポリフォニー制限チェック
                if len(self.active_voices) >= self.max_polyphony:
                    stolen_voice_id = self._steal_voice(audio_params, priority)
                    if stolen_voice_id is None:
                        return None  # スティールに失敗
                
                # シンセサイザーでボイス生成
                synth_voice_id = self.synthesizer.play_audio_parameters(audio_params)
                if synth_voice_id is None:
                    return None
                
                # ボイス情報作成
                voice_info = VoiceInfo(
                    voice_id=voice_id,
                    audio_params=audio_params,
                    synthesizer_voice_id=synth_voice_id,
                    start_time=time.perf_counter(),
                    state=VoiceState.ATTACK,
                    priority=priority,
                    spatial_position=spatial_position or np.array([0.0, 0.0, 0.0]),
                    current_volume=audio_params.velocity,
                    current_pan=audio_params.pan,
                    current_reverb=audio_params.reverb
                )
                
                # ボイス管理に追加
                self.active_voices[voice_id] = voice_info
                self.instrument_voices[audio_params.instrument].append(voice_id)
                
                # 統計更新
                self.stats['total_voices_created'] += 1
                self.stats['instrument_usage'][audio_params.instrument] += 1
                self.stats['max_simultaneous_voices'] = max(
                    self.stats['max_simultaneous_voices'],
                    len(self.active_voices)
                )
                
                # 空間処理時間統計
                spatial_time_ms = (time.perf_counter() - start_time) * 1000
                self.stats['spatial_processing_time_ms'] = spatial_time_ms
                
                return voice_id
                
            except Exception as e:
                print(f"Error allocating voice: {e}")
                return None
    
    def deallocate_voice(self, voice_id: str, fade_out: bool = True):
        """
        ボイスを解放
        
        Args:
            voice_id: ボイスID
            fade_out: フェードアウトするか
        """
        with self._lock:
            if voice_id not in self.active_voices:
                return
            
            try:
                voice_info = self.active_voices[voice_id]
                
                # シンセサイザーからボイス停止
                if voice_info.synthesizer_voice_id:
                    self.synthesizer.stop_voice(voice_info.synthesizer_voice_id)
                
                # アクティブリストから削除
                del self.active_voices[voice_id]
                
                # 楽器リストからも削除
                instrument = voice_info.audio_params.instrument
                if instrument in self.instrument_voices:
                    if voice_id in self.instrument_voices[instrument]:
                        self.instrument_voices[instrument].remove(voice_id)
            
            except Exception as e:
                print(f"Error deallocating voice {voice_id}: {e}")
    
    def update_voice_parameters(
        self,
        voice_id: str,
        volume: Optional[float] = None,
        pan: Optional[float] = None,
        reverb: Optional[float] = None,
        spatial_position: Optional[np.ndarray] = None
    ):
        """
        ボイスパラメータをリアルタイム更新
        
        Args:
            voice_id: ボイスID
            volume: 音量
            pan: パンニング
            reverb: リバーブ量
            spatial_position: 空間位置
        """
        with self._lock:
            if voice_id not in self.active_voices:
                return
            
            voice_info = self.active_voices[voice_id]
            
            # パラメータ更新
            if volume is not None:
                voice_info.current_volume = max(0.0, min(1.0, volume))
            
            if pan is not None:
                voice_info.current_pan = max(-1.0, min(1.0, pan))
            
            if reverb is not None:
                voice_info.current_reverb = max(0.0, min(1.0, reverb))
            
            if spatial_position is not None:
                voice_info.spatial_position = spatial_position.copy()
                # 空間処理を再適用
                updated_params = self._apply_spatial_processing(
                    voice_info.audio_params, spatial_position
                )
                voice_info.current_pan = updated_params.pan
                voice_info.current_reverb = updated_params.reverb
            
            # TODO: シンセサイザーへのリアルタイム更新
    
    def cleanup_finished_voices(self):
        """終了したボイスをクリーンアップ"""
        with self._lock:
            finished_voices = []
            
            for voice_id, voice_info in self.active_voices.items():
                if voice_info.estimated_remaining_time <= 0:
                    finished_voices.append(voice_id)
            
            for voice_id in finished_voices:
                self.deallocate_voice(voice_id, fade_out=False)
    
    def _steal_voice(self, new_audio_params: AudioParameters, new_priority: int) -> Optional[str]:
        """
        ボイススティール戦略に基づいてボイスを停止
        
        Args:
            new_audio_params: 新しい音響パラメータ
            new_priority: 新しいボイスの優先度
            
        Returns:
            停止したボイスのID
        """
        if not self.active_voices:
            return None
        
        candidates = list(self.active_voices.keys())
        stolen_voice_id = None
        
        if self.steal_strategy == StealStrategy.OLDEST:
            # 最も古いボイスを選択
            oldest_voice_id = min(candidates, key=lambda vid: self.active_voices[vid].start_time)
            stolen_voice_id = oldest_voice_id
            
        elif self.steal_strategy == StealStrategy.QUIETEST:
            # 最も音量の小さいボイスを選択
            quietest_voice_id = min(candidates, key=lambda vid: self.active_voices[vid].current_volume)
            stolen_voice_id = quietest_voice_id
            
        elif self.steal_strategy == StealStrategy.LOWEST_PRIORITY:
            # 最も優先度の低いボイスを選択
            if self.enable_priority_system:
                lowest_priority_voice_id = min(candidates, key=lambda vid: self.active_voices[vid].priority)
                if self.active_voices[lowest_priority_voice_id].priority < new_priority:
                    stolen_voice_id = lowest_priority_voice_id
            
        elif self.steal_strategy == StealStrategy.SAME_INSTRUMENT:
            # 同じ楽器のボイスを選択
            same_instrument_voices = [
                vid for vid in candidates
                if self.active_voices[vid].audio_params.instrument == new_audio_params.instrument
            ]
            if same_instrument_voices:
                # 同じ楽器の中で最も古いものを選択
                stolen_voice_id = min(same_instrument_voices, key=lambda vid: self.active_voices[vid].start_time)
            else:
                # 同じ楽器がない場合は最も古いボイス
                stolen_voice_id = min(candidates, key=lambda vid: self.active_voices[vid].start_time)
                
        elif self.steal_strategy == StealStrategy.NEAREST_PITCH:
            # 最も近い音高のボイスを選択
            new_pitch = new_audio_params.pitch
            nearest_voice_id = min(
                candidates,
                key=lambda vid: abs(self.active_voices[vid].audio_params.pitch - new_pitch)
            )
            stolen_voice_id = nearest_voice_id
        
        if stolen_voice_id:
            self.deallocate_voice(stolen_voice_id, fade_out=True)
            self.stats['total_voices_stolen'] += 1
            self.stats['steal_strategy_usage'][self.steal_strategy] += 1
            
        return stolen_voice_id
    
    def _apply_spatial_processing(
        self,
        audio_params: AudioParameters,
        spatial_position: np.ndarray
    ) -> AudioParameters:
        """
        空間音響処理を適用
        
        Args:
            audio_params: 音響パラメータ
            spatial_position: 空間位置
            
        Returns:
            空間処理が適用された音響パラメータ
        """
        # パラメータをコピー
        processed_params = audio_params
        
        if self.spatial_config.mode == SpatialMode.STEREO_PAN:
            # ステレオパンニング
            pan = self._calculate_stereo_pan(spatial_position)
            processed_params.pan = pan
            
            # 距離による音量減衰
            if self.spatial_config.distance_attenuation:
                distance = np.linalg.norm(spatial_position - self.spatial_config.listener_position)
                attenuation = self._calculate_distance_attenuation(distance)
                processed_params.velocity *= attenuation
            
            # 距離によるリバーブ調整
            distance = np.linalg.norm(spatial_position - self.spatial_config.listener_position)
            reverb_factor = min(distance / self.spatial_config.room_size, 1.0)
            processed_params.reverb = min(processed_params.reverb + reverb_factor * 0.3, 1.0)
        
        elif self.spatial_config.mode == SpatialMode.BINAURAL:
            # バイノーラル処理（TODO: HRTF実装）
            pass
            
        elif self.spatial_config.mode == SpatialMode.SURROUND:
            # サラウンド処理（TODO: 実装）
            pass
            
        elif self.spatial_config.mode == SpatialMode.AMBISONICS:
            # アンビソニックス処理（TODO: 実装）
            pass
        
        return processed_params
    
    def _calculate_stereo_pan(self, position: np.ndarray) -> float:
        """ステレオパンニングを計算"""
        # リスナーとの相対位置
        relative_pos = position - self.spatial_config.listener_position
        
        # X座標をパンニングに変換（-1.0～1.0）
        max_range = self.spatial_config.room_size / 2
        pan = np.clip(relative_pos[0] / max_range, -1.0, 1.0)
        
        return pan
    
    def _calculate_distance_attenuation(self, distance: float) -> float:
        """距離減衰を計算"""
        # 逆二乗則による減衰
        min_distance = 0.1  # 最小距離（ゼロ除算回避）
        effective_distance = max(distance, min_distance)
        
        # 1メートルでの基準音量を1.0とする
        attenuation = 1.0 / (effective_distance ** 2)
        
        # 空気吸収を考慮
        air_loss = np.exp(-self.spatial_config.air_absorption * distance)
        
        return min(attenuation * air_loss, 1.0)
    
    def get_voice_info(self, voice_id: str) -> Optional[VoiceInfo]:
        """ボイス情報を取得"""
        return self.active_voices.get(voice_id)
    
    def get_active_voices_by_instrument(self, instrument: InstrumentType) -> List[str]:
        """楽器別のアクティブボイス一覧を取得"""
        return self.instrument_voices[instrument].copy()
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'current_active_voices': len(self.active_voices),
                'polyphony_usage_percent': (len(self.active_voices) / self.max_polyphony) * 100,
                'voice_distribution_by_instrument': {
                    instrument.value: len(voices)
                    for instrument, voices in self.instrument_voices.items()
                }
            })
            
            # 平均ポリフォニー計算
            if self.stats['total_voices_created'] > 0:
                stats['average_polyphony'] = (
                    sum(len(voices) for voices in self.instrument_voices.values()) /
                    self.stats['total_voices_created']
                )
            
            return stats
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_voices_created': 0,
            'total_voices_stolen': 0,
            'max_simultaneous_voices': 0,
            'average_polyphony': 0.0,
            'steal_strategy_usage': {strategy: 0 for strategy in StealStrategy},
            'instrument_usage': {instrument: 0 for instrument in InstrumentType},
            'spatial_processing_time_ms': 0.0
        }
    
    def set_steal_strategy(self, strategy: StealStrategy):
        """ボイススティール戦略を変更"""
        self.steal_strategy = strategy
    
    def set_max_polyphony(self, max_polyphony: int):
        """最大ポリフォニー数を変更"""
        self.max_polyphony = max(1, max_polyphony)
    
    def update_spatial_config(self, config: SpatialConfig):
        """空間音響設定を更新"""
        self.spatial_config = config
    
    def stop_all_voices(self, fade_out_time: float = 0.05):
        """
        全てのアクティブなボイスを停止
        
        Args:
            fade_out_time: フェードアウト時間（秒）
        """
        with self._lock:
            # active_voicesのキーのリストをコピーしてイテレート（ループ内で辞書を変更するため）
            voice_ids_to_stop = list(self.active_voices.keys())
            
            for voice_id in voice_ids_to_stop:
                voice_info = self.active_voices.get(voice_id)
                if voice_info and voice_info.synthesizer_voice_id:
                    # TODO: 本来はsynth側でフェードアウトを実装すべきだが、一旦即時停止
                    self.synthesizer.stop_voice(voice_info.synthesizer_voice_id)
            
            # 全てのボイスをクリア
            self.active_voices.clear()
            for instrument in self.instrument_voices:
                self.instrument_voices[instrument].clear()
            
            print(f"Stopped all {len(voice_ids_to_stop)} voices.")


# 便利関数

def create_voice_manager(
    synthesizer: AudioSynthesizer,
    max_polyphony: int = 32,
    steal_strategy: StealStrategy = StealStrategy.OLDEST
) -> VoiceManager:
    """
    ボイス管理システムを作成（簡単なインターフェース）
    
    Args:
        synthesizer: 音響シンセサイザー
        max_polyphony: 最大ポリフォニー
        steal_strategy: スティール戦略
        
    Returns:
        設定されたボイス管理システム
    """
    return VoiceManager(
        synthesizer=synthesizer,
        max_polyphony=max_polyphony,
        steal_strategy=steal_strategy
    )


def allocate_and_play(
    voice_manager: VoiceManager,
    audio_params: AudioParameters,
    priority: int = 5,
    spatial_position: Optional[np.ndarray] = None
) -> Optional[str]:
    """
    ボイスを割り当てて即座に再生
    
    Args:
        voice_manager: ボイス管理システム
        audio_params: 音響パラメータ
        priority: 優先度
        spatial_position: 空間位置
        
    Returns:
        ボイスID
    """
    return voice_manager.allocate_voice(audio_params, priority, spatial_position) 