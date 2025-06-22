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
from .mapping import AudioParameters
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
        self.instrument_voices: Dict[str, List[str]] = {}
        
        # パフォーマンス統計
        self.stats = {
            'total_voices_created': 0,
            'total_voices_stolen': 0,
            'max_simultaneous_voices': 0,
            'average_polyphony': 0.0,
            'steal_strategy_usage': {strategy: 0 for strategy in StealStrategy},
            'instrument_usage': {},
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
                self.instrument_voices[audio_params.instrument] = self.instrument_voices.get(audio_params.instrument, []) + [voice_id]
                
                # 統計更新
                self.stats['total_voices_created'] += 1
                self.stats['instrument_usage'][audio_params.instrument] = self.stats['instrument_usage'].get(audio_params.instrument, 0) + 1
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
                voice_info = self.active_voices.pop(voice_id)
                
                # シンセサイザーからボイス停止
                if voice_info.synthesizer_voice_id:
                    fade_duration = 0.05 if fade_out else 0.0
                    self.synthesizer.stop_voice(
                        voice_info.synthesizer_voice_id,
                        fadeout_sec=fade_duration
                    )

                # 楽器別リストからも削除
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
        """ボイスパラメータを動的に更新"""
        with self._lock:
            if voice_id not in self.active_voices:
                return
            
            voice_info = self.active_voices[voice_id]
            
            # 空間位置の更新
            if spatial_position is not None:
                # 再計算
                updated_params = self._apply_spatial_processing(voice_info.audio_params, spatial_position)
                volume = updated_params.velocity
                pan = updated_params.pan
                reverb = updated_params.reverb
                voice_info.spatial_position = spatial_position
            
            # シンセサイザーにパラメータ変更を通知
            self.synthesizer.update_voice_parameters(
                voice_id=voice_info.synthesizer_voice_id,
                volume=volume,
                pan=pan,
                reverb=reverb
            )
            
            # ローカル情報も更新
            if volume is not None: voice_info.current_volume = volume
            if pan is not None: voice_info.current_pan = pan
            if reverb is not None: voice_info.current_reverb = reverb
    
    def cleanup_finished_voices(self):
        """
        再生が完了したボイスをクリーンアップする。
        実際の処理はAudioSynthesizer側に移譲する。
        """
        self.synthesizer.cleanup_finished_voices()
    
    def _steal_voice(self, new_audio_params: AudioParameters, new_priority: int) -> Optional[str]:
        """
        ボイススティール戦略に基づいて、停止するボイスを選択して停止する
        """
        victim_id: Optional[str] = None
        
        with self._lock:
            if not self.active_voices:
                return None

            active_voices_list = list(self.active_voices.values())

            if self.steal_strategy == StealStrategy.OLDEST:
                victim_id = min(active_voices_list, key=lambda v: v.start_time).voice_id
            
            elif self.steal_strategy == StealStrategy.QUIETEST:
                victim_id = min(active_voices_list, key=lambda v: v.current_volume).voice_id

            elif self.steal_strategy == StealStrategy.LOWEST_PRIORITY:
                # 優先度が同じ場合は最も古いものを選択
                victim_id = min(active_voices_list, key=lambda v: (v.priority, v.start_time)).voice_id

            # TODO: 他戦略の実装
            
            if victim_id:
                # 短いフェードアウト付きでボイスを停止
                self.deallocate_voice(victim_id, fade_out=True)
                self.stats['total_voices_stolen'] += 1
                self.stats['steal_strategy_usage'][self.steal_strategy] += 1
                return victim_id
        
        return None
    
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
                attenuation = self._calculate_distance_attenuation(float(distance))
                processed_params.velocity *= attenuation
            
            # 距離によるリバーブ調整
            distance = np.linalg.norm(spatial_position - self.spatial_config.listener_position)
            reverb_factor = min(float(distance) / self.spatial_config.room_size, 1.0)
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
        """ステレオパンを計算"""
        # Note: This method is called within a lock
        x_pos = position[0]
        listener_x = self.spatial_config.listener_position[0]
        room_width = self.spatial_config.room_size
        
        # -1 (左) から +1 (右) に正規化
        pan = np.clip((x_pos - listener_x) / (room_width / 2.0), -1.0, 1.0)
        return pan
    
    def _calculate_distance_attenuation(self, distance: float) -> float:
        """距離減衰を計算"""
        # Note: This method is called within a lock
        # 簡単な逆二乗則
        return 1.0 / (1.0 + distance**2 * 0.1)
    
    def get_voice_info(self, voice_id: str) -> Optional[VoiceInfo]:
        """ボイス情報を取得"""
        with self._lock:
            return self.active_voices.get(voice_id)
    
    def get_active_voices_by_instrument(self, instrument: str) -> List[str]:
        """楽器名でアクティブなボイスを取得"""
        with self._lock:
            return self.instrument_voices.get(instrument.upper(), [])
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計を取得"""
        with self._lock:
            # 移動平均を計算
            if self.stats['total_voices_created'] > 0:
                self.stats['average_polyphony'] = (
                    np.mean([v.age for v in self.active_voices.values()])
                    if self.active_voices else 0.0
                )
            return self.stats.copy()
    
    def get_active_voice_count(self) -> int:
        """アクティブなボイス数を取得"""
        with self._lock:
            return len(self.active_voices)
    
    def reset_stats(self):
        """統計情報をリセット"""
        with self._lock:
            self.stats = {
                'total_voices_created': 0,
                'total_voices_stolen': 0,
                'max_simultaneous_voices': 0,
                'average_polyphony': 0.0,
                'steal_strategy_usage': {strategy.value: 0 for strategy in StealStrategy},
                'instrument_usage': {},
                'spatial_processing_time_ms': 0.0
            }
    
    def set_steal_strategy(self, strategy: StealStrategy):
        """ボイススティール戦略を設定"""
        with self._lock:
            self.steal_strategy = strategy
    
    def set_max_polyphony(self, max_polyphony: int):
        """最大ポリフォニー数を設定"""
        with self._lock:
            self.max_polyphony = max_polyphony
    
    def update_spatial_config(self, config: SpatialConfig):
        """空間音響設定を更新"""
        with self._lock:
            self.spatial_config = config
    
    def shutdown(self, fade_out_duration: float = 0.5):
        """ボイスマネージャをシャットダウン"""
        with self._lock:
            all_voices = list(self.active_voices.keys())
            for voice_id in all_voices:
                self.deallocate_voice(voice_id, fade_out=True)
            # ToDo: Implement graceful fadeout for shutdown
            print("VoiceManager shut down.")


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