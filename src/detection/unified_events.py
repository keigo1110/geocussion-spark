#!/usr/bin/env python3
"""
統一手イベントシステム
複数の手検出・追跡・衝突イベントを単一のイベントストリームに統合
外部アプリケーションからの複雑な状態管理を隠蔽
"""

import time
import threading
from typing import List, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hands3d import Hand3DResult
    from .tracker_60fps import HighFrequencyTrackedHand
    from ..collision.events import CollisionEvent

from ..data_types import HandednessType
from .tracker_60fps import HandTrackingState
from src import get_logger

logger = get_logger(__name__)


class UnifiedHandEventType(Enum):
    """統一手イベントタイプ"""
    # 基本イベント
    HAND_DETECTED = "hand_detected"        # 手が検出された
    HAND_TRACKING = "hand_tracking"        # 手を追跡中
    HAND_LOST = "hand_lost"               # 手を見失った
    HAND_RECOVERED = "hand_recovered"      # 手を再発見
    
    # 動作イベント
    HAND_MOVING = "hand_moving"           # 手が動いている
    HAND_STOPPED = "hand_stopped"         # 手が停止した
    HAND_ACCELERATING = "hand_accelerating"  # 手が加速中
    
    # 衝突イベント
    COLLISION_DETECTED = "collision_detected"  # 衝突検出
    COLLISION_ONGOING = "collision_ongoing"    # 衝突継続中
    COLLISION_ENDED = "collision_ended"        # 衝突終了
    
    # システムイベント
    TRACKING_STARTED = "tracking_started"      # トラッキング開始
    TRACKING_STOPPED = "tracking_stopped"     # トラッキング停止
    CALIBRATION_NEEDED = "calibration_needed" # キャリブレーション必要
    
    # 音響イベント
    AUDIO_TRIGGERED = "audio_triggered"        # 音響再生開始


@dataclass
class UnifiedHandEvent:
    """統一手イベント"""
    # 基本情報
    event_type: UnifiedHandEventType
    hand_id: str
    timestamp: float
    
    # 手の情報
    handedness: HandednessType
    position_3d: np.ndarray
    velocity_3d: np.ndarray
    acceleration_3d: np.ndarray
    
    # 信頼度
    confidence: float
    tracking_quality: float  # 0.0-1.0
    
    # 動作情報
    speed: float  # m/s
    acceleration_magnitude: float  # m/s²
    is_moving: bool
    is_accelerating: bool
    
    # 衝突情報（衝突イベント時のみ）
    collision_position: Optional[np.ndarray] = None
    collision_normal: Optional[np.ndarray] = None
    collision_intensity: Optional[float] = None
    collision_penetration: Optional[float] = None
    collision_depth: Optional[float] = None
    
    # 音響情報（音響イベント時のみ）
    audio_frequency: Optional[float] = None
    audio_velocity: Optional[float] = None
    
    # メタデータ
    interpolated: bool = False  # 補間による推定値か
    prediction_horizon: Optional[float] = None  # 予測時間（秒）
    
    # 生データ参照（デバッグ用）
    raw_detection: Optional[Any] = None
    raw_tracking: Optional[Any] = None
    raw_collision: Optional[Any] = None
    
    @property
    def position_2d(self) -> np.ndarray:
        """2D位置（X, Z座標）"""
        return np.array([self.position_3d[0], self.position_3d[2]])
    
    @property
    def velocity_2d(self) -> np.ndarray:
        """2D速度（X, Z成分）"""
        return np.array([self.velocity_3d[0], self.velocity_3d[2]])
    
    @property
    def height(self) -> float:
        """高さ（Y座標）"""
        return self.position_3d[1]
    
    @property
    def is_collision_event(self) -> bool:
        """衝突イベントかどうか"""
        return self.event_type in [
            UnifiedHandEventType.COLLISION_DETECTED,
            UnifiedHandEventType.COLLISION_ONGOING,
            UnifiedHandEventType.COLLISION_ENDED
        ]
    
    @property
    def is_motion_event(self) -> bool:
        """動作イベントかどうか"""
        return self.event_type in [
            UnifiedHandEventType.HAND_MOVING,
            UnifiedHandEventType.HAND_STOPPED,
            UnifiedHandEventType.HAND_ACCELERATING
        ]
    
    @property
    def hand_position(self) -> np.ndarray:
        """手の位置（3D）"""
        return self.position_3d
    
    @property
    def hand_velocity(self) -> np.ndarray:
        """手の速度（3D）"""
        return self.velocity_3d


class UnifiedHandEventStream:
    """統一手イベントストリーム"""
    
    def __init__(
        self,
        max_event_history: int = 1000,
        motion_threshold: float = 0.05,  # 5cm/s
        acceleration_threshold: float = 0.2,  # 0.2m/s²
        collision_timeout: float = 0.5,  # 0.5秒
        enable_prediction: bool = True
    ):
        """
        初期化
        
        Args:
            max_event_history: 最大イベント履歴数
            motion_threshold: 動作検出閾値
            acceleration_threshold: 加速度検出閾値
            collision_timeout: 衝突タイムアウト
            enable_prediction: 予測機能有効化
        """
        self.max_event_history = max_event_history
        self.motion_threshold = motion_threshold
        self.acceleration_threshold = acceleration_threshold
        self.collision_timeout = collision_timeout
        self.enable_prediction = enable_prediction
        
        # イベント履歴
        self.event_history: deque = deque(maxlen=max_event_history)
        
        # 手の状態追跡
        self.hand_states: Dict[str, Dict[str, Any]] = {}
        
        # 衝突状態追跡
        self.collision_states: Dict[str, Dict[str, Any]] = {}
        
        # イベントリスナー
        self.event_listeners: List[Callable[[UnifiedHandEvent], None]] = []
        
        # 統計情報
        self.stats = {
            'total_events': 0,
            'events_by_type': {},
            'active_hands': 0,
            'active_collisions': 0,
            'avg_tracking_quality': 0.0
        }
        
        # スレッドセーフティ
        self._lock = threading.RLock()
        
        logger.info("UnifiedHandEventStream initialized")
    
    def add_event_listener(self, listener: Callable[[UnifiedHandEvent], None]) -> None:
        """イベントリスナーを追加"""
        with self._lock:
            self.event_listeners.append(listener)
            logger.debug(f"Added event listener: {listener}")
    
    def remove_event_listener(self, listener: Callable[[UnifiedHandEvent], None]) -> None:
        """イベントリスナーを削除"""
        with self._lock:
            if listener in self.event_listeners:
                self.event_listeners.remove(listener)
                logger.debug(f"Removed event listener: {listener}")
    
    def process_tracking_update(
        self,
        tracked_hands: List['HighFrequencyTrackedHand']
    ) -> List[UnifiedHandEvent]:
        """トラッキング更新を処理"""
        with self._lock:
            events = []
            current_time = time.perf_counter()
            
            # 現在のアクティブ手IDを取得
            current_hand_ids = {hand.id for hand in tracked_hands}
            previous_hand_ids = set(self.hand_states.keys())
            
            # 新しい手の検出
            new_hands = current_hand_ids - previous_hand_ids
            for hand in tracked_hands:
                if hand.id in new_hands:
                    event = self._create_detection_event(hand, current_time)
                    events.append(event)
            
            # 既存の手の更新
            for hand in tracked_hands:
                if hand.id in previous_hand_ids:
                    hand_events = self._update_hand_state(hand, current_time)
                    events.extend(hand_events)
            
            # 消失した手の処理
            lost_hands = previous_hand_ids - current_hand_ids
            for hand_id in lost_hands:
                event = self._create_lost_event(hand_id, current_time)
                events.append(event)
                del self.hand_states[hand_id]
            
            # イベント発火
            for event in events:
                self._fire_event(event)
            
            # 統計更新
            self._update_stats(tracked_hands)
            
            return events
    
    def process_collision_event(
        self,
        collision_event: 'CollisionEvent',
        hand_id: str
    ) -> Optional[UnifiedHandEvent]:
        """衝突イベントを処理"""
        with self._lock:
            current_time = time.perf_counter()
            
            # 衝突状態の更新
            if hand_id not in self.collision_states:
                # 新しい衝突
                self.collision_states[hand_id] = {
                    'start_time': current_time,
                    'last_update': current_time,
                    'event_count': 1
                }
                event_type = UnifiedHandEventType.COLLISION_DETECTED
            else:
                # 継続中の衝突
                self.collision_states[hand_id]['last_update'] = current_time
                self.collision_states[hand_id]['event_count'] += 1
                event_type = UnifiedHandEventType.COLLISION_ONGOING
            
            # 統一イベント作成
            event = self._create_collision_event(
                collision_event, hand_id, event_type, current_time
            )
            
            # イベント発火
            self._fire_event(event)
            
            return event
    
    def _create_detection_event(
        self,
        hand: 'HighFrequencyTrackedHand',
        timestamp: float
    ) -> UnifiedHandEvent:
        """検出イベントを作成"""
        # 手の状態を初期化
        self.hand_states[hand.id] = {
            'last_position': hand.position.copy(),
            'last_velocity': hand.velocity.copy(),
            'last_speed': hand.speed,
            'last_acceleration_magnitude': hand.acceleration_magnitude,
            'was_moving': hand.is_moving,
            'was_accelerating': hand.is_accelerating,
            'last_update': timestamp
        }
        
        # FSM状態に基づいてイベントタイプを決定
        if hand.fsm_state == HandTrackingState.RECOVERED:
            event_type = UnifiedHandEventType.HAND_RECOVERED
        else:
            event_type = UnifiedHandEventType.HAND_DETECTED
        
        return UnifiedHandEvent(
            event_type=event_type,
            hand_id=hand.id,
            timestamp=timestamp,
            handedness=hand.handedness,
            position_3d=hand.position.copy(),
            velocity_3d=hand.velocity.copy(),
            acceleration_3d=hand.acceleration.copy(),
            confidence=hand.confidence_tracking,
            tracking_quality=self._calculate_tracking_quality(hand),
            speed=hand.speed,
            acceleration_magnitude=hand.acceleration_magnitude,
            is_moving=hand.is_moving,
            is_accelerating=hand.is_accelerating,
            interpolated=hand.is_interpolated,
            raw_tracking=hand
        )
    
    def _update_hand_state(
        self,
        hand: 'HighFrequencyTrackedHand',
        timestamp: float
    ) -> List[UnifiedHandEvent]:
        """手の状態を更新"""
        events = []
        prev_state = self.hand_states[hand.id]
        
        # 基本トラッキングイベント
        tracking_event = UnifiedHandEvent(
            event_type=UnifiedHandEventType.HAND_TRACKING,
            hand_id=hand.id,
            timestamp=timestamp,
            handedness=hand.handedness,
            position_3d=hand.position.copy(),
            velocity_3d=hand.velocity.copy(),
            acceleration_3d=hand.acceleration.copy(),
            confidence=hand.confidence_tracking,
            tracking_quality=self._calculate_tracking_quality(hand),
            speed=hand.speed,
            acceleration_magnitude=hand.acceleration_magnitude,
            is_moving=hand.is_moving,
            is_accelerating=hand.is_accelerating,
            interpolated=hand.is_interpolated,
            raw_tracking=hand
        )
        events.append(tracking_event)
        
        # 動作状態変化の検出
        if not prev_state['was_moving'] and hand.is_moving:
            # 停止→動作
            motion_event = UnifiedHandEvent(
                event_type=UnifiedHandEventType.HAND_MOVING,
                hand_id=hand.id,
                timestamp=timestamp,
                handedness=hand.handedness,
                position_3d=hand.position.copy(),
                velocity_3d=hand.velocity.copy(),
                acceleration_3d=hand.acceleration.copy(),
                confidence=hand.confidence_tracking,
                tracking_quality=self._calculate_tracking_quality(hand),
                speed=hand.speed,
                acceleration_magnitude=hand.acceleration_magnitude,
                is_moving=hand.is_moving,
                is_accelerating=hand.is_accelerating,
                interpolated=hand.is_interpolated,
                raw_tracking=hand
            )
            events.append(motion_event)
        
        elif prev_state['was_moving'] and not hand.is_moving:
            # 動作→停止
            stop_event = UnifiedHandEvent(
                event_type=UnifiedHandEventType.HAND_STOPPED,
                hand_id=hand.id,
                timestamp=timestamp,
                handedness=hand.handedness,
                position_3d=hand.position.copy(),
                velocity_3d=hand.velocity.copy(),
                acceleration_3d=hand.acceleration.copy(),
                confidence=hand.confidence_tracking,
                tracking_quality=self._calculate_tracking_quality(hand),
                speed=hand.speed,
                acceleration_magnitude=hand.acceleration_magnitude,
                is_moving=hand.is_moving,
                is_accelerating=hand.is_accelerating,
                interpolated=hand.is_interpolated,
                raw_tracking=hand
            )
            events.append(stop_event)
        
        # 加速度変化の検出
        if not prev_state['was_accelerating'] and hand.is_accelerating:
            acceleration_event = UnifiedHandEvent(
                event_type=UnifiedHandEventType.HAND_ACCELERATING,
                hand_id=hand.id,
                timestamp=timestamp,
                handedness=hand.handedness,
                position_3d=hand.position.copy(),
                velocity_3d=hand.velocity.copy(),
                acceleration_3d=hand.acceleration.copy(),
                confidence=hand.confidence_tracking,
                tracking_quality=self._calculate_tracking_quality(hand),
                speed=hand.speed,
                acceleration_magnitude=hand.acceleration_magnitude,
                is_moving=hand.is_moving,
                is_accelerating=hand.is_accelerating,
                interpolated=hand.is_interpolated,
                raw_tracking=hand
            )
            events.append(acceleration_event)
        
        # 状態を更新
        prev_state.update({
            'last_position': hand.position.copy(),
            'last_velocity': hand.velocity.copy(),
            'last_speed': hand.speed,
            'last_acceleration_magnitude': hand.acceleration_magnitude,
            'was_moving': hand.is_moving,
            'was_accelerating': hand.is_accelerating,
            'last_update': timestamp
        })
        
        return events
    
    def _create_lost_event(self, hand_id: str, timestamp: float) -> UnifiedHandEvent:
        """消失イベントを作成"""
        prev_state = self.hand_states[hand_id]
        
        return UnifiedHandEvent(
            event_type=UnifiedHandEventType.HAND_LOST,
            hand_id=hand_id,
            timestamp=timestamp,
            handedness=HandednessType.UNKNOWN,
            position_3d=prev_state['last_position'],
            velocity_3d=prev_state['last_velocity'],
            acceleration_3d=np.zeros(3),
            confidence=0.0,
            tracking_quality=0.0,
            speed=prev_state['last_speed'],
            acceleration_magnitude=prev_state['last_acceleration_magnitude'],
            is_moving=prev_state['was_moving'],
            is_accelerating=prev_state['was_accelerating'],
            interpolated=False
        )
    
    def _create_collision_event(
        self,
        collision_event: 'CollisionEvent',
        hand_id: str,
        event_type: UnifiedHandEventType,
        timestamp: float
    ) -> UnifiedHandEvent:
        """衝突イベントを作成"""
        # 手の状態を取得
        hand_state = self.hand_states.get(hand_id)
        if hand_state:
            position = hand_state['last_position']
            velocity = hand_state['last_velocity']
            handedness = HandednessType.UNKNOWN  # 衝突イベントから推定困難
            confidence = 0.8  # 衝突検出の信頼度
        else:
            position = collision_event.hand_position
            velocity = np.zeros(3)
            handedness = HandednessType.UNKNOWN
            confidence = 0.6
        
        return UnifiedHandEvent(
            event_type=event_type,
            hand_id=hand_id,
            timestamp=timestamp,
            handedness=handedness,
            position_3d=position,
            velocity_3d=velocity,
            acceleration_3d=np.zeros(3),
            confidence=confidence,
            tracking_quality=0.8,
            speed=collision_event.velocity,
            acceleration_magnitude=0.0,
            is_moving=collision_event.velocity > self.motion_threshold,
            is_accelerating=False,
            collision_position=collision_event.contact_position.copy(),
            collision_normal=collision_event.surface_normal.copy(),
            collision_intensity=collision_event.intensity.value / 8.0,  # 0-1に正規化
            collision_penetration=collision_event.penetration_depth,
            interpolated=False,
            raw_collision=collision_event
        )
    
    def _calculate_tracking_quality(self, hand: 'HighFrequencyTrackedHand') -> float:
        """トラッキング品質を計算"""
        quality = 0.0
        
        # 基本信頼度
        quality += hand.confidence_tracking * 0.4
        
        # 補間状態による減点
        if hand.is_interpolated:
            interpolation_penalty = min(0.3, hand.interpolation_frames * 0.05)
            quality -= interpolation_penalty
        
        # トラック長による加点
        track_bonus = min(0.2, hand.track_length * 0.01)
        quality += track_bonus
        
        # 速度の安定性
        if hand.speed < 2.0:  # 2m/s以下で安定
            quality += 0.1
        
        return max(0.0, min(1.0, quality))
    
    def _fire_event(self, event: UnifiedHandEvent) -> None:
        """イベントを発火"""
        # 履歴に追加
        self.event_history.append(event)
        
        # 統計更新
        self.stats['total_events'] += 1
        event_type_str = event.event_type.value
        if event_type_str not in self.stats['events_by_type']:
            self.stats['events_by_type'][event_type_str] = 0
        self.stats['events_by_type'][event_type_str] += 1
        
        # リスナーに通知
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def _update_stats(self, tracked_hands: List['HighFrequencyTrackedHand']) -> None:
        """統計を更新"""
        self.stats['active_hands'] = len(tracked_hands)
        self.stats['active_collisions'] = len(self.collision_states)
        
        if tracked_hands:
            avg_quality = sum(
                self._calculate_tracking_quality(hand) for hand in tracked_hands
            ) / len(tracked_hands)
            self.stats['avg_tracking_quality'] = avg_quality
    
    def cleanup_old_collisions(self) -> None:
        """古い衝突状態をクリーンアップ"""
        current_time = time.perf_counter()
        to_remove = []
        
        for hand_id, collision_state in self.collision_states.items():
            if current_time - collision_state['last_update'] > self.collision_timeout:
                to_remove.append(hand_id)
                
                # 終了イベントを発火
                end_event = UnifiedHandEvent(
                    event_type=UnifiedHandEventType.COLLISION_ENDED,
                    hand_id=hand_id,
                    timestamp=current_time,
                    handedness=HandednessType.UNKNOWN,
                    position_3d=np.zeros(3),
                    velocity_3d=np.zeros(3),
                    acceleration_3d=np.zeros(3),
                    confidence=0.0,
                    tracking_quality=0.0,
                    speed=0.0,
                    acceleration_magnitude=0.0,
                    is_moving=False,
                    is_accelerating=False,
                    interpolated=False
                )
                self._fire_event(end_event)
        
        for hand_id in to_remove:
            del self.collision_states[hand_id]
    
    def get_recent_events(
        self,
        event_types: Optional[List[UnifiedHandEventType]] = None,
        hand_id: Optional[str] = None,
        max_age: Optional[float] = None
    ) -> List[UnifiedHandEvent]:
        """最近のイベントを取得"""
        with self._lock:
            events = list(self.event_history)
            
            # フィルタリング
            if event_types:
                events = [e for e in events if e.event_type in event_types]
            
            if hand_id:
                events = [e for e in events if e.hand_id == hand_id]
            
            if max_age:
                current_time = time.perf_counter()
                events = [e for e in events if current_time - e.timestamp <= max_age]
            
            return events
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return self.stats.copy()
    
    def add_event(self, event: UnifiedHandEvent) -> None:
        """イベントを追加"""
        with self._lock:
            self.event_history.append(event)
            
            # 履歴サイズ制限
            if len(self.event_history) > self.max_event_history:
                self.event_history.popleft()
            
            # 統計更新
            self.stats['total_events'] += 1
            event_type_str = event.event_type.value
            if event_type_str not in self.stats['events_by_type']:
                self.stats['events_by_type'][event_type_str] = 0
            self.stats['events_by_type'][event_type_str] += 1
            
            # リスナーに通知
            for listener in self.event_listeners:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"Event listener error: {e}")
    
    def clear(self) -> None:
        """イベント履歴をクリア"""
        with self._lock:
            self.event_history.clear()
            self.hand_states.clear()
            self.collision_states.clear()
    
    def reset(self) -> None:
        """イベントストリームをリセット"""
        with self._lock:
            self.event_history.clear()
            self.hand_states.clear()
            self.collision_states.clear()
            self.stats = {
                'total_events': 0,
                'events_by_type': {},
                'active_hands': 0,
                'active_collisions': 0,
                'avg_tracking_quality': 0.0
            }
            logger.info("UnifiedHandEventStream reset")


# 便利関数

def create_unified_event_stream(
    max_events: int = 1000,
    event_retention_time: float = 5.0,
    motion_threshold: float = 0.05,
    acceleration_threshold: float = 0.2
) -> UnifiedHandEventStream:
    """統一イベントストリームを作成"""
    return UnifiedHandEventStream(
        max_event_history=max_events,
        motion_threshold=motion_threshold,
        acceleration_threshold=acceleration_threshold,
        collision_timeout=event_retention_time,
        enable_prediction=True
    )


def filter_events_by_type(
    events: List[UnifiedHandEvent],
    event_types: List[UnifiedHandEventType]
) -> List[UnifiedHandEvent]:
    """イベントタイプでフィルタリング"""
    return [event for event in events if event.event_type in event_types]


def get_collision_events(events: List[UnifiedHandEvent]) -> List[UnifiedHandEvent]:
    """衝突イベントのみを抽出"""
    return filter_events_by_type(events, [
        UnifiedHandEventType.COLLISION_DETECTED,
        UnifiedHandEventType.COLLISION_ONGOING,
        UnifiedHandEventType.COLLISION_ENDED
    ])


def get_motion_events(events: List[UnifiedHandEvent]) -> List[UnifiedHandEvent]:
    """動作イベントのみを抽出"""
    return filter_events_by_type(events, [
        UnifiedHandEventType.HAND_MOVING,
        UnifiedHandEventType.HAND_STOPPED,
        UnifiedHandEventType.HAND_ACCELERATING
    ]) 