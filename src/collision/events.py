#!/usr/bin/env python3
"""
衝突検出 - イベント生成

衝突結果を音響生成フェーズで使用できる形式にエンコードし、
イベントキューで管理する機能を提供します。
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum, IntEnum
from collections import deque
import numpy as np

# 他フェーズとの連携
from .sphere_tri import CollisionInfo, ContactPoint, CollisionType
from ..mesh.attributes import MeshAttributes


class EventType(Enum):
    """イベントタイプの列挙"""
    COLLISION_START = "start"
    COLLISION_CONTINUE = "continue"
    COLLISION_END = "end"
    COLLISION_IMPACT = "impact"


class CollisionIntensity(IntEnum):
    """衝突強度の列挙（音量レベルに対応）"""
    SILENT = 0
    WHISPER = 1
    SOFT = 2
    MEDIUM_SOFT = 3
    MEDIUM = 4
    MEDIUM_LOUD = 5
    LOUD = 6
    VERY_LOUD = 7
    MAXIMUM = 8


@dataclass
class CollisionEvent:
    """衝突イベントデータ"""
    event_id: str
    event_type: EventType
    timestamp: float
    duration_ms: float
    
    contact_position: np.ndarray
    hand_position: np.ndarray
    surface_normal: np.ndarray
    
    intensity: CollisionIntensity
    velocity: float
    penetration_depth: float
    contact_area: float
    
    pitch_hint: float
    timbre_hint: float
    spatial_position: np.ndarray
    
    triangle_index: int
    hand_id: str
    collision_type: CollisionType
    surface_properties: Dict[str, float] = field(default_factory=dict)
    
    @property
    def velocity_db(self) -> float:
        """速度をdBスケールに変換"""
        if self.velocity <= 0:
            return -60.0
        return min(20 * np.log10(self.velocity * 100), 0.0)
    
    @property
    def intensity_db(self) -> float:
        """強度をdBスケールに変換"""
        return -60.0 + (self.intensity.value / CollisionIntensity.MAXIMUM.value) * 60.0
    
    @property
    def midi_note(self) -> int:
        """Y座標をMIDIノート番号に変換"""
        normalized_y = max(0.0, min(1.0, self.pitch_hint))
        return int(48 + normalized_y * 36)


class CollisionEventQueue:
    """衝突イベントキューマネージャー"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.event_queue = deque(maxlen=max_queue_size)
        self.active_events = {}
        self.event_counter = 0
        self.stats = {
            'total_events_created': 0,
            'collisions_detected': 0,
            'queue_overflows': 0
        }
    
    def create_event(
        self,
        collision_info: CollisionInfo,
        hand_id: str,
        hand_position: np.ndarray,
        hand_velocity: Optional[np.ndarray] = None
    ) -> Optional[CollisionEvent]:
        """衝突イベントを作成"""
        if not collision_info.has_collision:
            return self._end_active_event(hand_id)
        
        primary_contact = collision_info.closest_point
        if primary_contact is None:
            return None
        
        velocity = np.linalg.norm(hand_velocity) if hand_velocity is not None else 0.0
        intensity = self._calculate_intensity(collision_info, velocity)
        
        event = CollisionEvent(
            event_id=f"collision_{self.event_counter:06d}",
            event_type=EventType.COLLISION_START,
            timestamp=time.perf_counter(),
            duration_ms=0.0,
            
            contact_position=primary_contact.position.copy(),
            hand_position=hand_position.copy(),
            surface_normal=primary_contact.normal.copy(),
            
            intensity=intensity,
            velocity=velocity,
            penetration_depth=primary_contact.depth,
            contact_area=self._estimate_contact_area(collision_info),
            
            pitch_hint=max(0.0, min(1.0, primary_contact.position[1])),
            timbre_hint=0.5,
            spatial_position=np.array([primary_contact.position[0], 0.0, primary_contact.position[2]]),
            
            triangle_index=primary_contact.triangle_index,
            hand_id=hand_id,
            collision_type=primary_contact.collision_type
        )
        
        self.event_counter += 1
        self.active_events[hand_id] = event
        self._add_to_queue(event)
        
        return event
    
    def _calculate_intensity(self, collision_info: CollisionInfo, velocity: float) -> CollisionIntensity:
        """衝突強度を計算"""
        velocity_score = min(velocity * 20, 1.0)
        depth_score = min(collision_info.max_penetration_depth * 100, 1.0)
        combined_score = (velocity_score + depth_score) / 2.0
        
        if combined_score < 0.125:
            return CollisionIntensity.WHISPER
        elif combined_score < 0.25:
            return CollisionIntensity.SOFT
        elif combined_score < 0.375:
            return CollisionIntensity.MEDIUM_SOFT
        elif combined_score < 0.5:
            return CollisionIntensity.MEDIUM
        elif combined_score < 0.625:
            return CollisionIntensity.MEDIUM_LOUD
        elif combined_score < 0.75:
            return CollisionIntensity.LOUD
        elif combined_score < 0.875:
            return CollisionIntensity.VERY_LOUD
        else:
            return CollisionIntensity.MAXIMUM
    
    def _estimate_contact_area(self, collision_info: CollisionInfo) -> float:
        """接触面積を推定"""
        base_area = 0.001
        num_contacts = collision_info.num_contacts
        avg_depth = collision_info.total_penetration_depth / max(num_contacts, 1)
        return min(base_area * num_contacts * (1.0 + avg_depth * 50), 0.01)
    
    def _end_active_event(self, hand_id: str) -> Optional[CollisionEvent]:
        """アクティブイベントを終了"""
        active_event = self.active_events.pop(hand_id, None)
        if active_event is None:
            return None
        
        end_event = CollisionEvent(
            event_id=f"collision_{self.event_counter:06d}",
            event_type=EventType.COLLISION_END,
            timestamp=time.perf_counter(),
            duration_ms=(time.perf_counter() - active_event.timestamp) * 1000,
            
            contact_position=active_event.contact_position,
            hand_position=active_event.hand_position,
            surface_normal=active_event.surface_normal,
            
            intensity=CollisionIntensity.SILENT,
            velocity=0.0,
            penetration_depth=0.0,
            contact_area=0.0,
            
            pitch_hint=active_event.pitch_hint,
            timbre_hint=active_event.timbre_hint,
            spatial_position=active_event.spatial_position,
            
            triangle_index=active_event.triangle_index,
            hand_id=hand_id,
            collision_type=active_event.collision_type
        )
        
        self.event_counter += 1
        self._add_to_queue(end_event)
        return end_event
    
    def _add_to_queue(self, event: CollisionEvent):
        """イベントをキューに追加"""
        if len(self.event_queue) >= self.max_queue_size:
            self.stats['queue_overflows'] += 1
        
        self.event_queue.append(event)
        self.stats['total_events_created'] += 1
    
    def get_events(self, max_events: Optional[int] = None) -> List[CollisionEvent]:
        """キューからイベントを取得"""
        if max_events is None:
            events = list(self.event_queue)
            self.event_queue.clear()
        else:
            events = []
            for _ in range(min(max_events, len(self.event_queue))):
                events.append(self.event_queue.popleft())
        return events
    
    def get_stats(self) -> dict:
        """統計取得"""
        return self.stats.copy()


# 便利関数
def create_collision_event(
    collision_info: CollisionInfo,
    hand_id: str,
    hand_position: np.ndarray,
    hand_velocity: Optional[np.ndarray] = None
) -> Optional[CollisionEvent]:
    """衝突イベントを作成（簡単なインターフェース）"""
    queue = CollisionEventQueue()
    return queue.create_event(collision_info, hand_id, hand_position, hand_velocity)


def process_collision_events(
    collision_infos: List[CollisionInfo],
    hand_ids: List[str],
    hand_positions: List[np.ndarray],
    hand_velocities: Optional[List[np.ndarray]] = None
) -> List[CollisionEvent]:
    """複数の衝突情報を一括処理"""
    queue = CollisionEventQueue()
    events = []
    
    for i, (collision_info, hand_id, hand_pos) in enumerate(zip(collision_infos, hand_ids, hand_positions)):
        velocity = hand_velocities[i] if hand_velocities else None
        event = queue.create_event(collision_info, hand_id, hand_pos, velocity)
        if event:
            events.append(event)
    
    return events 