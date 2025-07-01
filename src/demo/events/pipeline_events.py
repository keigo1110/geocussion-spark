#!/usr/bin/env python3
"""
パイプライン関連イベント

パイプライン処理で発生するイベントの定義
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from .base import Event, EventType
from ...types import FrameData, Hand3D
from ...detection import TrackedHand
from ...collision.events import CollisionEvent


@dataclass
class PipelineEvent(Event):
    """パイプラインイベントの基底クラス"""
    frame_number: int = 0
    pipeline_stage: str = ""


@dataclass
class FrameProcessedEvent(PipelineEvent):
    """フレーム処理完了イベント"""
    def __init__(self, frame_data: FrameData, frame_number: int) -> None:
        super().__init__(
            event_type=EventType.FRAME_PROCESSED,
            data={
                'frame_data': frame_data,
                'frame_number': frame_number
            }
        )
        self.frame_number = frame_number
        self.pipeline_stage = "input"
        self.frame_data = frame_data
        
    @property
    def has_depth(self) -> bool:
        """深度データが存在するか"""
        return self.frame_data.depth_frame is not None
    
    @property
    def has_color(self) -> bool:
        """カラーデータが存在するか"""
        return self.frame_data.color_frame is not None


@dataclass
class StageCompletedEvent(PipelineEvent):
    """ステージ処理完了イベント"""
    def __init__(self, stage_name: str, processing_time_ms: float, 
                 success: bool = True, error_message: Optional[str] = None) -> None:
        super().__init__(
            event_type=EventType.STAGE_COMPLETED,
            data={
                'stage_name': stage_name,
                'processing_time_ms': processing_time_ms,
                'success': success,
                'error_message': error_message
            }
        )
        self.pipeline_stage = stage_name
        self.stage_name = stage_name
        self.processing_time_ms = processing_time_ms
        self.success = success
        self.error_message = error_message


@dataclass
class ErrorEvent(PipelineEvent):
    """エラーイベント"""
    def __init__(self, stage_name: str, error_message: str, 
                 error_type: str = "UNKNOWN", recoverable: bool = True) -> None:
        super().__init__(
            event_type=EventType.PIPELINE_ERROR,
            data={
                'stage_name': stage_name,
                'error_message': error_message,
                'error_type': error_type,
                'recoverable': recoverable
            }
        )
        self.pipeline_stage = stage_name
        self.stage_name = stage_name
        self.error_message = error_message
        self.error_type = error_type
        self.recoverable = recoverable


@dataclass
class ConfigChangedEvent(PipelineEvent):
    """設定変更イベント"""
    def __init__(self, config_key: str, old_value: Any, new_value: Any) -> None:
        super().__init__(
            event_type=EventType.CONFIG_CHANGED,
            data={
                'config_key': config_key,
                'old_value': old_value,
                'new_value': new_value
            }
        )
        self.config_key = config_key
        self.old_value = old_value
        self.new_value = new_value


@dataclass
class MeshUpdatedEvent(PipelineEvent):
    """メッシュ更新イベント"""
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, 
                 colors: Optional[np.ndarray] = None) -> None:
        super().__init__(
            event_type=EventType.MESH_UPDATED,
            data={
                'vertex_count': len(vertices),
                'triangle_count': len(triangles),
                'has_colors': colors is not None
            }
        )
        self.pipeline_stage = "mesh"
        self.vertices = vertices
        self.triangles = triangles
        self.colors = colors
        
    @property
    def vertex_count(self) -> int:
        """頂点数"""
        return len(self.vertices)
    
    @property
    def triangle_count(self) -> int:
        """三角形数"""
        return len(self.triangles)


@dataclass
class CollisionDetectedEvent(PipelineEvent):
    """衝突検出イベント"""
    def __init__(self, collision_events: List[CollisionEvent], 
                 active_collisions: Dict[int, List[CollisionEvent]]) -> None:
        super().__init__(
            event_type=EventType.COLLISION_DETECTED,
            data={
                'collision_count': len(collision_events),
                'active_hand_count': len(active_collisions)
            }
        )
        self.pipeline_stage = "collision"
        self.collision_events = collision_events
        self.active_collisions = active_collisions
        
    @property
    def collision_count(self) -> int:
        """検出された衝突数"""
        return len(self.collision_events)
    
    @property
    def active_hands(self) -> List[int]:
        """衝突中の手のID"""
        return list(self.active_collisions.keys())


@dataclass
class AudioTriggeredEvent(PipelineEvent):
    """音響トリガーイベント"""
    def __init__(self, collision_event: CollisionEvent, note: int, 
                 velocity: float, duration: float) -> None:
        super().__init__(
            event_type=EventType.AUDIO_TRIGGERED,
            data={
                'note': note,
                'velocity': velocity,
                'duration': duration,
                'hand_id': collision_event.hand_id,
                'position': collision_event.position.tolist()
            }
        )
        self.pipeline_stage = "audio"
        self.collision_event = collision_event
        self.note = note
        self.velocity = velocity
        self.duration = duration


@dataclass 
class HandsDetectedEvent(PipelineEvent):
    """手検出イベント"""
    def __init__(self, hands_2d: List[Any], hands_3d: List[Hand3D], 
                 tracked_hands: List[TrackedHand]) -> None:
        super().__init__(
            event_type=EventType.STAGE_COMPLETED,
            data={
                'hands_2d_count': len(hands_2d),
                'hands_3d_count': len(hands_3d),
                'tracked_count': len(tracked_hands)
            }
        )
        self.pipeline_stage = "detection"
        self.hands_2d = hands_2d
        self.hands_3d = hands_3d
        self.tracked_hands = tracked_hands


@dataclass
class PointCloudGeneratedEvent(PipelineEvent):
    """点群生成イベント"""
    def __init__(self, point_cloud: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
        super().__init__(
            event_type=EventType.STAGE_COMPLETED,
            data={
                'point_count': len(point_cloud),
                'has_colors': colors is not None
            }
        )
        self.pipeline_stage = "input"
        self.point_cloud = point_cloud
        self.colors = colors
        
    @property
    def point_count(self) -> int:
        """点群の点数"""
        return len(self.point_cloud)