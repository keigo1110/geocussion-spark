#!/usr/bin/env python3
"""
イベントシステムモジュール

Observer/Publisherパターンによるイベント駆動アーキテクチャ
"""

from .base import Event, EventType, EventHandler, EventDispatcher, get_event_dispatcher
from .pipeline_events import (
    PipelineEvent,
    FrameProcessedEvent,
    StageCompletedEvent,
    ErrorEvent,
    ConfigChangedEvent,
    MeshUpdatedEvent,
    CollisionDetectedEvent,
    AudioTriggeredEvent,
    HandsDetectedEvent,
    PointCloudGeneratedEvent
)
from .ui_events import (
    UIEvent,
    KeyPressedEvent,
    WindowResizedEvent,
    ViewportChangedEvent
)

__all__ = [
    # Base
    'Event',
    'EventType',
    'EventHandler',
    'EventDispatcher',
    'get_event_dispatcher',
    
    # Pipeline Events
    'PipelineEvent',
    'FrameProcessedEvent',
    'StageCompletedEvent',
    'ErrorEvent',
    'ConfigChangedEvent',
    'MeshUpdatedEvent',
    'CollisionDetectedEvent',
    'AudioTriggeredEvent',
    'HandsDetectedEvent',
    'PointCloudGeneratedEvent',
    
    # UI Events
    'UIEvent',
    'KeyPressedEvent',
    'WindowResizedEvent',
    'ViewportChangedEvent'
]