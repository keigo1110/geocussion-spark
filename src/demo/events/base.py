#!/usr/bin/env python3
"""
イベントシステムの基本クラス

Observer/Publisherパターンの実装
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading
import weakref
from ... import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """イベントタイプの列挙"""
    # Pipeline Events
    FRAME_PROCESSED = auto()
    STAGE_COMPLETED = auto()
    PIPELINE_ERROR = auto()
    CONFIG_CHANGED = auto()
    
    # Mesh Events
    MESH_UPDATED = auto()
    MESH_GENERATION_FAILED = auto()
    
    # Collision Events
    COLLISION_DETECTED = auto()
    COLLISION_ENDED = auto()
    
    # Audio Events
    AUDIO_TRIGGERED = auto()
    AUDIO_STOPPED = auto()
    
    # UI Events
    KEY_PRESSED = auto()
    WINDOW_RESIZED = auto()
    VIEWPORT_CHANGED = auto()
    
    # System Events
    INITIALIZATION_COMPLETE = auto()
    SHUTDOWN_REQUESTED = auto()


@dataclass
class Event:
    """イベント基底クラス"""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[Any] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """イベントIDを生成"""
        self.event_id = f"{self.event_type.name}_{self.timestamp.timestamp()}"


class EventHandler(ABC):
    """イベントハンドラーのインターフェース"""
    
    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """
        イベントを処理
        
        Args:
            event: 処理するイベント
        """
        pass
    
    def can_handle(self, event_type: EventType) -> bool:
        """
        特定のイベントタイプを処理できるか
        
        Args:
            event_type: イベントタイプ
            
        Returns:
            処理可能な場合True
        """
        return True


class EventDispatcher:
    """
    イベントディスパッチャー
    
    イベントの登録、配信、管理を行う
    """
    
    def __init__(self, async_dispatch: bool = True) -> None:
        """
        初期化
        
        Args:
            async_dispatch: 非同期配信を有効にするか
        """
        self.async_dispatch = async_dispatch
        self._handlers: Dict[EventType, List[weakref.ref]] = {}
        self._lock = threading.RLock()
        self._event_queue: List[Event] = []
        self._dispatch_thread: Optional[threading.Thread] = None
        self._running = False
        
        if async_dispatch:
            self._start_dispatch_thread()
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        イベントハンドラーを登録
        
        Args:
            event_type: 監視するイベントタイプ
            handler: イベントハンドラー
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            
            # 弱参照で保持（循環参照を防ぐ）
            handler_ref = weakref.ref(handler, self._create_cleanup_callback(event_type))
            self._handlers[event_type].append(handler_ref)
            
            logger.debug(f"Handler subscribed to {event_type.name}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        イベントハンドラーの登録解除
        
        Args:
            event_type: イベントタイプ
            handler: 解除するハンドラー
        """
        with self._lock:
            if event_type in self._handlers:
                # 該当するハンドラーの弱参照を削除
                self._handlers[event_type] = [
                    ref for ref in self._handlers[event_type]
                    if ref() is not handler
                ]
    
    def publish(self, event: Event) -> None:
        """
        イベントを発行
        
        Args:
            event: 発行するイベント
        """
        if self.async_dispatch:
            # 非同期配信用のキューに追加
            with self._lock:
                self._event_queue.append(event)
        else:
            # 同期配信
            self._dispatch_event(event)
    
    def _dispatch_event(self, event: Event) -> None:
        """
        イベントを配信
        
        Args:
            event: 配信するイベント
        """
        with self._lock:
            handlers = self._handlers.get(event.event_type, [])
            # 死んだ参照を除去
            alive_handlers = []
            for handler_ref in handlers:
                handler = handler_ref()
                if handler is not None:
                    alive_handlers.append(handler_ref)
                    try:
                        if handler.can_handle(event.event_type):
                            handler.handle_event(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
            
            # 生きているハンドラーのみ保持
            self._handlers[event.event_type] = alive_handlers
    
    def _create_cleanup_callback(self, event_type: EventType) -> Callable:
        """
        弱参照クリーンアップコールバックを作成
        
        Args:
            event_type: イベントタイプ
            
        Returns:
            クリーンアップコールバック
        """
        def cleanup(ref):
            with self._lock:
                if event_type in self._handlers:
                    self._handlers[event_type] = [
                        r for r in self._handlers[event_type] if r is not ref
                    ]
        return cleanup
    
    def _start_dispatch_thread(self) -> None:
        """非同期配信スレッドを開始"""
        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True
        )
        self._dispatch_thread.start()
    
    def _dispatch_loop(self) -> None:
        """非同期配信ループ"""
        while self._running:
            try:
                # キューからイベントを取得
                with self._lock:
                    if self._event_queue:
                        event = self._event_queue.pop(0)
                    else:
                        event = None
                
                if event:
                    self._dispatch_event(event)
                else:
                    # キューが空の場合は少し待機
                    threading.Event().wait(0.01)
                    
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}")
    
    def shutdown(self) -> None:
        """ディスパッチャーをシャットダウン"""
        self._running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=1.0)
        
        with self._lock:
            self._handlers.clear()
            self._event_queue.clear()


# グローバルイベントディスパッチャー（シングルトン）
_global_dispatcher: Optional[EventDispatcher] = None
_dispatcher_lock = threading.Lock()


def get_event_dispatcher() -> EventDispatcher:
    """
    グローバルイベントディスパッチャーを取得
    
    Returns:
        イベントディスパッチャー
    """
    global _global_dispatcher
    with _dispatcher_lock:
        if _global_dispatcher is None:
            _global_dispatcher = EventDispatcher()
        return _global_dispatcher