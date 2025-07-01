#!/usr/bin/env python3
"""
UI関連イベント

ユーザーインターフェースで発生するイベントの定義
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
from enum import Enum

from .base import Event, EventType


class KeyCode(Enum):
    """キーコード定義"""
    # 制御キー
    ESCAPE = 27
    ENTER = 13
    SPACE = 32
    TAB = 9
    
    # 矢印キー
    UP = 265
    DOWN = 264
    LEFT = 263
    RIGHT = 262
    
    # ファンクションキー
    F1 = 290
    F2 = 291
    F3 = 292
    F4 = 293
    F5 = 294
    
    # 文字キー
    A = 65
    B = 66
    C = 67
    D = 68
    E = 69
    F = 70
    G = 71
    H = 72
    I = 73
    J = 74
    K = 75
    L = 76
    M = 77
    N = 78
    O = 79
    P = 80
    Q = 81
    R = 82
    S = 83
    T = 84
    U = 85
    V = 86
    W = 87
    X = 88
    Y = 89
    Z = 90
    
    # 数字キー
    NUM_0 = 48
    NUM_1 = 49
    NUM_2 = 50
    NUM_3 = 51
    NUM_4 = 52
    NUM_5 = 53
    NUM_6 = 54
    NUM_7 = 55
    NUM_8 = 56
    NUM_9 = 57
    
    # 特殊キー
    DELETE = 261
    BACKSPACE = 259
    PAGE_UP = 266
    PAGE_DOWN = 267
    HOME = 268
    END = 269
    
    # 不明なキー
    UNKNOWN = -1


@dataclass
class UIEvent(Event):
    """UIイベントの基底クラス"""
    window_id: Optional[str] = None
    

@dataclass
class KeyPressedEvent(UIEvent):
    """キー押下イベント"""
    def __init__(self, key_code: int, shift: bool = False, 
                 ctrl: bool = False, alt: bool = False) -> None:
        super().__init__(
            event_type=EventType.KEY_PRESSED,
            data={
                'key_code': key_code,
                'shift': shift,
                'ctrl': ctrl,
                'alt': alt
            }
        )
        self.key_code = key_code
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        
    @property
    def key(self) -> KeyCode:
        """キーコードをKeyCode列挙型に変換"""
        for key in KeyCode:
            if key.value == self.key_code:
                return key
        return KeyCode.UNKNOWN
    
    @property
    def is_modifier_pressed(self) -> bool:
        """修飾キーが押されているか"""
        return self.shift or self.ctrl or self.alt
    
    @property
    def char(self) -> Optional[str]:
        """文字キーの場合、対応する文字を返す"""
        if 32 <= self.key_code <= 126:  # 印字可能文字
            char = chr(self.key_code)
            if not self.shift and 65 <= self.key_code <= 90:
                char = char.lower()
            return char
        return None


@dataclass
class WindowResizedEvent(UIEvent):
    """ウィンドウサイズ変更イベント"""
    def __init__(self, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> None:
        super().__init__(
            event_type=EventType.WINDOW_RESIZED,
            data={
                'old_width': old_size[0],
                'old_height': old_size[1],
                'new_width': new_size[0],
                'new_height': new_size[1]
            }
        )
        self.old_size = old_size
        self.new_size = new_size
        
    @property
    def old_width(self) -> int:
        """変更前の幅"""
        return self.old_size[0]
    
    @property
    def old_height(self) -> int:
        """変更前の高さ"""
        return self.old_size[1]
    
    @property
    def new_width(self) -> int:
        """変更後の幅"""
        return self.new_size[0]
    
    @property
    def new_height(self) -> int:
        """変更後の高さ"""
        return self.new_size[1]
    
    @property
    def aspect_ratio_changed(self) -> bool:
        """アスペクト比が変更されたか"""
        old_ratio = self.old_width / self.old_height if self.old_height > 0 else 0
        new_ratio = self.new_width / self.new_height if self.new_height > 0 else 0
        return abs(old_ratio - new_ratio) > 0.01


@dataclass
class ViewportChangedEvent(UIEvent):
    """ビューポート変更イベント"""
    def __init__(self, camera_position: Tuple[float, float, float],
                 camera_target: Tuple[float, float, float],
                 camera_up: Tuple[float, float, float],
                 fov: float = 60.0) -> None:
        super().__init__(
            event_type=EventType.VIEWPORT_CHANGED,
            data={
                'camera_position': camera_position,
                'camera_target': camera_target,
                'camera_up': camera_up,
                'fov': fov
            }
        )
        self.camera_position = camera_position
        self.camera_target = camera_target
        self.camera_up = camera_up
        self.fov = fov
        
    @property
    def distance_to_target(self) -> float:
        """カメラからターゲットまでの距離"""
        import numpy as np
        pos = np.array(self.camera_position)
        target = np.array(self.camera_target)
        return np.linalg.norm(target - pos)


@dataclass
class MouseEvent(UIEvent):
    """マウスイベントの基底クラス"""
    x: int = 0
    y: int = 0
    button: int = 0  # 0: left, 1: middle, 2: right
    

@dataclass
class MouseClickedEvent(MouseEvent):
    """マウスクリックイベント"""
    def __init__(self, x: int, y: int, button: int = 0) -> None:
        super().__init__(
            event_type=EventType.KEY_PRESSED,  # マウスイベント用の新しいタイプが必要
            data={
                'x': x,
                'y': y,
                'button': button
            }
        )
        self.x = x
        self.y = y
        self.button = button


@dataclass
class MouseMovedEvent(MouseEvent):
    """マウス移動イベント"""
    def __init__(self, x: int, y: int, dx: int, dy: int) -> None:
        super().__init__(
            event_type=EventType.KEY_PRESSED,  # マウスイベント用の新しいタイプが必要
            data={
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy
            }
        )
        self.x = x
        self.y = y
        self.dx = dx  # X方向の移動量
        self.dy = dy  # Y方向の移動量


@dataclass
class RenderRequestEvent(UIEvent):
    """レンダリング要求イベント"""
    def __init__(self, force_update: bool = False) -> None:
        super().__init__(
            event_type=EventType.VIEWPORT_CHANGED,
            data={
                'force_update': force_update
            }
        )
        self.force_update = force_update