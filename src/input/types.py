#!/usr/bin/env python3
"""
入力フェーズの共通データ構造

このモジュールは、入力に関連するdataclassやEnumを定義し、
モジュール間の循環参照を防ぐために使用されます。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


class OBFormat(Enum):
    """OrbbecSDKカラーフォーマット"""
    RGB = "RGB"
    BGR = "BGR"
    MJPG = "MJPG"


@dataclass
class FrameData:
    """フレームデータ構造"""
    depth_image: np.ndarray
    color_data: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_number: int = 0
    
    @property
    def has_color(self) -> bool:
        """カラーデータが存在するか"""
        return self.color_data is not None


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    def project_point(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """3D点を2D画像座標に投影"""
        x, y, z = point_3d
        if z <= 0:
            return -1, -1
        
        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)
        
        if 0 <= u < self.width and 0 <= v < self.height:
            return u, v
        return -1, -1 