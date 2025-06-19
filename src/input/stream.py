#!/usr/bin/env python3
"""
Orbbec カメラ抽象化クラス
既存のpoint_cloud_realtime_viewer.pyからフレーム取得処理を切り出し
"""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    from pyorbbecsdk import *
except ImportError:
    # テスト用のモック定義
    class Pipeline:
        def __init__(self): pass
        def get_stream_profile_list(self, sensor_type): return None
        def start(self, config): pass
        def stop(self): pass
        def wait_for_frames(self, timeout): return None
    
    class Config:
        def __init__(self): pass
        def enable_stream(self, profile): pass
    
    class OBSensorType:
        DEPTH_SENSOR = "depth"
        COLOR_SENSOR = "color"


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass 
class FrameData:
    """フレームデータコンテナ"""
    depth_frame: Optional[Any] = None
    color_frame: Optional[Any] = None
    timestamp_ms: float = 0.0
    frame_number: int = 0


class OrbbecCamera:
    """Orbbec カメラ抽象化クラス"""
    
    def __init__(self, enable_color: bool = True):
        """
        初期化
        
        Args:
            enable_color: カラーストリームを有効にするか
        """
        self.pipeline: Optional[Pipeline] = None
        self.config: Optional[Config] = None
        self.depth_intrinsics: Optional[CameraIntrinsics] = None
        self.color_intrinsics: Optional[CameraIntrinsics] = None
        self.has_color = False
        self.is_started = False
        self.enable_color = enable_color
        self.frame_count = 0
        
    def initialize(self) -> bool:
        """
        カメラを初期化
        
        Returns:
            成功した場合True
        """
        try:
            self.pipeline = Pipeline()
            self.config = Config()
            
            # 深度ストリーム設定
            if not self._setup_depth_stream():
                return False
                
            # カラーストリーム設定（オプション）
            if self.enable_color:
                self._setup_color_stream()
                
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def _setup_depth_stream(self) -> bool:
        """深度ストリームを設定"""
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            print("No depth sensor found!")
            return False
        
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(depth_profile)
        
        print(f"Depth: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps")
        
        # 深度カメラ内部パラメータ取得
        try:
            depth_intrinsic = depth_profile.get_intrinsic()
            self.depth_intrinsics = CameraIntrinsics(
                fx=depth_intrinsic.fx,
                fy=depth_intrinsic.fy,
                cx=depth_intrinsic.cx,
                cy=depth_intrinsic.cy,
                width=depth_profile.get_width(),
                height=depth_profile.get_height()
            )
            print(f"Depth intrinsics: fx={depth_intrinsic.fx}, fy={depth_intrinsic.fy}, "
                  f"cx={depth_intrinsic.cx}, cy={depth_intrinsic.cy}")
        except Exception:
            # デフォルト値使用
            print("Using default depth intrinsics")
            self.depth_intrinsics = CameraIntrinsics(
                fx=depth_profile.get_width(),
                fy=depth_profile.get_width(),
                cx=depth_profile.get_width() / 2,
                cy=depth_profile.get_height() / 2,
                width=depth_profile.get_width(),
                height=depth_profile.get_height()
            )
            
        return True
    
    def _setup_color_stream(self) -> None:
        """カラーストリームを設定"""
        try:
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profile_list is not None:
                color_profile = color_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(color_profile)
                self.has_color = True
                
                # カラーカメラ内部パラメータ取得
                try:
                    color_intrinsic = color_profile.get_intrinsic()
                    self.color_intrinsics = CameraIntrinsics(
                        fx=color_intrinsic.fx,
                        fy=color_intrinsic.fy,
                        cx=color_intrinsic.cx,
                        cy=color_intrinsic.cy,
                        width=color_profile.get_width(),
                        height=color_profile.get_height()
                    )
                except Exception:
                    self.color_intrinsics = None
                
                print(f"Color: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
        except Exception:
            print("No color sensor available")
            self.has_color = False
    
    def start(self) -> bool:
        """
        ストリーミング開始
        
        Returns:
            成功した場合True
        """
        if not self.pipeline or not self.config:
            print("Camera not initialized")
            return False
            
        try:
            self.pipeline.start(self.config)
            self.is_started = True
            print("Pipeline started!")
            return True
        except Exception as e:
            print(f"Failed to start pipeline: {e}")
            return False
    
    def stop(self) -> None:
        """ストリーミング停止"""
        if self.pipeline and self.is_started:
            self.pipeline.stop()
            self.is_started = False
            print("Pipeline stopped")
    
    def get_frame(self, timeout_ms: int = 100) -> Optional[FrameData]:
        """
        フレーム取得
        
        Args:
            timeout_ms: タイムアウト時間（ms）
            
        Returns:
            フレームデータまたはNone
        """
        if not self.is_started:
            return None
            
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                return None
            
            frame_data = FrameData(
                depth_frame=frames.get_depth_frame(),
                color_frame=frames.get_color_frame() if self.has_color else None,
                timestamp_ms=time.perf_counter() * 1000,
                frame_number=self.frame_count
            )
            
            self.frame_count += 1
            return frame_data
            
        except Exception as e:
            print(f"Frame acquisition error: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            "frame_count": self.frame_count,
            "has_color": self.has_color,
            "is_started": self.is_started,
            "depth_intrinsics": self.depth_intrinsics,
            "color_intrinsics": self.color_intrinsics
        }
    
    def __enter__(self):
        """コンテキストマネージャー: 開始"""
        if not self.initialize():
            raise RuntimeError("Failed to initialize camera")
        if not self.start():
            raise RuntimeError("Failed to start camera")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 終了"""
        self.stop() 