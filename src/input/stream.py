#!/usr/bin/env python3
"""
Orbbec カメラ抽象化クラス
既存のpoint_cloud_realtime_viewer.pyからフレーム取得処理を切り出し
"""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

# 他フェーズとの連携  
from ..types import FrameData, CameraIntrinsics

# ロギング設定
from src import get_logger
from ..resource_manager import ManagedResource, get_resource_manager
logger = get_logger(__name__)

try:
    # vendor配下から直接import
    import sys
    import os
    vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vendor', 'pyorbbecsdk')
    sys.path.insert(0, vendor_path)
    from pyorbbecsdk import Pipeline, Config, OBSensorType, FrameSet, OBError
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
    
    class FrameSet:
        def __init__(self): pass
        def get_depth_frame(self): return None
        def get_color_frame(self): return None
    
    class OBError(Exception):
        pass


# CameraIntrinsics と FrameData は src/types.py で定義済み


class OrbbecCamera(ManagedResource):
    """Orbbec カメラ抽象化クラス（リソース管理対応）"""
    
    def __init__(self, enable_color: bool = True, resource_id: Optional[str] = None,
                 depth_width: Optional[int] = None, depth_height: Optional[int] = None):
        """
        Orbbec カメラ初期化
        
        Args:
            enable_color: カラーストリームを有効にするか
            resource_id: リソース識別子
            depth_width: 深度ストリーム幅（Noneの場合デフォルト）
            depth_height: 深度ストリーム高さ（Noneの場合デフォルト）
        """
        super().__init__(resource_id or f"orbbec_camera_{int(time.time())}")
        self.enable_color = enable_color
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.pipeline: Optional[Pipeline] = None
        self.config = None
        self.depth_intrinsics: Optional[CameraIntrinsics] = None
        self.color_intrinsics: Optional[CameraIntrinsics] = None
        self.has_color = False
        self.is_started = False
        self.frame_count = 0
        
        # リソースマネージャーに自動登録
        manager = get_resource_manager()
        manager.register_resource(self, memory_estimate=100 * 1024 * 1024)  # 100MB推定
        
    def initialize(self) -> bool:
        """
        カメラを初期化
        
        Returns:
            成功した場合True
            
        Raises:
            RuntimeError: 致命的な初期化エラー
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
            
        except OBError as e:
            # OrbbecSDK固有のエラー
            logger.error(f"Orbbec SDK error during initialization: {e}")
            return False
        except (OSError, IOError) as e:
            # システムリソースエラー（致命的）
            logger.error(f"System resource error during camera initialization: {e}")
            raise RuntimeError(f"Camera hardware access failed: {e}")
        except Exception as e:
            # その他の予期しないエラー（致命的）
            logger.error(f"Unexpected camera initialization error: {e}")
            raise RuntimeError(f"Camera initialization failed: {e}") from e
    
    def _setup_depth_stream(self) -> bool:
        """深度ストリームを設定"""
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            logger.error("No depth sensor found!")
            return False
        
        # 希望する解像度が指定されている場合は、それに近いプロファイルを探す
        depth_profile = None
        if self.depth_width is not None and self.depth_height is not None:
            logger.info(f"🔍 Searching for depth profile: {self.depth_width}x{self.depth_height}")
            
            # 指定解像度に近いプロファイルを検索
            best_profile = None
            min_diff = float('inf')
            available_profiles = []
            
            profile_count = depth_profile_list.get_count()
            logger.info(f"🔍 Available depth profiles: {profile_count}")
            
            for i in range(profile_count):
                profile = depth_profile_list.get_profile(i)
                if hasattr(profile, 'get_width') and hasattr(profile, 'get_height'):
                    width = profile.get_width()
                    height = profile.get_height()
                    available_profiles.append(f"{width}x{height}")
                    
                    # 解像度の差を計算
                    diff = abs(width - self.depth_width) + abs(height - self.depth_height)
                    logger.debug(f"  Profile {i}: {width}x{height}, diff={diff}")
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_profile = profile
            
            logger.info(f"🔍 Available profiles: {', '.join(available_profiles)}")
                        
            if best_profile is not None:
                depth_profile = best_profile
                actual_width = depth_profile.get_width()
                actual_height = depth_profile.get_height()
                logger.info(f"✅ Selected depth profile: {actual_width}x{actual_height} "
                          f"(requested: {self.depth_width}x{self.depth_height}, diff={min_diff})")
                
                # 解像度が大きく異なる場合は警告
                if min_diff > 100:  # 100ピクセル以上の差
                    logger.warning(f"⚠️  RESOLUTION MISMATCH: Requested {self.depth_width}x{self.depth_height} "
                                 f"but using {actual_width}x{actual_height}")
                    logger.warning(f"⚠️  This may impact performance significantly!")
            else:
                logger.warning(f"❌ No depth profile found for {self.depth_width}x{self.depth_height}")
        
        # 指定がない場合または見つからない場合はデフォルトを使用
        if depth_profile is None:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            logger.info(f"Using default depth profile: {depth_profile.get_width()}x{depth_profile.get_height()}")
            
            # デフォルト使用時の警告（低解像度モード指定時）
            if self.depth_width is not None and self.depth_height is not None:
                logger.error(f"🚨 CRITICAL: Could not apply low resolution {self.depth_width}x{self.depth_height}!")
                logger.error(f"🚨 Performance will be significantly impacted!")
        
        self.config.enable_stream(depth_profile)
        
        final_width = depth_profile.get_width()
        final_height = depth_profile.get_height()
        logger.info(f"Depth: {final_width}x{final_height}@{depth_profile.get_fps()}fps")
        
        # 最終確認ログ
        if self.depth_width is not None and self.depth_height is not None:
            if final_width == self.depth_width and final_height == self.depth_height:
                logger.info(f"✅ Resolution optimization successful: {final_width}x{final_height}")
            else:
                logger.error(f"❌ Resolution optimization FAILED: wanted {self.depth_width}x{self.depth_height}, "
                           f"got {final_width}x{final_height}")
        
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
            logger.info(f"Depth intrinsics: fx={depth_intrinsic.fx}, fy={depth_intrinsic.fy}, "
                  f"cx={depth_intrinsic.cx}, cy={depth_intrinsic.cy}")
        except (AttributeError, TypeError, ValueError) as e:
            # デフォルト値使用
            logger.warning(f"Failed to get depth intrinsics, using defaults: {e}")
            logger.info("Using default depth intrinsics")
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
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to get color intrinsics: {e}")
                    self.color_intrinsics = None
                
                logger.info(f"Color: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
        except (AttributeError, RuntimeError) as e:
            logger.info(f"Color sensor setup failed (expected if no color sensor): {e}")
            logger.info("No color sensor available")
            self.has_color = False
    
    def start(self) -> bool:
        """
        ストリーミング開始
        
        Returns:
            成功した場合True
            
        Raises:
            RuntimeError: 致命的な開始エラー
        """
        if not self.pipeline or not self.config:
            logger.error("Camera not initialized")
            return False
            
        try:
            self.pipeline.start(self.config)
            self.is_started = True
            logger.info("Pipeline started!")
            return True
        except OBError as e:
            # OrbbecSDK固有のエラー（復旧可能）
            logger.error(f"Orbbec SDK error during start: {e}")
            return False
        except (OSError, IOError) as e:
            # システムリソースエラー（致命的）
            logger.error(f"System resource error during pipeline start: {e}")
            raise RuntimeError(f"Camera pipeline start failed: {e}")
        except Exception as e:
            # その他の予期しないエラー（致命的）
            logger.error(f"Unexpected pipeline start error: {e}")
            raise RuntimeError(f"Pipeline start failed: {e}") from e
    
    def cleanup(self) -> bool:
        """リソースクリーンアップ（ManagedResourceインターフェース）"""
        try:
            self.stop()
            if self.pipeline:
                # パイプラインの完全なクリーンアップ
                try:
                    del self.pipeline
                    self.pipeline = None
                except (AttributeError, RuntimeError) as e:
                    # パイプライン削除エラー（警告レベル）
                    logger.warning(f"Error cleaning up pipeline: {e}")
                except Exception as e:
                    # 予期しないエラー
                    logger.error(f"Unexpected error cleaning up pipeline: {e}")
            
            if self.config:
                try:
                    del self.config
                    self.config = None
                except (AttributeError, RuntimeError) as e:
                    # 設定削除エラー（警告レベル）
                    logger.warning(f"Error cleaning up config: {e}")
                except Exception as e:
                    # 予期しないエラー
                    logger.error(f"Unexpected error cleaning up config: {e}")
            
            self.depth_intrinsics = None
            self.color_intrinsics = None
            self.has_color = False
            self.is_started = False
            
            logger.info(f"Camera resource cleaned up: {self.resource_id}")
            return True
        except Exception as e:
            logger.error(f"Error in camera cleanup: {e}")
            return False
    
    def stop(self) -> None:
        """ストリーミング停止"""
        if self.pipeline and self.is_started:
            try:
                self.pipeline.stop()
                self.is_started = False
                logger.info("Pipeline stopped")
            except (OBError, RuntimeError) as e:
                # パイプライン停止エラー（警告レベル）
                logger.warning(f"Error stopping pipeline: {e}")
                self.is_started = False
            except Exception as e:
                # 予期しないエラー
                logger.error(f"Unexpected error stopping pipeline: {e}")
                self.is_started = False
    
    def get_frame(self, timeout_ms: int = 100) -> Optional[FrameData]:
        """
        フレーム取得
        
        Args:
            timeout_ms: タイムアウト時間（ms）
            
        Returns:
            フレームデータまたはNone
            
        Raises:
            RuntimeError: 致命的なフレーム取得エラー
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
            
        except OBError as e:
            # OrbbecSDK固有のエラー（タイムアウト等、復旧可能）
            logger.debug(f"Orbbec SDK frame acquisition error: {e}")
            return None
        except (OSError, IOError, MemoryError) as e:
            # システムリソースエラー（致命的）
            logger.error(f"System resource error during frame acquisition: {e}")
            raise RuntimeError(f"Frame acquisition failed: {e}")
        except Exception as e:
            # その他の予期しないエラー（警告してNoneを返す）
            logger.warning(f"Unexpected frame acquisition error: {e}")
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
    
    @property
    def resource_type(self) -> str:
        """リソースタイプ（ManagedResourceインターフェース）"""
        return "orbbec_camera"
    
    def get_memory_usage(self) -> int:
        """メモリ使用量を取得（ManagedResourceインターフェース）"""
        # 概算値の計算
        base_memory = 50 * 1024 * 1024  # 50MB基本
        if self.has_color:
            base_memory += 30 * 1024 * 1024  # カラーで+30MB
        frame_buffer_memory = self.frame_count * 1024  # 1KB/フレーム
        return base_memory + frame_buffer_memory 