#!/usr/bin/env python3
"""
Orbbec カメラ抽象化クラス
既存のpoint_cloud_realtime_viewer.pyからフレーム取得処理を切り出し
GPU リサイズ最適化対応
"""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2

# 他フェーズとの連携  
from ..data_types import FrameData, CameraIntrinsics

# ロギング設定
from src import get_logger
from ..resource_manager import ManagedResource, get_resource_manager
logger = get_logger(__name__)

# CUDA利用可能性チェック（depth_filter.py と同様）
try:
    cv2.cuda.getCudaEnabledDeviceCount()
    HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if HAS_CUDA:
        logger.info(f"CUDA resize available: {cv2.cuda.getCudaEnabledDeviceCount()} devices")
except (cv2.error, AttributeError):
    HAS_CUDA = False
    logger.info("CUDA resize not available, using CPU fallback")

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


class FastResize:
    """高速リサイズ処理クラス（GPU/ROI最適化）"""
    
    def __init__(self, use_cuda: bool = True, enable_roi: bool = True):
        """
        初期化
        
        Args:
            use_cuda: CUDA リサイズを使用するか
            enable_roi: ROI クロッピングを有効にするか
        """
        self.use_cuda = use_cuda and HAS_CUDA
        self.enable_roi = enable_roi
        self.gpu_mat_cache = {}
        self.stream = cv2.cuda.Stream() if self.use_cuda else None
        self.initialized = False
        
        logger.info(f"FastResize initialized: CUDA={self.use_cuda}, ROI={enable_roi}")
    
    def initialize(self, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> bool:
        """
        リサイズ用GPU行列を初期化
        
        Args:
            src_shape: (height, width) 入力画像サイズ
            dst_shape: (height, width) 出力画像サイズ
            
        Returns:
            初期化成功時True
        """
        if not self.use_cuda:
            return True
            
        try:
            src_h, src_w = src_shape
            dst_h, dst_w = dst_shape
            
            # GPU行列をキャッシュ
            self.gpu_mat_cache['src_uint16'] = cv2.cuda.GpuMat(src_h, src_w, cv2.CV_16U)
            self.gpu_mat_cache['dst_uint16'] = cv2.cuda.GpuMat(dst_h, dst_w, cv2.CV_16U)
            self.gpu_mat_cache['src_uint8'] = cv2.cuda.GpuMat(src_h, src_w, cv2.CV_8UC3)
            self.gpu_mat_cache['dst_uint8'] = cv2.cuda.GpuMat(dst_h, dst_w, cv2.CV_8UC3)
            
            self.initialized = True
            logger.debug(f"GPU resize matrices initialized: {src_w}x{src_h} → {dst_w}x{dst_h}")
            return True
            
        except Exception as e:
            logger.warning(f"GPU resize initialization failed: {e}")
            return False
    
    def resize_depth(self, depth_image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        深度画像リサイズ（GPU/ROI最適化）
        
        Args:
            depth_image: 深度画像 (H, W) uint16
            target_size: (width, height) 目標サイズ
            
        Returns:
            リサイズ済み深度画像
        """
        target_w, target_h = target_size
        
        # ROI クロッピング最適化（1/2以上のダウンサンプリング時）
        if self.enable_roi and (target_w * 2 <= depth_image.shape[1] or target_h * 2 <= depth_image.shape[0]):
            # 中央領域をクロップして処理負荷軽減
            src_h, src_w = depth_image.shape
            crop_w = min(src_w, target_w * 2)
            crop_h = min(src_h, target_h * 2)
            
            start_x = (src_w - crop_w) // 2
            start_y = (src_h - crop_h) // 2
            
            cropped = depth_image[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            # クロップ後にリサイズ
            if self.use_cuda and self.initialized:
                return self._resize_cuda_uint16(cropped, (target_w, target_h))
            else:
                return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            # 通常リサイズ
            if self.use_cuda and self.initialized:
                return self._resize_cuda_uint16(depth_image, (target_w, target_h))
            else:
                return cv2.resize(depth_image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    def resize_color(self, color_image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        カラー画像リサイズ（GPU最適化）
        
        Args:
            color_image: カラー画像 (H, W, 3) uint8
            target_size: (width, height) 目標サイズ
            
        Returns:
            リサイズ済みカラー画像
        """
        target_w, target_h = target_size
        
        if self.use_cuda and self.initialized:
            return self._resize_cuda_uint8(color_image, (target_w, target_h))
        else:
            return cv2.resize(color_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    def _resize_cuda_uint16(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """CUDA uint16 リサイズ"""
        try:
            target_w, target_h = target_size
            
            # CPU → GPU転送
            gpu_src = self.gpu_mat_cache.get('src_uint16')
            gpu_dst = self.gpu_mat_cache.get('dst_uint16')
            
            if gpu_src is None or gpu_dst is None:
                raise RuntimeError("GPU matrices not initialized")
            
            # サイズが合わない場合は再確保
            if gpu_src.rows != image.shape[0] or gpu_src.cols != image.shape[1]:
                gpu_src = cv2.cuda.GpuMat(image.shape[0], image.shape[1], cv2.CV_16U)
            if gpu_dst.rows != target_h or gpu_dst.cols != target_w:
                gpu_dst = cv2.cuda.GpuMat(target_h, target_w, cv2.CV_16U)
            
            gpu_src.upload(image)
            
            # CUDA リサイズ実行
            cv2.cuda.resize(gpu_src, gpu_dst, (target_w, target_h), interpolation=cv2.INTER_NEAREST, stream=self.stream)
            
            # GPU → CPU転送
            result = gpu_dst.download()
            
            if self.stream:
                self.stream.waitForCompletion()
                
            return result
            
        except Exception as e:
            logger.warning(f"CUDA uint16 resize error: {e}")
            return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    
    def _resize_cuda_uint8(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """CUDA uint8 リサイズ"""
        try:
            target_w, target_h = target_size
            
            gpu_src = self.gpu_mat_cache.get('src_uint8')
            gpu_dst = self.gpu_mat_cache.get('dst_uint8')
            
            if gpu_src is None or gpu_dst is None:
                raise RuntimeError("GPU matrices not initialized")
            
            # サイズチェック
            if gpu_src.rows != image.shape[0] or gpu_src.cols != image.shape[1]:
                gpu_src = cv2.cuda.GpuMat(image.shape[0], image.shape[1], cv2.CV_8UC3)
            if gpu_dst.rows != target_h or gpu_dst.cols != target_w:
                gpu_dst = cv2.cuda.GpuMat(target_h, target_w, cv2.CV_8UC3)
            
            gpu_src.upload(image)
            
            cv2.cuda.resize(gpu_src, gpu_dst, (target_w, target_h), interpolation=cv2.INTER_LINEAR, stream=self.stream)
            
            result = gpu_dst.download()
            
            if self.stream:
                self.stream.waitForCompletion()
                
            return result
            
        except Exception as e:
            logger.warning(f"CUDA uint8 resize error: {e}")
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


# CameraIntrinsics と FrameData は src/types.py で定義済み


class OrbbecCamera(ManagedResource):
    """Orbbec カメラ抽象化クラス（リソース管理・高速リサイズ対応）"""
    
    def __init__(self, enable_color: bool = True, resource_id: Optional[str] = None,
                 depth_width: Optional[int] = None, depth_height: Optional[int] = None,
                 use_fast_resize: bool = True):
        """
        Orbbec カメラ初期化
        
        Args:
            enable_color: カラーストリームを有効にするか
            resource_id: リソース識別子
            depth_width: 深度ストリーム幅（Noneの場合デフォルト）
            depth_height: 深度ストリーム高さ（Noneの場合デフォルト）
            use_fast_resize: 高速リサイズを使用するか
        """
        super().__init__(resource_id or f"orbbec_camera_{int(time.time())}")
        self.enable_color = enable_color
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.use_fast_resize = use_fast_resize
        self.pipeline: Optional[Pipeline] = None
        self.config = None
        self.depth_intrinsics: Optional[CameraIntrinsics] = None
        self.color_intrinsics: Optional[CameraIntrinsics] = None
        self.has_color = False
        self.is_started = False
        self.frame_count = 0
        
        # 高速リサイズ処理
        self.fast_resize = FastResize() if use_fast_resize else None
        
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
                try:
                    # OrbbecSDKの正しいAPIを使用
                    if hasattr(depth_profile_list, 'get_stream_profile_by_index'):
                        profile = depth_profile_list.get_stream_profile_by_index(i)
                    elif hasattr(depth_profile_list, 'get_profile'):
                        profile = depth_profile_list.get_profile(i)
                    elif hasattr(depth_profile_list, '__getitem__'):
                        profile = depth_profile_list[i]
                    else:
                        logger.error(f"Unknown StreamProfileList API methods: {dir(depth_profile_list)}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to get profile {i}: {e}")
                    continue
                    
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
            try:
                if hasattr(depth_profile_list, 'get_default_video_stream_profile'):
                    depth_profile = depth_profile_list.get_default_video_stream_profile()
                elif hasattr(depth_profile_list, 'get_video_stream_profile'):
                    depth_profile = depth_profile_list.get_video_stream_profile(0)  # 最初のプロファイル
                else:
                    logger.error(f"No method to get default profile. Available methods: {dir(depth_profile_list)}")
                    return False
                
                logger.info(f"Using default depth profile: {depth_profile.get_width()}x{depth_profile.get_height()}")
            except Exception as e:
                logger.error(f"Failed to get default depth profile: {e}")
                return False
            
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
                try:
                    if hasattr(color_profile_list, 'get_default_video_stream_profile'):
                        color_profile = color_profile_list.get_default_video_stream_profile()
                    elif hasattr(color_profile_list, 'get_stream_profile_by_index'):
                        color_profile = color_profile_list.get_stream_profile_by_index(0)
                    else:
                        logger.warning(f"No method to get color profile. Available methods: {dir(color_profile_list)}")
                        self.has_color = False
                        return
                        
                    self.config.enable_stream(color_profile)
                    self.has_color = True
                except Exception as e:
                    logger.warning(f"Failed to get color profile: {e}")
                    self.has_color = False
                    return
                
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