#!/usr/bin/env python3
"""
OAK-D カメラ抽象化クラス
OrbbecCameraと同じシンプルなインターフェースで実装
"""

import time
from typing import Optional, Dict, Any
import numpy as np

# 他フェーズとの連携  
from ..data_types import FrameData, CameraIntrinsics

# ロギング設定
from src import get_logger
from ..resource_manager import ManagedResource, get_resource_manager

logger = get_logger(__name__)

# DepthAI インポート
try:
    import depthai as dai
    HAS_DEPTHAI = True
    logger.info("DepthAI import successful")
except ImportError:
    HAS_DEPTHAI = False
    logger.warning("DepthAI not available, OAK-D will not function")


class OakCamera(ManagedResource):
    """OAK-D カメラ抽象化クラス（OrbbecCamera互換インターフェース）"""
    
    def __init__(self, enable_color: bool = True, resource_id: Optional[str] = None,
                 depth_width: Optional[int] = None, depth_height: Optional[int] = None,
                 use_fast_resize: bool = True):
        """
        OAK-D カメラ初期化
        
        Args:
            enable_color: カラーストリームを有効にするか
            resource_id: リソース識別子
            depth_width: 深度ストリーム幅（互換性のため、無視される）
            depth_height: 深度ストリーム高さ（互換性のため、無視される）
            use_fast_resize: 高速リサイズを使用するか（互換性のため、無視される）
        """
        super().__init__(resource_id or f"oak_camera_{int(time.time())}")
        self.enable_color = enable_color
        
        # OAK-D関連
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None
        self.q_rgb: Optional[Any] = None  # RGB キュー
        self.q_depth: Optional[Any] = None  # 深度キュー
        
        # カメラパラメータ
        self.depth_intrinsics: Optional[CameraIntrinsics] = None
        self.color_intrinsics: Optional[CameraIntrinsics] = None
        self.has_color = enable_color
        self.is_started = False
        self.frame_count = 0
        
        # デバッグ統計（簡素化）
        self.successful_frames = 0
        self.failed_frames = 0
        self.startup_time = None
        
        # リソースマネージャーに自動登録
        manager = get_resource_manager()
        manager.register_resource(self, memory_estimate=100 * 1024 * 1024)
        
        logger.info(f"OAK-D カメラ初期化: enable_color={enable_color}")
        
    def initialize(self) -> bool:
        """
        カメラを初期化
        
        Returns:
            成功した場合True
        """
        if not HAS_DEPTHAI:
            logger.error("DepthAI not available, cannot initialize OAK-D")
            return False
            
        try:
            # パイプライン作成
            self.pipeline = dai.Pipeline()
            
            # RGBカメラの設定（シンプルな設定）
            rgb_cam = self.pipeline.create(dai.node.ColorCamera)
            rgb_cam.setPreviewSize(640, 480)  # 手検出に十分な解像度
            rgb_cam.setInterleaved(False)
            rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            rgb_cam.setFps(15)  # フレームレートを下げて安定化
            
            # RGB出力
            rgb_out = self.pipeline.create(dai.node.XLinkOut)
            rgb_out.setStreamName("rgb")
            rgb_cam.preview.link(rgb_out.input)
            
            # 深度カメラの設定（シンプルな設定）
            mono_left = self.pipeline.create(dai.node.MonoCamera)
            mono_right = self.pipeline.create(dai.node.MonoCamera)
            depth = self.pipeline.create(dai.node.StereoDepth)
            
            # 低解像度設定でパフォーマンスを向上
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            
            # シンプルな深度設定
            depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            depth.setLeftRightCheck(False)  # より安定した動作のためオフ
            depth.setSubpixel(False)  # より安定した動作のためオフ
            
            mono_left.out.link(depth.left)
            mono_right.out.link(depth.right)
            
            depth_out = self.pipeline.create(dai.node.XLinkOut)
            depth_out.setStreamName("depth")
            depth.depth.link(depth_out.input)
            
            logger.info("OAK-D パイプライン作成完了")
            return True
            
        except Exception as e:
            logger.error(f"OAK-D 初期化エラー: {e}")
            return False
    
    def start(self) -> bool:
        """
        ストリーミング開始
        
        Returns:
            成功した場合True
        """
        if not self.pipeline:
            logger.error("Pipeline not initialized")
            return False
            
        try:
            # デバイス接続
            logger.info("OAK-D デバイスに接続中...")
            self.device = dai.Device(self.pipeline)
            logger.info(f"OAK-D デバイス接続完了: {self.device.getDeviceName()}")
            
            # キューの設定（非ブロッキング、小さなバッファ）
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            # カメラ内部パラメータを設定
            self._setup_intrinsics()
            
            # 短いウォームアップ期間
            logger.info("OAK-D ウォームアップ中...")
            time.sleep(2.0)  # 2秒間のウォームアップ
            
            self.startup_time = time.time()
            self.is_started = True
            logger.info("OAK-D ストリーミング開始完了")
            return True
            
        except Exception as e:
            logger.error(f"OAK-D 開始エラー: {e}")
            return False
    
    def _setup_intrinsics(self) -> None:
        """カメラ内部パラメータを設定"""
        try:
            # OAK-D用のデフォルト内部パラメータ（RGB 640x480用）
            self.color_intrinsics = CameraIntrinsics(
                fx=640.0,  # 焦点距離 x
                fy=640.0,  # 焦点距離 y
                cx=320.0,  # 画像中心 x
                cy=240.0,  # 画像中心 y
                width=640,
                height=480
            )
            
            # 深度カメラの内部パラメータ（720x1280 -> RGB座標系に合わせて調整）
            self.depth_intrinsics = CameraIntrinsics(
                fx=640.0,  # RGB と同じ焦点距離に調整
                fy=640.0,
                cx=320.0,  # RGB と同じ中心点に調整
                cy=240.0,
                width=640,  # RGB解像度に合わせる
                height=480
            )
            
            logger.info(f"OAK-D 内部パラメータ設定完了: RGB={self.color_intrinsics.width}x{self.color_intrinsics.height}")
            
        except Exception as e:
            logger.error(f"OAK-D 内部パラメータ設定エラー: {e}")
    
    def get_frame(self, timeout_ms: int = 100) -> Optional[FrameData]:
        """
        フレーム取得（シンプル化）
        
        Args:
            timeout_ms: タイムアウト時間（ms）（互換性のため、実際は無視される）
            
        Returns:
            フレームデータまたはNone
        """
        if not self.is_started or not self.q_rgb:
            return None
            
        try:
            # RGBフレーム取得（シンプル化）
            rgb_frame = None
            depth_frame = None
            
            # RGB取得
            if self.q_rgb:
                in_rgb = self.q_rgb.tryGet()
                if in_rgb is not None:
                    rgb_frame = in_rgb.getCvFrame()
                
            # 深度取得
            if self.q_depth:
                in_depth = self.q_depth.tryGet()
                if in_depth is not None:
                    depth_frame = in_depth.getFrame()
            
            # フレームが取得できない場合
            if rgb_frame is None and depth_frame is None:
                self.failed_frames += 1
                # 失敗ログを大幅に減らす
                if self.failed_frames % 1000 == 0:
                    logger.debug(f"OAK-D フレーム取得失敗: {self.failed_frames} 回")
                return None
            
            # 成功統計更新
            self.successful_frames += 1
            
            # Orbbec互換のフレームラッパーを作成
            depth_frame_wrapped = OakFrameWrapper(depth_frame) if depth_frame is not None else None
            color_frame_wrapped = OakFrameWrapper(rgb_frame) if rgb_frame is not None else None
            
            frame_data = FrameData(
                depth_frame=depth_frame_wrapped,
                color_frame=color_frame_wrapped,
                timestamp_ms=time.perf_counter() * 1000,
                frame_number=self.frame_count
            )
            
            self.frame_count += 1
            
            # 成功率ログを減らす
            if self.frame_count % 300 == 0:
                success_rate = self.successful_frames / (self.successful_frames + self.failed_frames) * 100
                logger.info(f"OAK-D 統計: 成功率 {success_rate:.1f}%, フレーム数 {self.frame_count}")
            
            return frame_data
            
        except Exception as e:
            self.failed_frames += 1
            if self.failed_frames % 500 == 0:  # エラーログも減らす
                logger.warning(f"OAK-D フレーム取得エラー: {e}")
            return None


    def cleanup(self) -> bool:
        """リソースクリーンアップ"""
        try:
            self.stop()
            
            if self.device:
                try:
                    self.device.close()
                except:
                    pass
                self.device = None
            
            self.pipeline = None
            self.q_rgb = None
            self.q_depth = None
            self.is_started = False
            
            # 最終統計ログ
            if self.successful_frames + self.failed_frames > 0:
                success_rate = self.successful_frames / (self.successful_frames + self.failed_frames) * 100
                logger.info(f"OAK-D 最終統計: 成功率 {success_rate:.1f}%, 総フレーム数 {self.frame_count}")
            
            logger.info(f"OAK-D リソースクリーンアップ完了: {self.resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"OAK-D クリーンアップエラー: {e}")
            return False
    
    def stop(self) -> None:
        """ストリーミング停止"""
        if self.is_started:
            self.is_started = False
            logger.info("OAK-D ストリーミング停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        success_rate = 0.0
        if self.successful_frames + self.failed_frames > 0:
            success_rate = self.successful_frames / (self.successful_frames + self.failed_frames) * 100
            
        return {
            "frame_count": self.frame_count,
            "has_color": self.has_color,
            "is_started": self.is_started,
            "successful_frames": self.successful_frames,
            "failed_frames": self.failed_frames,
            "success_rate": success_rate,
            "uptime_seconds": time.time() - self.startup_time if self.startup_time else 0,
        }
    
    @property
    def resource_type(self) -> str:
        """リソースタイプ"""
        return "oak_camera"
    
    def get_memory_usage(self) -> int:
        """メモリ使用量を取得"""
        base_memory = 50 * 1024 * 1024  # 50MB基本
        if self.has_color:
            base_memory += 20 * 1024 * 1024  # カラーで+20MB
        return base_memory
    
    def __enter__(self):
        """コンテキストマネージャー: 開始"""
        if not self.initialize():
            raise RuntimeError("Failed to initialize OAK-D")
        if not self.start():
            raise RuntimeError("Failed to start OAK-D")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 終了"""
        self.cleanup()


class OakFrameWrapper:
    """OAK-D フレームをOrbbec互換インターフェースでラップ"""
    
    def __init__(self, frame_data: np.ndarray):
        if frame_data is None:
            raise ValueError("Frame data cannot be None")
            
        # 深度フレームのリサイズ処理（様々なサイズに対応）
        if len(frame_data.shape) == 2:
            import cv2
            original_shape = frame_data.shape
            # アプリが期待するサイズ（480, 640）にリサイズ
            if frame_data.shape != (480, 640):
                frame_data = cv2.resize(frame_data, (640, 480), interpolation=cv2.INTER_NEAREST)
                logger.debug(f"深度フレームリサイズ: {original_shape} -> (480, 640)")
        
        self.frame_data = frame_data
    
    def get_data(self) -> bytes:
        """フレームデータをバイト列で取得（Orbbec互換）"""
        return self.frame_data.tobytes()
    
    def get_format(self):
        """フレームフォーマットを取得（Orbbec互換）"""
        # カラーフレームの場合はBGR、深度フレームの場合はDEPTH
        if len(self.frame_data.shape) == 3:
            # カラーフレーム
            from ..data_types import OBFormat
            return OBFormat.BGR
        else:
            # 深度フレーム（文字列で返す）
            return "DEPTH"
    
    def __getattr__(self, name):
        """その他の属性は元のnumpy.ndarrayに委譲"""
        return getattr(self.frame_data, name) 