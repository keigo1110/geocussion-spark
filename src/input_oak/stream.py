#!/usr/bin/env python3
"""
OAK-D S2 カメラ抽象化クラス
既存のOrbbecCameraインターフェースを踏襲しながら、DepthAI APIを使用
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

try:
    import depthai as dai
    HAS_DEPTHAI = True
    logger.info("DepthAI import successful")
except ImportError:
    HAS_DEPTHAI = False
    logger.warning("DepthAI not available, OAK-D S2 will not function")
    
    # テスト用モッククラス
    class dai:
        class Pipeline:
            def __init__(self): pass
            def create(self, node_type): return MockNode()
        
        class Device:
            def __init__(self, pipeline): pass
            def getOutputQueue(self, name): return MockQueue()
            def readCalibration(self): return MockCalibration()
        
        class CameraBoardSocket:
            LEFT = "left"
            RIGHT = "right"
            RGB = "rgb"
        
        class MonoCameraProperties:
            class SensorResolution:
                THE_720_P = "720p"
        
        class ColorCameraProperties:
            class SensorResolution:
                THE_1080_P = "1080p"
        
        class node:
            class MonoCamera:
                def setBoardSocket(self, socket): pass
                def setResolution(self, res): pass
                def setFps(self, fps): pass
                @property
                def out(self): return MockOutput()
            
            class ColorCamera:
                def setResolution(self, res): pass
                def setFps(self, fps): pass
                @property
                def video(self): return MockOutput()
            
            class StereoDepth:
                def setDefaultProfilePreset(self, preset): pass
                def setDepthAlign(self, socket): pass
                def setLeftRightCheck(self, check): pass
                def setSubpixel(self, sub): pass
                @property
                def depth(self): return MockOutput()
                
                class PresetMode:
                    HIGH_DENSITY = "high_density"
            
            class XLinkOut:
                def setStreamName(self, name): pass
                @property
                def input(self): return MockInput()
    
    class MockNode:
        def setBoardSocket(self, socket): pass
        def setResolution(self, res): pass
        def setFps(self, fps): pass
        def setDefaultProfilePreset(self, preset): pass
        def setDepthAlign(self, socket): pass
        def setLeftRightCheck(self, check): pass
        def setSubpixel(self, sub): pass
        def setStreamName(self, name): pass
        
        @property
        def out(self): return MockOutput()
        @property
        def video(self): return MockOutput()
        @property
        def depth(self): return MockOutput()
        @property
        def input(self): return MockInput()
    
    class MockOutput:
        def link(self, input_port): pass
    
    class MockInput:
        pass
    
    class MockQueue:
        def get(self): return MockFrame()
        def tryGet(self): return None
    
    class MockFrame:
        def getFrame(self): return np.zeros((240, 424), dtype=np.uint16)
        def getCvFrame(self): 
            # よりリアルなカラーフレームを生成（グラデーション）
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 青から赤へのグラデーション
            frame[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)  # Blue
            frame[:, :, 1] = np.linspace(0, 128, 480).reshape(-1, 1).astype(np.uint8)  # Green
            frame[:, :, 2] = np.linspace(255, 0, 640).astype(np.uint8)  # Red
            return frame
        def get_data(self): return np.zeros((240, 424), dtype=np.uint16).tobytes()
    
    class MockCalibration:
        def getCameraIntrinsics(self, socket, width, height):
            return [width, width, width/2, height/2, 0, 0, 0, 0]


class OakCamera(ManagedResource):
    """OAK-D S2 カメラ抽象化クラス（OrbbecCamera互換インターフェース）"""
    
    def __init__(self, enable_color: bool = True, resource_id: Optional[str] = None,
                 depth_width: Optional[int] = None, depth_height: Optional[int] = None,
                 use_fast_resize: bool = True, downsample_factor: int = 3):
        """
        OAK-D S2 カメラ初期化
        
        Args:
            enable_color: カラーストリームを有効にするか
            resource_id: リソース識別子
            depth_width: 深度ストリーム幅（OAK-D S2では固定）
            depth_height: 深度ストリーム高さ（OAK-D S2では固定）
            use_fast_resize: 高速リサイズを使用するか（OAK-D S2では不要）
            downsample_factor: ダウンサンプリング係数（Orbbec互換性のため）
        """
        super().__init__(resource_id or f"oak_camera_{int(time.time())}")
        self.enable_color = enable_color
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.use_fast_resize = use_fast_resize
        self.downsample_factor = downsample_factor
        
        # DepthAI関連
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None
        self.q_depth: Optional[dai.DataOutputQueue] = None
        self.q_rgb: Optional[dai.DataOutputQueue] = None
        
        # カメラパラメータ
        self.depth_intrinsics: Optional[CameraIntrinsics] = None
        self.color_intrinsics: Optional[CameraIntrinsics] = None
        self.has_color = False
        self.is_started = False
        self.frame_count = 0
        
        # OAK-D S2固有の設定
        self.oak_depth_width = 1280 // downsample_factor
        self.oak_depth_height = 720 // downsample_factor
        
        # リソースマネージャーに自動登録
        manager = get_resource_manager()
        manager.register_resource(self, memory_estimate=200 * 1024 * 1024)  # 200MB推定
        
        logger.info(f"OAK-D S2 カメラ初期化: enable_color={enable_color}, downsample={downsample_factor}x")
        logger.info(f"OAK-D S2 ダウンサンプリング後解像度: {self.oak_depth_width}x{self.oak_depth_height}")
        
    def initialize(self) -> bool:
        """
        カメラを初期化
        
        Returns:
            成功した場合True
            
        Raises:
            RuntimeError: 致命的な初期化エラー
        """
        if not HAS_DEPTHAI:
            logger.error("DepthAI not available, cannot initialize OAK-D S2")
            return False
            
        try:
            # パイプライン作成
            self.pipeline = dai.Pipeline()
            
            # 深度ストリーム設定
            if not self._setup_depth_stream():
                return False
                
            # カラーストリーム設定（オプション）
            if self.enable_color:
                if not self._setup_color_stream():
                    logger.warning("カラーストリーム設定に失敗しましたが、続行します")
                    self.enable_color = False
                    self.has_color = False
                else:
                    self.has_color = True
                    
            # デバイス接続
            self.device = dai.Device(self.pipeline)
            
            # 出力キューの設定
            self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
            if self.has_color:
                self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
            
            # カメラ内部パラメータの取得
            self._setup_intrinsics()
            
            logger.info("OAK-D S2 カメラ初期化が完了しました")
            return True
            
        except Exception as e:
            logger.error(f"OAK-D S2 初期化エラー: {e}")
            return False
    
    def _setup_depth_stream(self) -> bool:
        """深度ストリームを設定"""
        try:
            # ステレオカメラ設定
            mono_left = self.pipeline.create(dai.node.MonoCamera)
            mono_right = self.pipeline.create(dai.node.MonoCamera)
            
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            mono_left.setFps(60)
            
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            mono_right.setFps(60)
            
            # ステレオ深度設定
            stereo = self.pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # RGB座標系に合わせる
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            
            # 深度出力解像度を明示的に設定（APIが存在する場合のみ）
            try:
                # OAK-D S2の実際の深度出力を 1280x720 に固定
                if hasattr(stereo, 'setOutputSize'):
                    stereo.setOutputSize(1280, 720)
                    logger.info("深度出力サイズを1280x720に設定しました")
                else:
                    logger.info("setOutputSize メソッドが利用できません。デフォルト設定を使用します")
            except Exception as e:
                logger.warning(f"深度出力サイズ設定エラー: {e}")
            
            # 出力設定
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            
            # 接続
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)
            stereo.depth.link(xout_depth.input)
            
            logger.info("深度ストリーム設定が完了しました (1280x720)")
            return True
            
        except Exception as e:
            logger.error(f"深度ストリーム設定エラー: {e}")
            return False
    
    def _setup_color_stream(self) -> bool:
        """カラーストリームを設定"""
        try:
            # RGBカメラ設定
            rgb = self.pipeline.create(dai.node.ColorCamera)
            rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            rgb.setFps(60)
            
            # 出力設定
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            
            # 接続
            rgb.video.link(xout_rgb.input)
            
            logger.info("カラーストリーム設定が完了しました")
            return True
            
        except Exception as e:
            logger.error(f"カラーストリーム設定エラー: {e}")
            return False
    
    def _setup_intrinsics(self) -> None:
        """カメラ内部パラメータを設定"""
        try:
            if not self.device:
                return
                
            calib = self.device.readCalibration()
            
            # 深度カメラの内部パラメータ（ダウンサンプリング調整）
            try:
                depth_intrinsics = calib.getCameraIntrinsics(
                    dai.CameraBoardSocket.LEFT, 1280, 720
                )
                
                if depth_intrinsics and len(depth_intrinsics) >= 4:
                    # リスト型の場合は適切にアンパック
                    if isinstance(depth_intrinsics, list):
                        fx, fy, cx, cy = depth_intrinsics[:4]
                    else:
                        fx, fy, cx, cy = depth_intrinsics[:4]
                    
                    # ダウンサンプリングによる内部パラメータの調整
                    fx_adj = float(fx) / self.downsample_factor
                    fy_adj = float(fy) / self.downsample_factor
                    cx_adj = float(cx) / self.downsample_factor
                    cy_adj = float(cy) / self.downsample_factor
                    
                    self.depth_intrinsics = CameraIntrinsics(
                        fx=fx_adj,
                        fy=fy_adj,
                        cx=cx_adj,
                        cy=cy_adj,
                        width=self.oak_depth_width,
                        height=self.oak_depth_height
                    )
                    logger.info(f"深度内部パラメータ（ダウンサンプリング調整）: fx={fx_adj:.1f}, fy={fy_adj:.1f}, cx={cx_adj:.1f}, cy={cy_adj:.1f}")
                    logger.info(f"深度解像度: {self.oak_depth_width}x{self.oak_depth_height}")
                elif depth_intrinsics and len(depth_intrinsics) == 3:
                    # 3つの値しか返されない場合の処理
                    if isinstance(depth_intrinsics, list):
                        fx, fy, f_mean = depth_intrinsics[:3]
                    else:
                        fx, fy, f_mean = depth_intrinsics[:3]
                    cx, cy = 1280 / 2, 720 / 2  # 中央値を使用
                    
                    # ダウンサンプリングによる内部パラメータの調整
                    fx_adj = float(fx) / self.downsample_factor
                    fy_adj = float(fy) / self.downsample_factor
                    cx_adj = float(cx) / self.downsample_factor
                    cy_adj = float(cy) / self.downsample_factor
                    
                    self.depth_intrinsics = CameraIntrinsics(
                        fx=fx_adj,
                        fy=fy_adj,
                        cx=cx_adj,
                        cy=cy_adj,
                        width=self.oak_depth_width,
                        height=self.oak_depth_height
                    )
                    logger.info(f"深度内部パラメータ（3値・ダウンサンプリング調整）: fx={fx_adj:.1f}, fy={fy_adj:.1f}, cx={cx_adj:.1f}, cy={cy_adj:.1f}")
                else:
                    raise ValueError("内部パラメータの値が不足しています")
                    
            except Exception as e:
                logger.warning(f"深度内部パラメータ取得エラー: {e}")
                # デフォルト値を使用（ダウンサンプリング調整）
                self.depth_intrinsics = CameraIntrinsics(
                    fx=640.0 / self.downsample_factor,
                    fy=640.0 / self.downsample_factor,
                    cx=640.0 / self.downsample_factor,
                    cy=360.0 / self.downsample_factor,
                    width=self.oak_depth_width,
                    height=self.oak_depth_height
                )
                logger.info("深度内部パラメータ: ダウンサンプリング調整デフォルト値を使用")
            
            # カラーカメラの内部パラメータ（1080p）
            if self.has_color:
                try:
                    color_intrinsics = calib.getCameraIntrinsics(
                        dai.CameraBoardSocket.RGB, 1920, 1080
                    )
                    
                    if color_intrinsics and len(color_intrinsics) >= 4:
                        # リスト型の場合は適切にアンパック
                        if isinstance(color_intrinsics, list):
                            fx, fy, cx, cy = color_intrinsics[:4]
                        else:
                            fx, fy, cx, cy = color_intrinsics[:4]
                        
                        self.color_intrinsics = CameraIntrinsics(
                            fx=float(fx),
                            fy=float(fy),
                            cx=float(cx),
                            cy=float(cy),
                            width=1920,
                            height=1080
                        )
                        logger.info(f"カラー内部パラメータ: fx={float(fx):.1f}, fy={float(fy):.1f}, cx={float(cx):.1f}, cy={float(cy):.1f}")
                    elif color_intrinsics and len(color_intrinsics) == 3:
                        # 3つの値しか返されない場合の処理
                        if isinstance(color_intrinsics, list):
                            fx, fy, f_mean = color_intrinsics[:3]
                        else:
                            fx, fy, f_mean = color_intrinsics[:3]
                        cx, cy = 1920 / 2, 1080 / 2  # 中央値を使用
                        
                        self.color_intrinsics = CameraIntrinsics(
                            fx=float(fx),
                            fy=float(fy),
                            cx=float(cx),
                            cy=float(cy),
                            width=1920,
                            height=1080
                        )
                        logger.info(f"カラー内部パラメータ（3値）: fx={float(fx):.1f}, fy={float(fy):.1f}, cx={float(cx):.1f}, cy={float(cy):.1f}")
                    else:
                        raise ValueError("内部パラメータの値が不足しています")
                        
                except Exception as e:
                    logger.warning(f"カラー内部パラメータ取得エラー: {e}")
                    # デフォルト値を使用
                    self.color_intrinsics = CameraIntrinsics(
                        fx=960.0,
                        fy=960.0,
                        cx=960.0,
                        cy=540.0,
                        width=1920,
                        height=1080
                    )
                    logger.info("カラー内部パラメータ: デフォルト値を使用")
                    
        except Exception as e:
            logger.error(f"内部パラメータ設定エラー: {e}")
            # 全てデフォルト値を使用（ダウンサンプリング調整）
            self.depth_intrinsics = CameraIntrinsics(
                fx=640.0 / self.downsample_factor,
                fy=640.0 / self.downsample_factor,
                cx=640.0 / self.downsample_factor,
                cy=360.0 / self.downsample_factor,
                width=self.oak_depth_width,
                height=self.oak_depth_height
            )
            if self.has_color:
                self.color_intrinsics = CameraIntrinsics(
                    fx=960.0,
                    fy=960.0,
                    cx=960.0,
                    cy=540.0,
                    width=1920,
                    height=1080
                )
            logger.info("内部パラメータ: ダウンサンプリング調整デフォルト値を使用")
    
    def start(self) -> bool:
        """
        ストリーミング開始
        
        Returns:
            成功した場合True
            
        Raises:
            RuntimeError: 致命的な開始エラー
        """
        if not self.device:
            logger.error("Device not initialized")
            return False
            
        try:
            # OAK-D S2はDevice初期化と同時にストリーミング開始
            self.is_started = True
            logger.info("OAK-D S2 ストリーミング開始")
            return True
            
        except Exception as e:
            logger.error(f"ストリーミング開始エラー: {e}")
            return False
    
    def cleanup(self) -> bool:
        """リソースクリーンアップ（ManagedResourceインターフェース）"""
        try:
            self.stop()
            
            if self.device:
                try:
                    self.device.close()
                    self.device = None
                except Exception as e:
                    logger.warning(f"デバイスクローズエラー: {e}")
            
            self.pipeline = None
            self.q_depth = None
            self.q_rgb = None
            self.depth_intrinsics = None
            self.color_intrinsics = None
            self.has_color = False
            self.is_started = False
            
            logger.info(f"OAK-D S2 リソースクリーンアップ完了: {self.resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")
            return False
    
    def stop(self) -> None:
        """ストリーミング停止"""
        if self.is_started:
            try:
                # OAK-D S2はデバイスクローズでストリーミング停止
                self.is_started = False
                logger.info("OAK-D S2 ストリーミング停止")
            except Exception as e:
                logger.warning(f"ストリーミング停止エラー: {e}")
    
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
        if not self.is_started or not self.q_depth:
            return None
            
        try:
            # 深度フレーム取得
            depth_frame = None
            in_depth = self.q_depth.tryGet()
            if in_depth is not None:
                depth_frame = in_depth.getFrame()
                
                # 実際のフレームサイズを確認してデバッグ出力
                if depth_frame is not None:
                    logger.debug(f"実際の深度フレーム形状: {depth_frame.shape}, データタイプ: {depth_frame.dtype}")
                    
                    # OAK-D S2固有のダウンサンプリング処理
                    if self.downsample_factor > 1:
                        # 高速ダウンサンプリング（点群数を大幅に削減）
                        depth_frame = depth_frame[::self.downsample_factor, ::self.downsample_factor]
                        logger.debug(f"ダウンサンプリング後の深度フレーム形状: {depth_frame.shape}")
                    
                    # 期待される形状に調整
                    target_shape = (self.oak_depth_height, self.oak_depth_width)
                    if depth_frame.shape != target_shape:
                        logger.debug(f"深度フレーム形状調整: {depth_frame.shape} -> {target_shape}")
                        depth_frame = cv2.resize(depth_frame, (self.oak_depth_width, self.oak_depth_height), interpolation=cv2.INTER_NEAREST)
                        
            # カラーフレーム取得
            color_frame = None
            if self.has_color and self.q_rgb:
                in_rgb = self.q_rgb.tryGet()
                if in_rgb is not None:
                    color_frame = in_rgb.getCvFrame()
                    
                    # カラーフレームの詳細デバッグ
                    if color_frame is not None:
                        logger.debug(f"カラーフレーム取得成功: 形状={color_frame.shape}, データタイプ={color_frame.dtype}")
                        logger.debug(f"カラーフレーム値範囲: min={color_frame.min()}, max={color_frame.max()}")
                        
                        # カラーフレームが真っ黒かどうかチェック
                        if color_frame.max() == 0:
                            logger.warning("カラーフレームが真っ黒です（全ての値が0）")
                        else:
                            logger.debug(f"カラーフレームは正常です（非ゼロ値: {np.count_nonzero(color_frame)} / {color_frame.size}）")
                    else:
                        logger.warning("カラーフレームが None です")
                else:
                    logger.debug("カラーフレームキューが空です")
            elif self.has_color:
                logger.debug("カラーフレームが有効ですが、キューが初期化されていません")
                    
            # フレームデータ作成
            if depth_frame is not None:
                # OAK-D S2フレームをOrbbec互換形式に変換
                oak_depth_frame = OakDepthFrame(depth_frame)
                oak_color_frame = OakColorFrame(color_frame) if color_frame is not None else None
                
                frame_data = FrameData(
                    depth_frame=oak_depth_frame,
                    color_frame=oak_color_frame,
                    timestamp_ms=time.perf_counter() * 1000,
                    frame_number=self.frame_count
                )
                
                self.frame_count += 1
                return frame_data
            
            return None
            
        except Exception as e:
            logger.warning(f"フレーム取得エラー: {e}")
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
            raise RuntimeError("Failed to initialize OAK-D S2")
        if not self.start():
            raise RuntimeError("Failed to start OAK-D S2")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 終了"""
        self.stop()
    
    @property
    def resource_type(self) -> str:
        """リソースタイプ（ManagedResourceインターフェース）"""
        return "oak_camera"
    
    def get_memory_usage(self) -> int:
        """メモリ使用量を取得（ManagedResourceインターフェース）"""
        # 概算値の計算
        base_memory = 100 * 1024 * 1024  # 100MB基本
        if self.has_color:
            base_memory += 50 * 1024 * 1024  # カラーで+50MB
        frame_buffer_memory = self.frame_count * 2048  # 2KB/フレーム
        return base_memory + frame_buffer_memory


class OakDepthFrame:
    """OAK-D S2深度フレームのOrbbec互換ラッパー"""
    
    def __init__(self, depth_frame: np.ndarray):
        # 深度フレームの形状とデータタイプを確認
        logger.debug(f"OakDepthFrame 初期化: 形状={depth_frame.shape}, データタイプ={depth_frame.dtype}")
        
        # uint16 形式に変換（必要に応じて）
        if depth_frame.dtype != np.uint16:
            logger.warning(f"深度フレームデータタイプ変換: {depth_frame.dtype} -> uint16")
            if depth_frame.dtype == np.float32 or depth_frame.dtype == np.float64:
                # float から uint16 への変換（スケーリング）
                depth_frame = (depth_frame * 1000).astype(np.uint16)  # メートル -> ミリメートル
            else:
                depth_frame = depth_frame.astype(np.uint16)
        
        # C-contiguous配列に変換
        self.depth_frame = np.ascontiguousarray(depth_frame)
        
        logger.debug(f"OakDepthFrame 処理完了: 形状={self.depth_frame.shape}, データタイプ={self.depth_frame.dtype}")
    
    def get_data(self) -> bytes:
        """深度データをバイト列で取得"""
        return self.depth_frame.tobytes()


class OakColorFrame:
    """OAK-D S2カラーフレームのOrbbec互換ラッパー"""
    
    def __init__(self, color_frame: Optional[np.ndarray]):
        if color_frame is None:
            # カラーフレームが None の場合は、デフォルトの空フレームを作成
            logger.debug("OakColorFrame: カラーフレームが None のため、デフォルトフレームを作成")
            self.color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.is_valid = False
        else:
            # カラーフレームの形状とデータタイプを確認
            logger.debug(f"OakColorFrame 初期化: 形状={color_frame.shape}, データタイプ={color_frame.dtype}")
            
            # uint8 形式に変換（必要に応じて）
            if color_frame.dtype != np.uint8:
                logger.warning(f"カラーフレームデータタイプ変換: {color_frame.dtype} -> uint8")
                if color_frame.dtype == np.float32 or color_frame.dtype == np.float64:
                    # float から uint8 への変換（正規化）
                    color_frame = (color_frame * 255).astype(np.uint8)
                else:
                    color_frame = color_frame.astype(np.uint8)
            
            # RGB→BGR変換が必要かチェック（OAK-D S2はBGRで出力）
            if len(color_frame.shape) == 3 and color_frame.shape[2] == 3:
                # すでにBGRと仮定
                self.color_frame = np.ascontiguousarray(color_frame)
                self.is_valid = True
            else:
                logger.warning(f"カラーフレーム形状が不正: {color_frame.shape}")
                self.color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.is_valid = False
            
            logger.debug(f"OakColorFrame 処理完了: 形状={self.color_frame.shape}, データタイプ={self.color_frame.dtype}, 有効={self.is_valid}")
    
    def get_data(self) -> bytes:
        """カラーデータをバイト列で取得"""
        return self.color_frame.tobytes()
        
    def get_format(self):
        """フォーマットを取得（BGR固定）"""
        from ..data_types import OBFormat
        return OBFormat.BGR 