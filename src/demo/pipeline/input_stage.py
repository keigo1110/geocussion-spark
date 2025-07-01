#!/usr/bin/env python3
"""
入力ステージ: カメラ入力、フィルタリング、点群変換

カメラからのフレーム取得、深度フィルタリング、点群変換を担当します。
"""

from typing import Optional, Tuple, Any
import numpy as np
import cv2
from dataclasses import dataclass

from .base import PipelineStage, StageResult
from ...types import FrameData, OBFormat
from ...input.stream import OrbbecCamera
from ...input.depth_filter import DepthFilter, FilterType
from ...input.pointcloud import PointCloudConverter


@dataclass
class InputStageConfig:
    """入力ステージの設定"""
    enable_filter: bool = True
    filter_type: FilterType = FilterType.COMBINED
    enable_voxel_downsampling: bool = True
    voxel_size: float = 0.005
    # カメラ解像度
    depth_width: int = 424
    depth_height: int = 240
    rgb_width: int = 640
    rgb_height: int = 480
    # フレームフォーマット
    color_format: OBFormat = OBFormat.RGB


@dataclass 
class InputStageResult(StageResult):
    """入力ステージの処理結果"""
    frame_data: Optional[FrameData] = None
    point_cloud: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    color_image: Optional[np.ndarray] = None  # MediaPipe用
    color_mjpg_data: Optional[bytes] = None  # MJPG圧縮データ


class InputStage(PipelineStage):
    """入力ステージの実装"""
    
    def __init__(self, config: InputStageConfig, camera: Optional[OrbbecCamera] = None) -> None:
        """
        初期化
        
        Args:
            config: 入力ステージ設定
            camera: カメラインスタンス（外部から注入される場合）
        """
        super().__init__(config)
        self.config: InputStageConfig = config
        self.camera = camera
        self.depth_filter: Optional[DepthFilter] = None
        self.point_cloud_converter: Optional[PointCloudConverter] = None
        
    def initialize(self) -> bool:
        """ステージの初期化"""
        try:
            # 深度フィルタ初期化
            if self.config.enable_filter:
                # FilterTypeをリストとして渡す
                filter_types = [self.config.filter_type] if self.config.filter_type != FilterType.COMBINED else None
                self.depth_filter = DepthFilter(filter_types=filter_types)
                self.logger.info(f"深度フィルタを初期化しました: {self.config.filter_type}")
            
            # 点群変換器は最初のフレームで初期化
            # （カメラのintrinsicsが必要なため）
            self.logger.info("点群変換器は最初のフレーム処理時に初期化されます")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"入力ステージの初期化に失敗: {e}")
            return False
    
    def process(self, frame_data: Optional[FrameData] = None) -> InputStageResult:
        """
        フレームデータを処理
        
        Args:
            frame_data: カメラからのフレームデータ（Noneの場合は内部カメラから取得）
            
        Returns:
            処理結果
        """
        if not self._initialized:
            return InputStageResult(
                success=False,
                error_message="Stage not initialized"
            )
        
        try:
            # フレームデータの取得
            if frame_data is None and self.camera is not None:
                frame_data = self.camera.get_frame()
                if frame_data is None:
                    return InputStageResult(
                        success=False,
                        error_message="Failed to get frame from camera"
                    )
            elif frame_data is None:
                return InputStageResult(
                    success=False,
                    error_message="No frame data provided and no camera available"
                )
            
            # 深度フィルタ適用
            if self.depth_filter and frame_data.depth_frame is not None:
                # depth_frameから実際のデータを取得
                depth_data = frame_data.depth_frame.get_data() if hasattr(frame_data.depth_frame, 'get_data') else frame_data.depth_frame
                if depth_data is not None:
                    # depth_dataをnumpy配列に変換
                    if not isinstance(depth_data, np.ndarray):
                        # バイト配列からuint16配列に変換
                        depth_array = np.frombuffer(depth_data, dtype=np.uint16)
                        # カメラのintrinsicsから形状を取得
                        if self.camera and hasattr(self.camera, 'depth_intrinsics') and self.camera.depth_intrinsics:
                            height = self.camera.depth_intrinsics.height
                            width = self.camera.depth_intrinsics.width
                            expected_size = height * width
                            if depth_array.size == expected_size:
                                depth_data = depth_array.reshape((height, width))
                            else:
                                # サイズが一致しない場合は警告
                                self.logger.warning(f"深度データサイズ不一致: 期待値={expected_size}, 実際={depth_array.size}")
                                # 最も近い有効な形状にリサイズ
                                if depth_array.size > expected_size:
                                    depth_data = depth_array[:expected_size].reshape((height, width))
                                else:
                                    # データが少ない場合はスキップ
                                    return InputStageResult(
                                        success=False,
                                        error_message=f"Invalid depth data size: {depth_array.size}"
                                    )
                    else:
                        # すでにnumpy配列の場合、形状を確認
                        if depth_data.ndim == 1:
                            # 1次元配列の場合はreshape
                            if self.camera and hasattr(self.camera, 'depth_intrinsics') and self.camera.depth_intrinsics:
                                height = self.camera.depth_intrinsics.height
                                width = self.camera.depth_intrinsics.width
                                expected_size = height * width
                                if depth_data.size == expected_size:
                                    depth_data = depth_data.reshape((height, width))
                    
                    # 深度データの形状を確認し、必要に応じてreshape
                    if depth_data.ndim == 1:
                        # 1次元配列の場合、カメラのintrinsicsから形状を推定
                        if self.camera and hasattr(self.camera, 'depth_intrinsics'):
                            height = self.camera.depth_intrinsics.height
                            width = self.camera.depth_intrinsics.width
                            expected_size = height * width
                            if depth_data.size == expected_size:
                                depth_data = depth_data.reshape((height, width))
                                self.logger.debug(f"深度データを{height}x{width}にreshapeしました")
                            else:
                                self.logger.error(f"深度データサイズ不一致: 期待値={expected_size}, 実際={depth_data.size}")
                                return InputStageResult(
                                    success=False,
                                    error_message=f"Invalid depth data size: expected {expected_size}, got {depth_data.size}"
                                )
                        else:
                            self.logger.error(f"1次元深度データをreshapeできません: intrinsicsが利用できません")
                            return InputStageResult(
                                success=False,
                                error_message=f"Cannot reshape 1D depth data without camera intrinsics"
                            )
                    elif depth_data.ndim != 2:
                        self.logger.error(f"無効な深度データ形状: {depth_data.shape}")
                        return InputStageResult(
                            success=False,
                            error_message=f"Invalid depth data shape: {depth_data.shape}"
                        )
                    
                    filtered_depth = self.depth_filter.apply_filter(depth_data)
                    
                    # フィルタ結果が2次元配列であることを確認
                    if filtered_depth.ndim > 2:
                        # 余分な次元を削除（例: (H, W, 1) -> (H, W))
                        filtered_depth = filtered_depth.squeeze()
                    
                    # フィルタ済みデータでモックフレームを更新
                    class FilteredFrame:
                        def __init__(self, data):
                            self.data = data
                        def get_data(self):
                            return self.data
                    
                    frame_data = FrameData(
                        depth_frame=FilteredFrame(filtered_depth),
                        color_frame=frame_data.color_frame,
                        timestamp_ms=frame_data.timestamp_ms,
                        frame_number=frame_data.frame_number
                    )
            else:
                # フィルタが無効な場合でも、深度データの形状を確認
                if frame_data.depth_frame is not None:
                    depth_data = frame_data.depth_frame.get_data() if hasattr(frame_data.depth_frame, 'get_data') else frame_data.depth_frame
                    if depth_data is not None:
                        # numpy配列に変換
                        if not isinstance(depth_data, np.ndarray):
                            # バイト配列からuint16配列に変換
                            depth_array = np.frombuffer(depth_data, dtype=np.uint16)
                        else:
                            depth_array = depth_data
                        
                        # カメラのintrinsicsから形状を取得
                        if self.camera and hasattr(self.camera, 'depth_intrinsics') and self.camera.depth_intrinsics:
                            height = self.camera.depth_intrinsics.height
                            width = self.camera.depth_intrinsics.width
                            expected_size = height * width
                            
                            # 1次元配列の場合はreshape
                            if depth_array.ndim == 1 and depth_array.size == expected_size:
                                depth_data = depth_array.reshape((height, width))
                                # reshape済みデータでフレームを更新
                                class ReshapedFrame:
                                    def __init__(self, data):
                                        self.data = data
                                    def get_data(self):
                                        return self.data
                                
                                frame_data = FrameData(
                                    depth_frame=ReshapedFrame(depth_data),
                                    color_frame=frame_data.color_frame,
                                    timestamp_ms=frame_data.timestamp_ms,
                                    frame_number=frame_data.frame_number
                                )
            
            # 点群変換器の遅延初期化（カメラから直接intrinsicsを取得）
            if self.point_cloud_converter is None and self.camera and hasattr(self.camera, 'depth_intrinsics') and self.camera.depth_intrinsics:
                self.point_cloud_converter = PointCloudConverter(
                    depth_intrinsics=self.camera.depth_intrinsics,
                    enable_voxel_downsampling=self.config.enable_voxel_downsampling,
                    voxel_size=self.config.voxel_size
                )
                self.logger.info("点群変換器を初期化しました")
            
            # 点群変換
            point_cloud = None
            colors = None
            if frame_data.depth_frame is not None and self.point_cloud_converter is not None:
                # 深度データを取得（フィルタ済みのデータを使用）
                depth_data = frame_data.depth_frame.get_data() if hasattr(frame_data.depth_frame, 'get_data') else frame_data.depth_frame
                
                # numpy配列に変換
                if isinstance(depth_data, np.ndarray):
                    depth_array = depth_data
                else:
                    # バイト配列の場合はuint16に変換
                    depth_array = np.frombuffer(depth_data, dtype=np.uint16)
                
                # 形状確認とreshape（必要な場合）
                if self.camera and hasattr(self.camera, 'depth_intrinsics') and self.camera.depth_intrinsics:
                    height = self.camera.depth_intrinsics.height
                    width = self.camera.depth_intrinsics.width
                    expected_size = height * width
                    
                    if depth_array.ndim == 1:
                        # 1次元配列の場合はreshape
                        if depth_array.size == expected_size:
                            depth_array = depth_array.reshape((height, width))
                            self.logger.debug(f"点群変換用に深度データを{height}x{width}にreshapeしました")
                        else:
                            self.logger.error(f"深度データサイズ不一致: 期待値={expected_size}, 実際={depth_array.size}")
                            return InputStageResult(
                                success=False,
                                error_message=f"Invalid depth data size in pointcloud conversion: {depth_array.size}"
                            )
                    elif depth_array.ndim != 2:
                        self.logger.error(f"無効な深度データ次元数: {depth_array.ndim}")
                        return InputStageResult(
                            success=False,
                            error_message=f"Invalid depth data dimensions: {depth_array.ndim}"
                        )
                
                # カラーデータを取得（オプション）
                color_array = None
                if frame_data.color_frame is not None:
                    color_array = frame_data.color_frame
                
                # numpy_to_pointcloudメソッドを使用
                point_cloud, colors = self.point_cloud_converter.numpy_to_pointcloud(
                    depth_array,
                    color_array=color_array
                )
            
            # MediaPipe用のカラー画像抽出
            color_image = None
            color_mjpg_data = None
            if frame_data.color_frame is not None:
                color_image, color_mjpg_data = self._extract_color_image(frame_data)
            
            return InputStageResult(
                success=True,
                frame_data=frame_data,
                point_cloud=point_cloud,
                colors=colors,
                color_image=color_image,
                color_mjpg_data=color_mjpg_data
            )
            
        except Exception as e:
            self.logger.error(f"入力処理エラー: {e}")
            return InputStageResult(
                success=False,
                error_message=str(e)
            )
    
    def _extract_color_image(self, frame_data: FrameData) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
        """
        MediaPipe用のカラー画像を抽出
        
        Args:
            frame_data: フレームデータ
            
        Returns:
            (RGB画像, MJPGデータ)のタプル
        """
        if frame_data.color_frame is None:
            return None, None
        
        # カラーフレームからデータを取得
        color_data = frame_data.color_frame.get_data() if hasattr(frame_data.color_frame, 'get_data') else frame_data.color_frame
        mjpg_data = None
        
        # フォーマットを推定（実際のカメラから取得すべきだが、ここではMJPGと仮定）
        color_format = getattr(frame_data.color_frame, 'get_format', lambda: OBFormat.MJPG)()
        
        # MJPGフォーマットの場合はデコード
        if str(color_format) == "OBFormat.MJPG" or str(color_format) == "MJPG":
            try:
                # 元のMJPGデータを保存
                mjpg_data = color_data
                
                # バイト配列に変換
                if isinstance(color_data, np.ndarray):
                    jpeg_bytes = color_data.tobytes()
                elif hasattr(color_data, 'tobytes'):
                    jpeg_bytes = color_data.tobytes()
                else:
                    jpeg_bytes = bytes(color_data)
                
                # データサイズチェック
                if len(jpeg_bytes) < 100:  # JPEGヘッダー最小サイズ
                    self.logger.warning(f"MJPG data too small: {len(jpeg_bytes)} bytes")
                    return None, mjpg_data
                
                # デコード
                nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                color_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if color_image is None:
                    self.logger.warning("MJPGデコードに失敗しました")
                    return None, mjpg_data
                
                # BGRからRGBに変換
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
            except Exception as e:
                self.logger.error(f"MJPG処理エラー: {e}")
                return None, mjpg_data
        else:
            # RGB/BGRフォーマットの場合
            color_image = color_data
            if frame_data.color_format == OBFormat.BGR:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        return color_image, mjpg_data
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.depth_filter:
            self.depth_filter = None
        if self.point_cloud_converter:
            self.point_cloud_converter = None
        self._initialized = False
        self.logger.info("入力ステージをクリーンアップしました")
    
    def get_camera_intrinsics(self):
        """カメラ内部パラメータを取得"""
        if self.camera and hasattr(self.camera, 'depth_intrinsics'):
            return self.camera.depth_intrinsics
        return None