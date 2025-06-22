#!/usr/bin/env python3
"""
点群変換処理モジュール
既存 depth_to_pointcloud 関数のリファクタリング版
numpy 最適化とゼロコピー転送対応
"""

import time
from typing import Optional, Tuple, Union
import numpy as np
import cv2

try:
    from pyorbbecsdk import OBFormat
except ImportError:
    # テスト用のモック定義
    class OBFormat:
        RGB = "RGB"
        BGR = "BGR"
        MJPG = "MJPG"

from .stream import CameraIntrinsics, FrameData


class PointCloudConverter:
    """深度フレームから点群への変換クラス"""
    
    def __init__(self, depth_intrinsics: CameraIntrinsics):
        """
        初期化
        
        Args:
            depth_intrinsics: 深度カメラの内部パラメータ
        """
        self.depth_intrinsics = depth_intrinsics
        self.depth_scale = 1000.0  # mm to m conversion
        
        # メッシュグリッドを事前計算（パフォーマンス最適化）
        self._precompute_meshgrid()
        
    def _precompute_meshgrid(self) -> None:
        """メッシュグリッドを事前計算"""
        width = self.depth_intrinsics.width
        height = self.depth_intrinsics.height
        
        # ピクセル座標のメッシュグリッド
        self.pixel_x, self.pixel_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        
        # カメラ座標系への変換係数を事前計算
        self.x_coeff = (self.pixel_x - self.depth_intrinsics.cx) / self.depth_intrinsics.fx
        self.y_coeff = -(self.pixel_y - self.depth_intrinsics.cy) / self.depth_intrinsics.fy  # 手の3D投影と座標系統一
    
    def depth_to_pointcloud(
        self,
        depth_frame,
        color_frame=None,
        depth_scale: Optional[float] = None,
        min_depth: float = 0.1,
        max_depth: float = 10.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        深度フレームから点群を生成（最適化版）
        
        Args:
            depth_frame: 深度フレーム
            color_frame: カラーフレーム（オプション）
            depth_scale: 深度スケール（None でデフォルト使用）
            min_depth: 最小深度閾値 (m)
            max_depth: 最大深度閾値 (m)
            
        Returns:
            (points, colors): 点群座標とカラー情報のタプル
        """
        start_time = time.perf_counter()
        
        # 深度データ取得
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_image = depth_data.reshape(
            (self.depth_intrinsics.height, self.depth_intrinsics.width)
        )
        
        # 深度スケール適用
        scale = depth_scale if depth_scale is not None else self.depth_scale
        z = depth_image.astype(np.float32) / scale
        
        # 深度範囲フィルタ
        valid_depth = (z > min_depth) & (z < max_depth)
        
        # 3D座標計算（vectorized operations）
        x = self.x_coeff * z
        y = self.y_coeff * z
        
        # 有効点のマスク
        valid = valid_depth
        
        # 点群座標を作成（ゼロコピー最適化）
        points = np.column_stack([
            x[valid],
            y[valid], 
            z[valid]
        ])
        
        # カラー情報処理
        colors = None
        if color_frame is not None:
            colors = self._extract_colors(color_frame, valid)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return points, colors
    
    def _extract_colors(self, color_frame, valid_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        カラーフレームからカラー情報を抽出
        
        Args:
            color_frame: カラーフレーム
            valid_mask: 有効点のマスク
            
        Returns:
            カラー配列 (N, 3) または None
        """
        try:
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_format = color_frame.get_format()
            
            # フォーマットに応じた処理
            if color_format == OBFormat.RGB:
                color_image = color_data.reshape((self.depth_intrinsics.height, self.depth_intrinsics.width, 3))
            elif color_format == OBFormat.BGR:
                color_image = color_data.reshape((self.depth_intrinsics.height, self.depth_intrinsics.width, 3))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            elif color_format == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if color_image is not None:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            else:
                return None
            
            if color_image is None:
                return None
                
            # サイズが異なる場合はリサイズ
            if color_image.shape[:2] != (self.depth_intrinsics.height, self.depth_intrinsics.width):
                color_image = cv2.resize(
                    color_image,
                    (self.depth_intrinsics.width, self.depth_intrinsics.height)
                )
            
            # 有効点のカラーを抽出（正規化）
            colors = color_image[valid_mask].astype(np.float32) / 255.0
            return colors
            
        except Exception as e:
            print(f"Color extraction error: {e}")
            return None
    
    def numpy_to_pointcloud(
        self,
        depth_array: np.ndarray,
        color_array: Optional[np.ndarray] = None,
        depth_scale: Optional[float] = None,
        min_depth: float = 0.1,
        max_depth: float = 10.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        numpy配列から直接点群を生成
        
        Args:
            depth_array: 深度配列 (H, W)
            color_array: カラー配列 (H, W, 3) オプション
            depth_scale: 深度スケール
            min_depth: 最小深度閾値
            max_depth: 最大深度閾値
            
        Returns:
            (points, colors): 点群座標とカラー情報
        """
        if depth_array.shape != (self.depth_intrinsics.height, self.depth_intrinsics.width):
            raise ValueError(f"Depth array shape mismatch. Expected {(self.depth_intrinsics.height, self.depth_intrinsics.width)}, got {depth_array.shape}")
        
        # 深度スケール適用
        scale = depth_scale if depth_scale is not None else self.depth_scale
        z = depth_array.astype(np.float32) / scale
        
        # 深度範囲フィルタ
        valid = (z > min_depth) & (z < max_depth)
        
        # 3D座標計算
        x = self.x_coeff * z
        y = self.y_coeff * z  # 既にy_coeffでマイナス符号適用済み
        
        # 点群作成
        points = np.column_stack([
            x[valid],
            y[valid],
            z[valid]
        ])
        
        # カラー情報
        colors = None
        if color_array is not None:
            if color_array.shape[:2] != depth_array.shape:
                color_array = cv2.resize(color_array, (depth_array.shape[1], depth_array.shape[0]))
            colors = color_array[valid].astype(np.float32) / 255.0
        
        return points, colors
    
    def update_intrinsics(self, new_intrinsics: CameraIntrinsics) -> None:
        """
        カメラ内部パラメータを更新
        
        Args:
            new_intrinsics: 新しい内部パラメータ
        """
        self.depth_intrinsics = new_intrinsics
        self._precompute_meshgrid()


def create_converter_from_frame_data(frame_data: FrameData, depth_intrinsics: CameraIntrinsics) -> PointCloudConverter:
    """
    FrameDataからPointCloudConverterを作成するヘルパー関数
    
    Args:
        frame_data: フレームデータ
        depth_intrinsics: 深度カメラ内部パラメータ
        
    Returns:
        PointCloudConverter インスタンス
    """
    return PointCloudConverter(depth_intrinsics)


# 互換性のための関数（既存コードとの互換性保持）
def depth_to_pointcloud(depth_frame, camera_intrinsic, color_frame=None):
    """
    レガシー互換関数
    既存のpoint_cloud_realtime_viewer.pyとの互換性のため
    """
    # CameraIntrinsicsオブジェクトに変換
    intrinsics = CameraIntrinsics(
        fx=camera_intrinsic.fx,
        fy=camera_intrinsic.fy,
        cx=camera_intrinsic.cx,
        cy=camera_intrinsic.cy,
        width=depth_frame.get_width(),
        height=depth_frame.get_height()
    )
    
    # ConverterでPointCloudを生成
    converter = PointCloudConverter(intrinsics)
    return converter.depth_to_pointcloud(depth_frame, color_frame) 