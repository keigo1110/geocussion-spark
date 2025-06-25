#!/usr/bin/env python3
"""
点群変換処理モジュール
既存 depth_to_pointcloud 関数のリファクタリング版
numpy 最適化とゼロコピー転送対応
"""

import time
from typing import Optional, Tuple, Union, Any
import numpy as np
import cv2

try:
    from pyorbbecsdk import OBFormat
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    # テスト用のモック定義（types.pyのOBFormatと統合）
    from ..types import OBFormat
    HAS_OPEN3D = False

from .stream import CameraIntrinsics, FrameData


class PointCloudConverter:
    """深度フレームから点群への変換クラス"""
    
    def __init__(
        self, 
        depth_intrinsics: CameraIntrinsics,
        depth_scale: float = 1000.0,
        enable_voxel_downsampling: bool = True,
        voxel_size: float = 0.005,  # 5mm voxel size for good balance
        enable_resolution_downsampling: bool = False,
        target_width: int = 424,
        target_height: int = 240
    ):
        """
        初期化
        
        Args:
            depth_intrinsics: 深度カメラの内部パラメータ
            depth_scale: 深度スケール (mm → m変換)
            enable_voxel_downsampling: ボクセルダウンサンプリングを有効にするか
            voxel_size: ボクセルサイズ (m) - 小さいほど高精度、大きいほど高速
            enable_resolution_downsampling: 解像度ダウンサンプリングを有効にするか
            target_width: 目標解像度（幅）
            target_height: 目標解像度（高さ）
        """
        self.depth_intrinsics = depth_intrinsics
        self.depth_scale = depth_scale
        self.enable_voxel_downsampling = enable_voxel_downsampling
        self.voxel_size = voxel_size
        
        # 解像度ダウンサンプリング設定
        self.enable_resolution_downsampling = enable_resolution_downsampling
        self.target_width = target_width
        self.target_height = target_height
        
        # 解像度ダウンサンプリング比率計算
        if enable_resolution_downsampling:
            self.width_ratio = target_width / depth_intrinsics.width
            self.height_ratio = target_height / depth_intrinsics.height
            
            # ダウンサンプリング後の内部パラメータ
            self.downsampled_intrinsics = CameraIntrinsics(
                fx=depth_intrinsics.fx * self.width_ratio,
                fy=depth_intrinsics.fy * self.height_ratio,
                cx=depth_intrinsics.cx * self.width_ratio,
                cy=depth_intrinsics.cy * self.height_ratio,
                width=target_width,
                height=target_height
            )
        else:
            self.downsampled_intrinsics = depth_intrinsics
        
        # メッシュグリッドを事前計算（パフォーマンス最適化）
        self._precompute_meshgrid()
        
        # 3D座標計算用の係数を事前計算（高速化）
        self._precompute_coefficients()
        
        # パフォーマンス統計
        self.performance_stats = {
            'total_conversions': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_input_points': 0,
            'last_output_points': 0,
            'last_downsampling_ratio': 0.0,
            'total_downsampling_time_ms': 0.0,
            'resolution_downsampling_enabled': enable_resolution_downsampling,
            'resolution_downsampling_ratio': f"{target_width}x{target_height}" if enable_resolution_downsampling else "disabled"
        }
        
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
    
    def _precompute_coefficients(self):
        """3D座標計算用の係数を事前計算"""
        # ピクセル座標のメッシュグリッドを作成
        height, width = self.depth_intrinsics.height, self.depth_intrinsics.width
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # X, Y座標の計算係数（Z値との乗算で3D座標が得られる）
        self.x_coeff = (u - self.depth_intrinsics.cx) / self.depth_intrinsics.fx
        self.y_coeff = -(v - self.depth_intrinsics.cy) / self.depth_intrinsics.fy  # Y軸反転
    
    def depth_to_pointcloud(
        self,
        depth_frame: Any,
        color_frame: Optional[Any] = None,
        depth_scale: Optional[float] = None,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        voxel_size: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        深度フレームから点群を生成（ボクセルダウンサンプリング対応）
        
        Args:
            depth_frame: 深度フレーム
            color_frame: カラーフレーム（オプション）
            depth_scale: 深度スケール（None でデフォルト使用）
            min_depth: 最小深度閾値 (m)
            max_depth: 最大深度閾値 (m)
            voxel_size: ボクセルサイズ（None でデフォルト使用）
            
        Returns:
            (points, colors): 点群座標とカラー情報のタプル
        """
        start_time = time.perf_counter()
        
        # 深度データ取得
        depth_data: np.ndarray = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_image = depth_data.reshape(
            (self.depth_intrinsics.height, self.depth_intrinsics.width)
        )
        
        # 解像度ダウンサンプリング適用
        if self.enable_resolution_downsampling:
            # 深度画像をリサイズ
            depth_image = cv2.resize(
                depth_image, 
                (self.target_width, self.target_height), 
                interpolation=cv2.INTER_NEAREST  # 深度値保持のためNearest使用
            )
            # ダウンサンプリング後の内部パラメータを使用
            effective_intrinsics = self.downsampled_intrinsics
        else:
            effective_intrinsics = self.depth_intrinsics
        
        # 深度スケール適用
        scale = depth_scale if depth_scale is not None else self.depth_scale
        z = depth_image.astype(np.float32) / scale
        
        # 深度範囲フィルタ
        valid_depth = (z > min_depth) & (z < max_depth)
        
        # 3D座標計算用のメッシュグリッド（ダウンサンプリング対応）
        if self.enable_resolution_downsampling:
            height, width = self.target_height, self.target_width
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # ダウンサンプリング後の座標計算係数
            x_coeff = (u - effective_intrinsics.cx) / effective_intrinsics.fx
            y_coeff = -(v - effective_intrinsics.cy) / effective_intrinsics.fy  # Y軸反転
        else:
            # 元の係数を使用
            x_coeff = self.x_coeff
            y_coeff = self.y_coeff
        
        # 3D座標計算（vectorized operations）
        x = x_coeff * z
        y = y_coeff * z
        
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
        
        # 入力点数を記録
        input_points = len(points)
        self.performance_stats['last_input_points'] = input_points
        
        # ボクセルダウンサンプリング適用
        if self.enable_voxel_downsampling and input_points > 0:
            downsampling_start = time.perf_counter()
            points, colors = self._apply_voxel_downsampling(
                points, colors, voxel_size or self.voxel_size
            )
            downsampling_time = (time.perf_counter() - downsampling_start) * 1000
            self.performance_stats['total_downsampling_time_ms'] += downsampling_time
            
            # ダウンサンプリング率を計算
            output_points = len(points)
            self.performance_stats['last_output_points'] = output_points
            self.performance_stats['last_downsampling_ratio'] = (
                output_points / input_points if input_points > 0 else 0.0
            )
        else:
            self.performance_stats['last_output_points'] = input_points
            self.performance_stats['last_downsampling_ratio'] = 1.0
        
        # パフォーマンス統計更新
        processing_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_stats(processing_time)
        
        return points, colors
    
    def _apply_voxel_downsampling(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        voxel_size: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        ボクセルダウンサンプリングを適用
        
        Args:
            points: 入力点群 (N, 3)
            colors: 入力カラー情報 (N, 3) または None
            voxel_size: ボクセルサイズ (m)
            
        Returns:
            ダウンサンプリング後の (points, colors)
        """
        if not HAS_OPEN3D or len(points) == 0:
            return points, colors
        
        try:
            # Open3D点群オブジェクト作成
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # ボクセルダウンサンプリング実行
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            
            # NumPy配列に変換
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_colors = None
            
            if colors is not None and len(downsampled_pcd.colors) > 0:
                downsampled_colors = np.asarray(downsampled_pcd.colors)
            
            return downsampled_points, downsampled_colors
            
        except Exception as e:
            print(f"Voxel downsampling failed, using original points: {e}")
            return points, colors
    
    def _extract_colors(self, color_frame: Any, valid_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        カラーフレームからカラー情報を抽出
        
        Args:
            color_frame: カラーフレーム
            valid_mask: 有効点のマスク
            
        Returns:
            カラー配列 (N, 3) または None
        """
        try:
            color_data: np.ndarray = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_format = color_frame.get_format()
            
            # フォーマットに応じた処理
            color_image = None
            if color_format == OBFormat.RGB:
                color_image = color_data.reshape((self.depth_intrinsics.height, self.depth_intrinsics.width, 3))
            elif color_format == OBFormat.BGR:
                color_image = color_data.reshape((self.depth_intrinsics.height, self.depth_intrinsics.width, 3))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            elif color_format == OBFormat.MJPG:
                decoded_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if decoded_image is not None:
                    color_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
            
            if color_image is None:
                return None
                
            # サイズが異なる場合はリサイズ
            if color_image.shape[:2] != (self.depth_intrinsics.height, self.depth_intrinsics.width):
                color_image = cv2.resize(
                    color_image,
                    (self.depth_intrinsics.width, self.depth_intrinsics.height)
                )
            
            # 有効点のカラーを抽出（正規化）
            colors: np.ndarray = color_image[valid_mask].astype(np.float32) / 255.0
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
        max_depth: float = 10.0,
        voxel_size: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        numpy配列から直接点群を生成（ボクセルダウンサンプリング対応）
        
        Args:
            depth_array: 深度配列 (H, W)
            color_array: カラー配列 (H, W, 3) オプション
            depth_scale: 深度スケール
            min_depth: 最小深度閾値
            max_depth: 最大深度閾値
            voxel_size: ボクセルサイズ（None でデフォルト使用）
            
        Returns:
            (points, colors): 点群座標とカラー情報
        """
        start_time = time.perf_counter()
        
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
        
        # 入力点数を記録
        input_points = len(points)
        self.performance_stats['last_input_points'] = input_points
        
        # ボクセルダウンサンプリング適用
        if self.enable_voxel_downsampling and input_points > 0:
            downsampling_start = time.perf_counter()
            points, colors = self._apply_voxel_downsampling(
                points, colors, voxel_size or self.voxel_size
            )
            downsampling_time = (time.perf_counter() - downsampling_start) * 1000
            self.performance_stats['total_downsampling_time_ms'] += downsampling_time
            
            # ダウンサンプリング率を計算
            output_points = len(points)
            self.performance_stats['last_output_points'] = output_points
            self.performance_stats['last_downsampling_ratio'] = (
                output_points / input_points if input_points > 0 else 0.0
            )
        else:
            self.performance_stats['last_output_points'] = input_points
            self.performance_stats['last_downsampling_ratio'] = 1.0
        
        # パフォーマンス統計更新
        processing_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_stats(processing_time)
        
        return points, colors
    
    def _update_performance_stats(self, processing_time_ms: float):
        """パフォーマンス統計を更新"""
        self.performance_stats['total_conversions'] += 1
        self.performance_stats['total_time_ms'] += processing_time_ms
        self.performance_stats['average_time_ms'] = (
            self.performance_stats['total_time_ms'] / self.performance_stats['total_conversions']
        )
    
    def set_voxel_size(self, voxel_size: float):
        """
        ボクセルサイズを動的に設定
        
        Args:
            voxel_size: 新しいボクセルサイズ (m)
        """
        self.voxel_size = max(0.001, min(0.1, voxel_size))  # 1mm-10cmの範囲に制限
        print(f"Voxel size updated to: {self.voxel_size*1000:.1f}mm")
    
    def toggle_voxel_downsampling(self):
        """ボクセルダウンサンプリングのON/OFF切り替え"""
        self.enable_voxel_downsampling = not self.enable_voxel_downsampling
        print(f"Voxel downsampling: {'enabled' if self.enable_voxel_downsampling else 'disabled'}")
    
    def toggle_resolution_downsampling(self):
        """解像度ダウンサンプリングのON/OFF切り替え"""
        self.enable_resolution_downsampling = not self.enable_resolution_downsampling
        print(f"Resolution downsampling: {'enabled' if self.enable_resolution_downsampling else 'disabled'} "
              f"(target: {self.target_width}x{self.target_height})")
        
        # 統計情報を更新
        self.performance_stats['resolution_downsampling_enabled'] = self.enable_resolution_downsampling
        
    def set_resolution_downsampling(self, enabled: bool, target_width: int = 424, target_height: int = 240):
        """解像度ダウンサンプリング設定を変更"""
        self.enable_resolution_downsampling = enabled
        self.target_width = target_width
        self.target_height = target_height
        
        if enabled:
            # 解像度比率を再計算
            self.width_ratio = target_width / self.depth_intrinsics.width
            self.height_ratio = target_height / self.depth_intrinsics.height
            
            # ダウンサンプリング後の内部パラメータを更新
            self.downsampled_intrinsics = CameraIntrinsics(
                fx=self.depth_intrinsics.fx * self.width_ratio,
                fy=self.depth_intrinsics.fy * self.height_ratio,
                cx=self.depth_intrinsics.cx * self.width_ratio,
                cy=self.depth_intrinsics.cy * self.height_ratio,
                width=target_width,
                height=target_height
            )
        
        # 統計情報を更新
        self.performance_stats['resolution_downsampling_enabled'] = enabled
        self.performance_stats['resolution_downsampling_ratio'] = f"{target_width}x{target_height}" if enabled else "disabled"
        
        print(f"Resolution downsampling: {'enabled' if enabled else 'disabled'} "
              f"(target: {target_width}x{target_height})")
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計を取得"""
        stats = self.performance_stats.copy()
        if HAS_OPEN3D:
            stats['open3d_available'] = True
            stats['voxel_downsampling_enabled'] = self.enable_voxel_downsampling
            stats['current_voxel_size_mm'] = self.voxel_size * 1000
        else:
            stats['open3d_available'] = False
            stats['voxel_downsampling_enabled'] = False
        
        return stats
    
    def print_performance_stats(self):
        """パフォーマンス統計を表示"""
        stats = self.get_performance_stats()
        print("\n" + "="*50)
        print("Point Cloud Converter Performance Stats")
        print("="*50)
        print(f"Total conversions: {stats['total_conversions']}")
        print(f"Average processing time: {stats['average_time_ms']:.2f}ms")
        print(f"Last input points: {stats['last_input_points']:,}")
        print(f"Last output points: {stats['last_output_points']:,}")
        print(f"Last downsampling ratio: {stats['last_downsampling_ratio']:.3f}")
        
        if stats['open3d_available']:
            print(f"Voxel downsampling: {'Enabled' if stats['voxel_downsampling_enabled'] else 'Disabled'}")
            if stats['voxel_downsampling_enabled']:
                print(f"Current voxel size: {stats['current_voxel_size_mm']:.1f}mm")
                if stats['total_conversions'] > 0:
                    avg_downsampling_time = stats['total_downsampling_time_ms'] / stats['total_conversions']
                    print(f"Average downsampling time: {avg_downsampling_time:.2f}ms")
        else:
            print("Open3D not available - downsampling disabled")
        
        print("="*50)


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
def depth_to_pointcloud(depth_frame: Any, camera_intrinsic: Any, color_frame: Optional[Any] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


# 後方互換性のための便利関数
def create_pointcloud_converter(
    depth_intrinsics: CameraIntrinsics,
    enable_voxel_downsampling: bool = True,
    voxel_size: float = 0.005
) -> PointCloudConverter:
    """
    PointCloudConverterを作成する便利関数
    
    Args:
        depth_intrinsics: 深度カメラの内部パラメータ
        enable_voxel_downsampling: ボクセルダウンサンプリングを有効にするか
        voxel_size: ボクセルサイズ (m)
        
    Returns:
        PointCloudConverterインスタンス
    """
    return PointCloudConverter(
        depth_intrinsics=depth_intrinsics,
        enable_voxel_downsampling=enable_voxel_downsampling,
        voxel_size=voxel_size
    ) 