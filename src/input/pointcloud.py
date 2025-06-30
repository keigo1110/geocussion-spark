#!/usr/bin/env python3
"""
点群変換処理モジュール
既存 depth_to_pointcloud 関数のリファクタリング版
numpy 最適化とゼロコピー転送対応 + 高速ボクセルダウンサンプリング
"""

import time
from typing import Optional, Tuple, Union, Any, Dict
import numpy as np
import cv2

try:
    # vendor配下から直接import
    import sys
    import os
    vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vendor', 'pyorbbecsdk')
    sys.path.insert(0, vendor_path)
    from pyorbbecsdk import OBFormat
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    # テスト用のモック定義（types.pyのOBFormatと統合）
    from ..types import OBFormat
    HAS_OPEN3D = False

from ..types import CameraIntrinsics, FrameData
from ..collision.optimization import ArrayPool
from src import get_logger

logger = get_logger(__name__)


class NumpyVoxelDownsampler:
    """NumPy ベース高速ボクセルダウンサンプリング"""
    
    @staticmethod
    def voxel_downsample_numpy(
        points: np.ndarray, 
        colors: Optional[np.ndarray], 
        voxel_size: float,
        color_strategy: str = "first"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        NumPy ベースの高速ボクセルダウンサンプリング
        
        Args:
            points: 入力点群 (N, 3)
            colors: 入力カラー (N, 3) または None
            voxel_size: ボクセルサイズ
            color_strategy: "first" (初回値) または "average" (平均値)
            
        Returns:
            ダウンサンプリング後の (points, colors)
        """
        if len(points) == 0:
            return points, colors
        
        # ボクセル座標に変換（整数化）
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        
        # unique な組み合わせを取得（高速化のためreturn_inverse=Trueのみ使用）
        unique_voxels, inverse_indices = np.unique(
            voxel_coords, axis=0, return_inverse=True
        )
        
        # 各ボクセルの代表点を選択
        if color_strategy == "first":
            # 最初に見つかった点を使用（高速）
            # np.unique は安定ソートなので、最初のインデックスを効率的に取得
            first_indices = np.zeros(len(unique_voxels), dtype=np.int32)
            seen = set()
            for i, voxel_idx in enumerate(inverse_indices):
                if voxel_idx not in seen:
                    first_indices[voxel_idx] = i
                    seen.add(voxel_idx)
            
            downsampled_points = points[first_indices]
            downsampled_colors = colors[first_indices] if colors is not None else None
            
        elif color_strategy == "average":
            # 各ボクセル内の平均値を計算（高品質）
            n_voxels = len(unique_voxels)
            downsampled_points = np.zeros((n_voxels, 3), dtype=np.float32)
            downsampled_colors = np.zeros((n_voxels, 3), dtype=np.float32) if colors is not None else None
            
            # ベクトル化された平均計算
            for voxel_idx in range(n_voxels):
                mask = (inverse_indices == voxel_idx)
                downsampled_points[voxel_idx] = np.mean(points[mask], axis=0)
                if colors is not None:
                    downsampled_colors[voxel_idx] = np.mean(colors[mask], axis=0)
        else:
            raise ValueError(f"Unknown color_strategy: {color_strategy}")
        
        return downsampled_points, downsampled_colors


class PointCloudConverter:
    """深度フレームから点群への変換クラス（高速化対応）"""
    
    def __init__(
        self, 
        depth_intrinsics: CameraIntrinsics,
        depth_scale: float = 1000.0,
        enable_voxel_downsampling: bool = True,
        voxel_size: float = 0.005,  # 5mm voxel size for good balance
        enable_resolution_downsampling: bool = False,
        target_width: int = 424,
        target_height: int = 240,
        use_numpy_voxel: bool = True,
        color_strategy: str = "first"
    ):
        """
        初期化
        
        Args:
            use_numpy_voxel: NumPy版ボクセルダウンサンプリングを使用するか
            color_strategy: "first" または "average" 
        """
        self.depth_intrinsics = depth_intrinsics
        self.depth_scale = depth_scale
        self.enable_voxel_downsampling = enable_voxel_downsampling
        self.voxel_size = voxel_size
        self.use_numpy_voxel = use_numpy_voxel
        self.color_strategy = color_strategy
        
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
        
        # メッシュグリッド係数キャッシュ（T-INP-003対応）
        self._coeff_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        
        # ArrayPool for memory optimization
        self.array_pool = ArrayPool()
        
        # 3D座標計算用の係数を事前計算（遅延初期化）
        self.x_coeff = None
        self.y_coeff = None
        
        # パフォーマンス統計
        self.performance_stats = {
            'total_conversions': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_input_points': 0,
            'last_output_points': 0,
            'last_downsampling_ratio': 0.0,
            'total_downsampling_time_ms': 0.0,
            'numpy_voxel_enabled': use_numpy_voxel,
            'resolution_downsampling_enabled': enable_resolution_downsampling,
            'resolution_downsampling_ratio': f"{target_width}x{target_height}" if enable_resolution_downsampling else "disabled"
        }
        
        logger.info(f"PointCloudConverter initialized: numpy_voxel={use_numpy_voxel}, "
                   f"color_strategy={color_strategy}")
        
    def _get_cached_coefficients(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        メッシュグリッド係数のキャッシュ取得（T-INP-003対応）
        
        Args:
            width: 幅
            height: 高さ
            
        Returns:
            (x_coeff, y_coeff): 3D座標計算用係数
        """
        cache_key = (width, height)
        
        if cache_key not in self._coeff_cache:
            # ピクセル座標のメッシュグリッドを作成
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # 内部パラメータを決定（解像度ダウンサンプリング対応）
            if self.enable_resolution_downsampling and (width, height) == (self.target_width, self.target_height):
                intrinsics = self.downsampled_intrinsics
            else:
                intrinsics = self.depth_intrinsics
            
            # X, Y座標の計算係数（Z値との乗算で3D座標が得られる）
            x_coeff = (u - intrinsics.cx) / intrinsics.fx
            y_coeff = -(v - intrinsics.cy) / intrinsics.fy  # Y軸反転
            
            self._coeff_cache[cache_key] = (x_coeff, y_coeff)
            logger.debug(f"Cached meshgrid coefficients for {width}x{height}")
        
        return self._coeff_cache[cache_key]
    
    def _precompute_coefficients(self):
        """3D座標計算用の係数を事前計算（統合版・lazy-init対応）"""
        if self.x_coeff is not None and self.y_coeff is not None:
            # 既に計算済みの場合はスキップ
            return
            
        # デフォルト解像度用の係数をキャッシュから取得
        width, height = self.depth_intrinsics.width, self.depth_intrinsics.height
        self.x_coeff, self.y_coeff = self._get_cached_coefficients(width, height)
        
        logger.debug(f"Precomputed default 3D projection coefficients for {width}x{height}")
    
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
        
        # numpy配列変換を直接呼び出し（高速化）
        points, colors = self.numpy_to_pointcloud(
            depth_image, color_frame, depth_scale, min_depth, max_depth, voxel_size
        )
        
        return points, colors
    
    def _apply_voxel_downsampling_numpy(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        voxel_size: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        NumPy 高速ボクセルダウンサンプリングを適用
        
        Args:
            points: 入力点群 (N, 3)
            colors: 入力カラー情報 (N, 3) または None
            voxel_size: ボクセルサイズ (m)
            
        Returns:
            ダウンサンプリング後の (points, colors)
        """
        if len(points) == 0:
            return points, colors
        
        try:
            return NumpyVoxelDownsampler.voxel_downsample_numpy(
                points, colors, voxel_size, self.color_strategy
            )
        except Exception as e:
            logger.warning(f"NumPy voxel downsampling error: {e}")
            return points, colors
    
    def _apply_voxel_downsampling_open3d(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        voxel_size: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Open3D ボクセルダウンサンプリングを適用（フォールバック用）
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
            
        except (ImportError, AttributeError) as e:
            # Open3D関連のエラー（復旧可能）
            logger.warning(f"Open3D voxel downsampling unavailable: {e}")
            return points, colors
        except (MemoryError, ValueError) as e:
            # メモリ不足やデータエラー（致命的）
            logger.error(f"Critical error during Open3D voxel downsampling: {e}")
            raise RuntimeError(f"Open3D voxel downsampling failed: {e}")
        except Exception as e:
            # その他のエラー（警告して元の点群を返す）
            logger.warning(f"Unexpected Open3D voxel downsampling error: {e}")
            return points, colors
    
    def _apply_voxel_downsampling(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        voxel_size: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        ボクセルダウンサンプリングを適用（NumPy/Open3D自動切替）
        """
        if self.use_numpy_voxel:
            return self._apply_voxel_downsampling_numpy(points, colors, voxel_size)
        else:
            return self._apply_voxel_downsampling_open3d(points, colors, voxel_size)
    
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
            
        except (AttributeError, ValueError, cv2.error) as e:
            # 画像処理エラー（デバッグレベル）
            logger.debug(f"Color extraction error: {e}")
            return None
        except Exception as e:
            # 予期しないエラー
            logger.warning(f"Unexpected color extraction error: {e}")
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
        
        # 入力配列の解像度を取得
        input_height, input_width = depth_array.shape
        
        # 適切な係数を取得（キャッシュ機能付き）
        x_coeff, y_coeff = self._get_cached_coefficients(input_width, input_height)
        
        # 深度スケール適用
        scale = depth_scale if depth_scale is not None else self.depth_scale
        z = depth_array.astype(np.float32) / scale
        
        # 深度範囲フィルタ
        valid = (z > min_depth) & (z < max_depth)
        
        # 3D座標計算
        x = x_coeff * z
        y = y_coeff * z  # 既にy_coeffでマイナス符号適用済み
        
        # 点群作成
        points = np.column_stack([
            x[valid],
            y[valid],
            z[valid]
        ])
        
        # カラー情報
        colors = None
        if color_array is not None:
            # ndarray とフレームオブジェクトを判別
            if isinstance(color_array, np.ndarray):
                # NumPy 配列の場合
                if color_array.shape[:2] != depth_array.shape:
                    color_array = cv2.resize(color_array, (depth_array.shape[1], depth_array.shape[0]))
                colors = color_array[valid].astype(np.float32) / 255.0
            elif hasattr(color_array, 'get_data'):
                # OrbbecSDK の ColorFrame オブジェクトの場合
                colors = self._extract_colors(color_array, valid)
            else:
                logger.debug("Unsupported color_array type: %s", type(color_array))
        
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
        logger.info(f"Voxel size updated to: {self.voxel_size*1000:.1f}mm")
    
    def toggle_voxel_downsampling(self):
        """ボクセルダウンサンプリングのON/OFF切り替え"""
        self.enable_voxel_downsampling = not self.enable_voxel_downsampling
        logger.info(f"Voxel downsampling: {'enabled' if self.enable_voxel_downsampling else 'disabled'}")
    
    def toggle_resolution_downsampling(self):
        """解像度ダウンサンプリングのON/OFF切り替え"""
        self.enable_resolution_downsampling = not self.enable_resolution_downsampling
        logger.info(f"Resolution downsampling: {'enabled' if self.enable_resolution_downsampling else 'disabled'} "
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
        
        logger.info(f"Resolution downsampling: {'enabled' if enabled else 'disabled'} "
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
        logger.debug("="*50)
        logger.debug("Point Cloud Converter Performance Stats")
        logger.debug("="*50)
        logger.debug(f"Total conversions: {stats['total_conversions']}")
        logger.debug(f"Average processing time: {stats['average_time_ms']:.2f}ms")
        logger.debug(f"Last input points: {stats['last_input_points']:,}")
        logger.debug(f"Last output points: {stats['last_output_points']:,}")
        logger.debug(f"Last downsampling ratio: {stats['last_downsampling_ratio']:.3f}")
        
        if stats['open3d_available']:
            logger.debug(f"Voxel downsampling: {'Enabled' if stats['voxel_downsampling_enabled'] else 'Disabled'}")
            if stats['voxel_downsampling_enabled']:
                logger.debug(f"Current voxel size: {stats['current_voxel_size_mm']:.1f}mm")
                if stats['total_conversions'] > 0:
                    avg_downsampling_time = stats['total_downsampling_time_ms'] / stats['total_conversions']
                    logger.debug(f"Average downsampling time: {avg_downsampling_time:.2f}ms")
        else:
            logger.debug("Open3D not available - downsampling disabled")
        
        logger.debug("="*50)


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