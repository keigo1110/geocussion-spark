#!/usr/bin/env python3
"""
入力フェーズの単体テスト
基本機能の動作確認
"""

import unittest
import numpy as np
import sys
import os

# テスト対象のモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.input.stream import CameraIntrinsics, FrameData
from src.input.pointcloud import PointCloudConverter
from src.input.depth_filter import DepthFilter, FilterType


class TestCameraIntrinsics(unittest.TestCase):
    """CameraIntrinsicsクラスのテスト"""
    
    def test_camera_intrinsics_creation(self):
        """カメラ内部パラメータの作成テスト"""
        intrinsics = CameraIntrinsics(
            fx=525.0,
            fy=525.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480
        )
        
        self.assertEqual(intrinsics.fx, 525.0)
        self.assertEqual(intrinsics.fy, 525.0)
        self.assertEqual(intrinsics.cx, 320.0)
        self.assertEqual(intrinsics.cy, 240.0)
        self.assertEqual(intrinsics.width, 640)
        self.assertEqual(intrinsics.height, 480)


class TestPointCloudConverter(unittest.TestCase):
    """PointCloudConverterクラスのテスト"""
    
    def setUp(self):
        """テスト用の設定"""
        self.intrinsics = CameraIntrinsics(
            fx=525.0,
            fy=525.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480
        )
        self.converter = PointCloudConverter(self.intrinsics)
    
    def test_converter_initialization(self):
        """コンバーターの初期化テスト"""
        self.assertIsNotNone(self.converter)
        self.assertEqual(self.converter.depth_intrinsics.width, 640)
        self.assertEqual(self.converter.depth_intrinsics.height, 480)
        
        # メッシュグリッドの事前計算確認
        self.assertIsNotNone(self.converter.pixel_x)
        self.assertIsNotNone(self.converter.pixel_y)
        self.assertIsNotNone(self.converter.x_coeff)
        self.assertIsNotNone(self.converter.y_coeff)
    
    def test_numpy_to_pointcloud(self):
        """numpy配列から点群への変換テスト"""
        # テスト用の深度画像作成（中央に正方形の物体）
        depth_array = np.zeros((480, 640), dtype=np.uint16)
        depth_array[200:280, 280:360] = 1000  # 1m の距離にある物体
        
        # カラー画像作成
        color_array = np.zeros((480, 640, 3), dtype=np.uint8)
        color_array[200:280, 280:360] = [255, 0, 0]  # 赤い物体
        
        # 点群変換
        points, colors = self.converter.numpy_to_pointcloud(
            depth_array, 
            color_array,
            min_depth=0.5,
            max_depth=2.0
        )
        
        # 結果検証
        self.assertGreater(len(points), 0)
        self.assertEqual(len(points), len(colors))
        self.assertEqual(points.shape[1], 3)  # x, y, z
        self.assertEqual(colors.shape[1], 3)  # r, g, b
        
        # 深度値の検証（おおよそ1mの距離）
        z_values = points[:, 2]
        self.assertTrue(np.all(z_values >= 0.9))
        self.assertTrue(np.all(z_values <= 1.1))
    
    def test_update_intrinsics(self):
        """内部パラメータ更新テスト"""
        new_intrinsics = CameraIntrinsics(
            fx=600.0,
            fy=600.0,
            cx=400.0,
            cy=300.0,
            width=800,
            height=600
        )
        
        self.converter.update_intrinsics(new_intrinsics)
        
        self.assertEqual(self.converter.depth_intrinsics.fx, 600.0)
        self.assertEqual(self.converter.depth_intrinsics.width, 800)
        
        # メッシュグリッドが更新されていることを確認
        self.assertEqual(self.converter.pixel_x.shape, (600, 800))


class TestDepthFilter(unittest.TestCase):
    """DepthFilterクラスのテスト"""
    
    def setUp(self):
        """テスト用の設定"""
        self.filter = DepthFilter(
            filter_types=[FilterType.MEDIAN],
            median_kernel_size=3,
            min_valid_depth=0.1,
            max_valid_depth=5.0
        )
    
    def test_filter_initialization(self):
        """フィルタの初期化テスト"""
        self.assertIsNotNone(self.filter)
        self.assertEqual(self.filter.median_kernel_size, 3)
        self.assertEqual(len(self.filter.filter_types), 1)
        self.assertEqual(self.filter.filter_types[0], FilterType.MEDIAN)
    
    def test_median_filter(self):
        """メディアンフィルタのテスト"""
        # ノイズを含む深度画像を作成
        depth_image = np.full((100, 100), 1000, dtype=np.uint16)
        
        # ランダムなノイズを追加
        noise_positions = np.random.choice(10000, 100, replace=False)
        flat_image = depth_image.flatten()
        flat_image[noise_positions] = 0  # 無効なピクセル
        depth_image = flat_image.reshape((100, 100))
        
        # フィルタ適用
        filtered_image = self.filter.apply_filter(depth_image)
        
        # 結果検証
        self.assertEqual(filtered_image.shape, depth_image.shape)
        self.assertEqual(filtered_image.dtype, np.uint16)
        
        # ノイズが減少していることを確認
        valid_original = np.count_nonzero(depth_image)
        valid_filtered = np.count_nonzero(filtered_image)
        self.assertGreaterEqual(valid_filtered, valid_original)
    
    def test_combined_filter(self):
        """複合フィルタのテスト"""
        combined_filter = DepthFilter(
            filter_types=[FilterType.COMBINED],
            temporal_alpha=0.5
        )
        
        # 連続する3フレームをシミュレート
        base_depth = np.full((50, 50), 2000, dtype=np.uint16)
        
        for i in range(3):
            # 若干の変動を加える
            depth_image = base_depth + np.random.randint(-50, 50, (50, 50)).astype(np.uint16)
            filtered_image = combined_filter.apply_filter(depth_image)
            
            self.assertEqual(filtered_image.shape, depth_image.shape)
            self.assertEqual(filtered_image.dtype, np.uint16)
        
        # パフォーマンス統計が記録されていることを確認
        stats = combined_filter.get_performance_stats()
        self.assertIn('combined', stats)
        self.assertGreater(stats['combined'], 0)
    
    def test_parameter_update(self):
        """パラメータ動的更新のテスト"""
        original_alpha = self.filter.temporal_alpha
        
        self.filter.update_parameters(temporal_alpha=0.8)
        self.assertEqual(self.filter.temporal_alpha, 0.8)
        self.assertNotEqual(self.filter.temporal_alpha, original_alpha)


class TestIntegration(unittest.TestCase):
    """統合テスト"""
    
    def test_full_pipeline(self):
        """入力フェーズの完全パイプラインテスト"""
        # カメラ内部パラメータ
        intrinsics = CameraIntrinsics(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0,
            width=640, height=480
        )
        
        # コンポーネント初期化
        converter = PointCloudConverter(intrinsics)
        depth_filter = DepthFilter(filter_types=[FilterType.COMBINED])
        
        # テスト用深度画像（階段状の構造）
        depth_array = np.zeros((480, 640), dtype=np.uint16)
        for i in range(5):
            y_start = i * 90
            y_end = y_start + 80
            depth_array[y_start:y_end, :] = 500 + i * 200  # 0.5m から 1.3m
        
        # ノイズ追加
        noise_mask = np.random.random((480, 640)) < 0.05
        depth_array[noise_mask] = 0
        
        # パイプライン実行
        # 1. フィルタ適用
        filtered_depth = depth_filter.apply_filter(depth_array)
        
        # 2. 点群変換
        points, colors = converter.numpy_to_pointcloud(
            filtered_depth,
            min_depth=0.3,
            max_depth=2.0
        )
        
        # 結果検証
        self.assertGreater(len(points), 1000)  # 十分な点数があること
        self.assertEqual(points.shape[1], 3)
        
        # 深度の範囲確認
        z_values = points[:, 2]
        self.assertTrue(np.all(z_values >= 0.3))
        self.assertTrue(np.all(z_values <= 2.0))
        
        # 階段状構造が保持されていることを確認
        unique_z = np.unique(np.round(z_values, 1))
        self.assertGreaterEqual(len(unique_z), 3)  # 複数の深度レベル
        
        print(f"Pipeline test completed: {len(points)} points generated")
        print(f"Depth range: {z_values.min():.3f} - {z_values.max():.3f}m")
        print(f"Filter performance: {depth_filter.get_performance_stats()}")


if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2) 