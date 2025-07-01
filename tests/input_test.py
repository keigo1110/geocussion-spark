#!/usr/bin/env python3
"""
入力フェーズのユニットテスト & パフォーマンステスト
pytest-benchmark 統合版
"""

import pytest
import numpy as np
import time
from typing import Optional, Tuple
from unittest.mock import Mock, patch

# テスト対象のインポート
from src.input.depth_filter import DepthFilter, FilterType, CudaBilateralFilter
from src.input.pointcloud import PointCloudConverter, NumpyVoxelDownsampler
from src.input.stream import OrbbecCamera, FastResize
from src.types import CameraIntrinsics, FrameData


class TestDepthFilter:
    """深度フィルタのテスト"""
    
    @pytest.fixture
    def sample_depth_image(self):
        """テスト用深度画像生成"""
        return np.random.randint(500, 5000, size=(240, 424), dtype=np.uint16)
    
    @pytest.fixture
    def depth_filter(self):
        """CUDAフィルタ"""
        return DepthFilter(
            filter_types=[FilterType.BILATERAL],
            use_cuda=True,
            enable_multiscale=True
        )
    
    @pytest.fixture
    def cpu_depth_filter(self):
        """CPUフィルタ（比較用）"""
        return DepthFilter(
            filter_types=[FilterType.BILATERAL],
            use_cuda=False
        )
    
    def test_filter_initialization(self):
        """フィルタ初期化テスト"""
        filter = DepthFilter()
        assert filter.filter_types == [FilterType.COMBINED]
        assert filter.bilateral_d == 9
        assert filter.temporal_alpha == 0.3
    
    def test_cuda_bilateral_filter(self, depth_filter, sample_depth_image):
        """CUDAバイラテラルフィルタテスト"""
        filtered = depth_filter.apply_filter(sample_depth_image)
        
        assert filtered.shape == sample_depth_image.shape
        assert filtered.dtype == np.uint16
        assert 'bilateral_cuda' in depth_filter.processing_times or 'bilateral_cpu' in depth_filter.processing_times
    
    def test_ema_temporal_filter(self, sample_depth_image):
        """EMA時間フィルタテスト"""
        filter = DepthFilter(filter_types=[FilterType.TEMPORAL])
        
        # 初回
        result1 = filter.apply_filter(sample_depth_image)
        assert result1.shape == sample_depth_image.shape
        assert filter.ema_initialized
        
        # 2回目（EMA適用）
        result2 = filter.apply_filter(sample_depth_image)
        assert result2.shape == sample_depth_image.shape
        assert 'temporal_ema' in filter.processing_times
    
    def test_performance_stats(self, depth_filter, sample_depth_image):
        """パフォーマンス統計テスト"""
        depth_filter.apply_filter(sample_depth_image)
        stats = depth_filter.get_performance_stats()
        
        assert 'total' in stats
        assert stats['total'] > 0
    
    @pytest.mark.benchmark(group="depth_filter")
    def test_benchmark_cuda_bilateral(self, benchmark, depth_filter, sample_depth_image):
        """CUDAバイラテラルフィルタベンチマーク"""
        result = benchmark(depth_filter.apply_filter, sample_depth_image)
        assert result is not None
    
    @pytest.mark.benchmark(group="depth_filter")
    def test_benchmark_cpu_bilateral(self, benchmark, cpu_depth_filter, sample_depth_image):
        """CPUバイラテラルフィルタベンチマーク（比較用）"""
        result = benchmark(cpu_depth_filter.apply_filter, sample_depth_image)
        assert result is not None


class TestNumpyVoxelDownsampler:
    """NumPyボクセルダウンサンプラーのテスト"""
    
    @pytest.fixture
    def sample_points(self):
        """テスト用点群生成"""
        return np.random.rand(10000, 3).astype(np.float32)
    
    @pytest.fixture
    def sample_colors(self):
        """テスト用カラー情報生成"""
        return np.random.rand(10000, 3).astype(np.float32)
    
    def test_voxel_downsample_first_strategy(self, sample_points, sample_colors):
        """first戦略のボクセルダウンサンプリングテスト"""
        voxel_size = 0.01
        
        downsampled_points, downsampled_colors = NumpyVoxelDownsampler.voxel_downsample_numpy(
            sample_points, sample_colors, voxel_size, "first"
        )
        
        assert len(downsampled_points) <= len(sample_points)
        assert len(downsampled_colors) == len(downsampled_points)
        assert downsampled_points.dtype == np.float32
        assert downsampled_colors.dtype == np.float32
    
    def test_voxel_downsample_average_strategy(self, sample_points, sample_colors):
        """average戦略のボクセルダウンサンプリングテスト"""
        voxel_size = 0.01
        
        downsampled_points, downsampled_colors = NumpyVoxelDownsampler.voxel_downsample_numpy(
            sample_points, sample_colors, voxel_size, "average"
        )
        
        assert len(downsampled_points) <= len(sample_points)
        assert len(downsampled_colors) == len(downsampled_points)
    
    def test_empty_points(self):
        """空の点群処理テスト"""
        empty_points = np.empty((0, 3), dtype=np.float32)
        
        result_points, result_colors = NumpyVoxelDownsampler.voxel_downsample_numpy(
            empty_points, None, 0.01, "first"
        )
        
        assert len(result_points) == 0
        assert result_colors is None
    
    @pytest.mark.benchmark(group="voxel_downsampling")
    def test_benchmark_numpy_voxel_first(self, benchmark, sample_points, sample_colors):
        """NumPyボクセルダウンサンプリング（first）ベンチマーク"""
        result = benchmark(
            NumpyVoxelDownsampler.voxel_downsample_numpy,
            sample_points, sample_colors, 0.005, "first"
        )
        assert result[0] is not None
    
    @pytest.mark.benchmark(group="voxel_downsampling")
    def test_benchmark_numpy_voxel_average(self, benchmark, sample_points, sample_colors):
        """NumPyボクセルダウンサンプリング（average）ベンチマーク"""
        result = benchmark(
            NumpyVoxelDownsampler.voxel_downsample_numpy,
            sample_points, sample_colors, 0.005, "average"
        )
        assert result[0] is not None


class TestPointCloudConverter:
    """点群コンバーターのテスト"""
    
    @pytest.fixture
    def camera_intrinsics(self):
        """テスト用カメラ内部パラメータ"""
        return CameraIntrinsics(
            fx=421.0, fy=421.0, cx=212.0, cy=120.0,
            width=424, height=240
        )
    
    @pytest.fixture
    def depth_array(self):
        """テスト用深度配列"""
        return np.random.randint(500, 5000, size=(240, 424), dtype=np.uint16)
    
    @pytest.fixture
    def converter_numpy(self, camera_intrinsics):
        """NumPy版点群コンバーター"""
        return PointCloudConverter(
            camera_intrinsics,
            use_numpy_voxel=True,
            color_strategy="first"
        )
    
    @pytest.fixture
    def converter_open3d(self, camera_intrinsics):
        """Open3D版点群コンバーター（比較用）"""
        return PointCloudConverter(
            camera_intrinsics,
            use_numpy_voxel=False
        )
    
    def test_coefficients_caching(self, camera_intrinsics):
        """メッシュグリッド係数キャッシュテスト"""
        converter = PointCloudConverter(camera_intrinsics)
        
        # 初回取得
        x_coeff1, y_coeff1 = converter._get_cached_coefficients(424, 240)
        assert x_coeff1.shape == (240, 424)
        assert y_coeff1.shape == (240, 424)
        
        # 2回目取得（キャッシュから）
        x_coeff2, y_coeff2 = converter._get_cached_coefficients(424, 240)
        assert np.array_equal(x_coeff1, x_coeff2)
        assert np.array_equal(y_coeff1, y_coeff2)
        
        # キャッシュ確認
        assert (424, 240) in converter._coeff_cache
    
    def test_numpy_to_pointcloud(self, converter_numpy, depth_array):
        """NumPy配列から点群生成テスト"""
        points, colors = converter_numpy.numpy_to_pointcloud(depth_array)
        
        assert points.ndim == 2
        assert points.shape[1] == 3
        assert len(points) > 0
        assert colors is None  # カラー配列なし
    
    def test_voxel_downsampling_numpy(self, converter_numpy, depth_array):
        """NumPyボクセルダウンサンプリングテスト"""
        converter_numpy.enable_voxel_downsampling = True
        points, colors = converter_numpy.numpy_to_pointcloud(depth_array)
        
        # ダウンサンプリング統計確認
        stats = converter_numpy.get_performance_stats()
        assert stats['last_input_points'] > 0
        assert stats['last_output_points'] <= stats['last_input_points']
        assert 0 <= stats['last_downsampling_ratio'] <= 1
    
    @pytest.mark.benchmark(group="pointcloud_conversion")
    def test_benchmark_numpy_pointcloud(self, benchmark, converter_numpy, depth_array):
        """NumPy点群変換ベンチマーク"""
        result = benchmark(converter_numpy.numpy_to_pointcloud, depth_array)
        assert result[0] is not None
    
    @pytest.mark.benchmark(group="pointcloud_conversion") 
    def test_benchmark_open3d_pointcloud(self, benchmark, converter_open3d, depth_array):
        """Open3D点群変換ベンチマーク（比較用）"""
        result = benchmark(converter_open3d.numpy_to_pointcloud, depth_array)
        assert result[0] is not None


class TestFastResize:
    """高速リサイズのテスト"""
    
    @pytest.fixture
    def fast_resize(self):
        """高速リサイズインスタンス"""
        return FastResize(use_cuda=True, enable_roi=True)
    
    @pytest.fixture
    def cpu_resize(self):
        """CPU版リサイズ（比較用）"""
        return FastResize(use_cuda=False, enable_roi=False)
    
    @pytest.fixture
    def sample_depth(self):
        """テスト用深度画像"""
        return np.random.randint(500, 5000, size=(480, 640), dtype=np.uint16)
    
    @pytest.fixture
    def sample_color(self):
        """テスト用カラー画像"""
        return np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    
    def test_resize_depth_roi(self, fast_resize, sample_depth):
        """ROI深度リサイズテスト"""
        target_size = (320, 240)  # 1/2ダウンサンプリング
        
        resized = fast_resize.resize_depth(sample_depth, target_size)
        
        assert resized.shape == (240, 320)
        assert resized.dtype == np.uint16
    
    def test_resize_color(self, fast_resize, sample_color):
        """カラーリサイズテスト"""
        target_size = (320, 240)
        
        resized = fast_resize.resize_color(sample_color, target_size)
        
        assert resized.shape == (240, 320, 3)
        assert resized.dtype == np.uint8
    
    @pytest.mark.benchmark(group="resize")
    def test_benchmark_cuda_depth_resize(self, benchmark, fast_resize, sample_depth):
        """CUDA深度リサイズベンチマーク"""
        target_size = (212, 120)
        
        # GPU初期化
        fast_resize.initialize((480, 640), (120, 212))
        
        result = benchmark(fast_resize.resize_depth, sample_depth, target_size)
        assert result.shape == (120, 212)
    
    @pytest.mark.benchmark(group="resize")
    def test_benchmark_cpu_depth_resize(self, benchmark, cpu_resize, sample_depth):
        """CPU深度リサイズベンチマーク（比較用）"""
        target_size = (212, 120)
        result = benchmark(cpu_resize.resize_depth, sample_depth, target_size)
        assert result.shape == (120, 212)


class TestInputPhaseIntegration:
    """入力フェーズ統合テスト"""
    
    @pytest.fixture
    def mock_frame_data(self):
        """モックフレームデータ"""
        mock_frame = Mock()
        mock_frame.get_data.return_value = np.random.randint(0, 5000, size=240*424, dtype=np.uint16).tobytes()
        return mock_frame
    
    @pytest.fixture  
    def camera_intrinsics(self):
        """カメラ内部パラメータ"""
        return CameraIntrinsics(
            fx=421.0, fy=421.0, cx=212.0, cy=120.0,
            width=424, height=240
        )
    
    def test_full_pipeline_performance(self, camera_intrinsics):
        """フル入力パイプラインパフォーマンステスト"""
        # 各コンポーネントを初期化
        depth_filter = DepthFilter(
            filter_types=[FilterType.BILATERAL],
            use_cuda=True,
            enable_multiscale=True
        )
        
        converter = PointCloudConverter(
            camera_intrinsics,
            use_numpy_voxel=True,
            enable_voxel_downsampling=True
        )
        
        fast_resize = FastResize(use_cuda=True, enable_roi=True)
        
        # テストデータ
        depth_image = np.random.randint(500, 5000, size=(240, 424), dtype=np.uint16)
        
        # パイプライン実行
        start_time = time.perf_counter()
        
        # 1. フィルタリング
        filtered_depth = depth_filter.apply_filter(depth_image)
        
        # 2. リサイズ（オプション）
        resized_depth = fast_resize.resize_depth(filtered_depth, (212, 120))
        
        # 3. 点群変換
        points, colors = converter.numpy_to_pointcloud(resized_depth)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # 性能目標検証
        print(f"Total input pipeline time: {total_time:.2f} ms")
        print(f"Input points: {len(points)}")
        print(f"Filter time: {depth_filter.processing_times.get('total', 0):.2f} ms")
        
        # 入力フェーズ目標：5ms以内
        assert total_time < 10.0  # 多少の余裕を持たせたテスト閾値
        assert len(points) > 0
    
    @pytest.mark.benchmark(group="input_pipeline")
    def test_benchmark_full_pipeline(self, benchmark, camera_intrinsics):
        """フル入力パイプラインベンチマーク"""
        def pipeline():
            depth_filter = DepthFilter(use_cuda=True, enable_multiscale=True)
            converter = PointCloudConverter(camera_intrinsics, use_numpy_voxel=True)
            depth_image = np.random.randint(500, 5000, size=(240, 424), dtype=np.uint16)
            
            filtered = depth_filter.apply_filter(depth_image)
            points, colors = converter.numpy_to_pointcloud(filtered)
            return points, colors
        
        result = benchmark(pipeline)
        assert result[0] is not None


# パフォーマンステスト設定
def pytest_configure(config):
    """pytest設定"""
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark for performance measurement"
    )


# テスト用データ生成ヘルパー
def generate_test_depth_image(width: int = 424, height: int = 240) -> np.ndarray:
    """テスト用深度画像生成"""
    return np.random.randint(500, 5000, size=(height, width), dtype=np.uint16)


def generate_test_color_image(width: int = 424, height: int = 240) -> np.ndarray:
    """テスト用カラー画像生成"""
    return np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8) 