#!/usr/bin/env python3
"""
pytest共通設定とフィクスチャ

テスト実行時の共通設定、モック、フィクスチャを提供し、
print()依存からlogging/assert依存への移行を支援します。
"""

import pytest
import logging
import sys
import os
import tempfile
import numpy as np
from typing import Generator, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# srcモジュールのパス追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import setup_logging, get_logger
from src.config import get_config
from src.data_types import CameraIntrinsics, FrameData
from src.constants import (
    DEFAULT_DEPTH_WIDTH, DEFAULT_DEPTH_HEIGHT,
    DEFAULT_COLOR_WIDTH, DEFAULT_COLOR_HEIGHT
)

# =============================================================================
# テストロギング設定
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """テスト全体のロギング設定"""
    setup_logging(level="DEBUG")
    logger = get_logger("test")
    logger.info("=== テストセッション開始 ===")
    yield
    logger.info("=== テストセッション終了 ===")


@pytest.fixture
def test_logger():
    """テスト用ロガー"""
    return get_logger("test")


# =============================================================================
# パフォーマンス計測
# =============================================================================

@dataclass
class PerformanceMeasurement:
    """パフォーマンス計測結果"""
    execution_time_ms: float
    memory_usage_mb: float
    operations_per_second: Optional[float] = None
    target_met: bool = False
    
    def log_results(self, logger: logging.Logger, test_name: str, target_ms: float = None):
        """結果をログ出力（print()の代替）"""
        logger.info(f"=== {test_name} パフォーマンス結果 ===")
        logger.info(f"実行時間: {self.execution_time_ms:.3f}ms")
        logger.info(f"メモリ使用量: {self.memory_usage_mb:.2f}MB")
        if self.operations_per_second:
            logger.info(f"処理速度: {self.operations_per_second:.1f} ops/sec")
        if target_ms:
            self.target_met = self.execution_time_ms <= target_ms
            status = "✓ 達成" if self.target_met else "✗ 未達成"
            logger.info(f"目標時間: {target_ms}ms {status}")


@pytest.fixture
def performance_tracker():
    """パフォーマンス計測ユーティリティ"""
    import time
    import psutil
    import gc
    
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            """計測開始"""
            gc.collect()  # GC実行してメモリを正規化
            self.start_time = time.perf_counter()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
        def stop(self, operations_count: int = None) -> PerformanceMeasurement:
            """計測終了"""
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time_ms = (end_time - self.start_time) * 1000
            memory_usage_mb = end_memory - self.start_memory
            
            ops_per_sec = None
            if operations_count and execution_time_ms > 0:
                ops_per_sec = operations_count / (execution_time_ms / 1000)
            
            return PerformanceMeasurement(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                operations_per_second=ops_per_sec
            )
    
    return PerformanceTracker()


# =============================================================================
# テストデータフィクスチャ
# =============================================================================

@pytest.fixture
def sample_camera_intrinsics() -> CameraIntrinsics:
    """テスト用カメラ内部パラメータ"""
    return CameraIntrinsics(
        fx=525.0, fy=525.0,
        cx=320.0, cy=240.0,
        width=DEFAULT_DEPTH_WIDTH,
        height=DEFAULT_DEPTH_HEIGHT
    )


@pytest.fixture
def sample_depth_image() -> np.ndarray:
    """テスト用深度画像"""
    depth = np.random.randint(500, 2000, (DEFAULT_DEPTH_HEIGHT, DEFAULT_DEPTH_WIDTH), dtype=np.uint16)
    # 中央部分に有効な深度値を配置
    center_y, center_x = DEFAULT_DEPTH_HEIGHT // 2, DEFAULT_DEPTH_WIDTH // 2
    depth[center_y-50:center_y+50, center_x-50:center_x+50] = 1000
    return depth


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """テスト用カラー画像"""
    return np.random.randint(0, 255, (DEFAULT_COLOR_HEIGHT, DEFAULT_COLOR_WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_data(sample_depth_image, sample_color_image) -> FrameData:
    """テスト用フレームデータ"""
    return FrameData(
        depth_image=sample_depth_image,
        color_data=sample_color_image,
        timestamp=0.0,
        frame_number=1
    )


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """一時ディレクトリ"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# =============================================================================
# モックフィクスチャ
# =============================================================================

@pytest.fixture
def mock_orbbec_camera():
    """OrbbecCameraのモック"""
    mock_camera = Mock()
    mock_camera.is_connected.return_value = True
    mock_camera.start_streams.return_value = True
    mock_camera.stop_streams.return_value = True
    mock_camera.get_intrinsics.return_value = CameraIntrinsics(
        fx=525.0, fy=525.0, cx=320.0, cy=240.0,
        width=DEFAULT_DEPTH_WIDTH, height=DEFAULT_DEPTH_HEIGHT
    )
    return mock_camera


@pytest.fixture
def mock_mediapipe_hands():
    """MediaPipe Handsのモック"""
    mock_hands = Mock()
    mock_hands.process.return_value = Mock(multi_hand_landmarks=None)
    return mock_hands


@pytest.fixture
def mock_audio_engine():
    """音響エンジンのモック"""
    mock_engine = Mock()
    mock_engine.start.return_value = True
    mock_engine.stop.return_value = True
    mock_engine.create_voice.return_value = "voice_001"
    mock_engine.is_running = True
    return mock_engine


# =============================================================================
# テストスイート選択
# =============================================================================

def pytest_configure(config):
    """pytest設定時に実行"""
    # マーカーの動的追加
    config.addinivalue_line(
        "markers", "unit_slow: 実行時間が長いユニットテスト"
    )
    config.addinivalue_line(
        "markers", "integration_slow: 実行時間が長い統合テスト"
    )


def pytest_collection_modifyitems(config, items):
    """テスト収集時の自動マーカー付与"""
    for item in items:
        # ファイル名ベースのマーカー付与
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_" in item.name and not any(mark in item.nodeid for mark in ["integration", "performance"]):
            item.add_marker(pytest.mark.unit)


# =============================================================================
# アサーション拡張
# =============================================================================

class TestAssertions:
    """拡張アサーション関数"""
    
    @staticmethod
    def assert_performance_target(measurement: PerformanceMeasurement, target_ms: float, operation_name: str):
        """パフォーマンス目標のアサーション（print()の代替）"""
        assert measurement.execution_time_ms <= target_ms, (
            f"{operation_name} パフォーマンス目標未達成: "
            f"{measurement.execution_time_ms:.3f}ms > {target_ms}ms"
        )
    
    @staticmethod
    def assert_array_shape_compatible(arr1: np.ndarray, arr2: np.ndarray, axis: int = None):
        """配列形状の互換性アサーション"""
        if axis is None:
            assert arr1.shape == arr2.shape, f"配列形状不一致: {arr1.shape} != {arr2.shape}"
        else:
            assert arr1.shape[axis] == arr2.shape[axis], (
                f"軸{axis}の配列サイズ不一致: {arr1.shape[axis]} != {arr2.shape[axis]}"
            )
    
    @staticmethod
    def assert_within_tolerance(actual: float, expected: float, tolerance: float, description: str = "値"):
        """許容誤差内アサーション"""
        diff = abs(actual - expected)
        assert diff <= tolerance, (
            f"{description}が許容誤差を超過: |{actual} - {expected}| = {diff} > {tolerance}"
        )


@pytest.fixture
def assert_helper():
    """アサーション拡張のヘルパー"""
    return TestAssertions()


# =============================================================================
# テストデータ生成
# =============================================================================

@pytest.fixture
def mesh_test_data():
    """メッシュテスト用データ生成器"""
    def generate_test_mesh(complexity: str = "simple"):
        """テスト用メッシュデータ生成"""
        if complexity == "simple":
            vertices = np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 1.0, 1.0]
            ], dtype=np.float32)
            triangles = np.array([
                [0, 1, 2], [3, 4, 5], [0, 3, 1], [1, 3, 4], [1, 4, 2], [2, 4, 5]
            ], dtype=np.int32)
        elif complexity == "medium":
            # 10x10グリッドメッシュ
            x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
            z = np.sin(x * np.pi) * np.sin(y * np.pi) * 0.1
            vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1).astype(np.float32)
            triangles = []
            for i in range(9):
                for j in range(9):
                    v0 = i * 10 + j
                    v1 = v0 + 1
                    v2 = v0 + 10
                    v3 = v2 + 1
                    triangles.extend([[v0, v1, v2], [v1, v3, v2]])
            triangles = np.array(triangles, dtype=np.int32)
        else:  # complex
            # 大規模ランダムメッシュ
            n_vertices = 1000
            vertices = np.random.rand(n_vertices, 3).astype(np.float32)
            # Delaunay三角分割の簡易版
            from scipy.spatial import Delaunay
            points_2d = vertices[:, :2]
            tri = Delaunay(points_2d)
            triangles = tri.simplices.astype(np.int32)
            
        return vertices, triangles
    
    return generate_test_mesh


# =============================================================================
# E2Eテスト支援
# =============================================================================

@pytest.fixture(scope="function")
def isolated_test_environment():
    """分離されたテスト環境"""
    # グローバル状態のリセット
    from src.collision.events import reset_global_collision_queue
    from src.resource_manager import ResourceManager
    
    # リソースマネージャーのクリーンアップ
    ResourceManager.get_instance().cleanup_all_resources()
    
    # グローバルキューのリセット
    reset_global_collision_queue()
    
    yield
    
    # テスト後のクリーンアップ
    ResourceManager.get_instance().cleanup_all_resources()
    reset_global_collision_queue() 