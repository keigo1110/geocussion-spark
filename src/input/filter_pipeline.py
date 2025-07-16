#!/usr/bin/env python3
"""
統合フィルタパイプライン
median + bilateral + temporal フィルタを単一カーネルで高速処理
speedup.md対応: 12-18ms → 3-4ms の改善目標
"""

import time
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np
import cv2
from dataclasses import dataclass

try:
    from numba import jit, cuda, uint16, float32, int32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src import get_logger
from src.performance.profiler import measure_performance, profile_phase

logger = get_logger(__name__)


@dataclass
class FilterConfig:
    """フィルタ設定"""
    enable_median: bool = True
    enable_bilateral: bool = True
    enable_temporal: bool = True
    median_kernel_size: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    temporal_alpha: float = 0.8
    min_depth: float = 0.1
    max_depth: float = 10.0


@dataclass
class PipelineStats:
    """パイプライン統計"""
    total_frames: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    cuda_accelerated: int = 0
    numba_accelerated: int = 0
    cpu_fallback: int = 0
    unified_kernel_usage: int = 0


class FilterStrategy(Enum):
    """フィルタ戦略"""
    UNIFIED_CUDA = "unified_cuda"        # 統合CUDAカーネル
    UNIFIED_NUMBA = "unified_numba"      # 統合Numbaカーネル
    SEQUENTIAL_CUDA = "sequential_cuda"   # 逐次CUDA処理
    SEQUENTIAL_CPU = "sequential_cpu"     # 逐次CPU処理


class UnifiedFilterPipeline:
    """統合フィルタパイプライン"""
    
    def __init__(self, config: Optional[FilterConfig] = None, strategy: Optional[FilterStrategy] = None):
        self.config = config or FilterConfig()
        self.stats = PipelineStats()
        
        # 戦略自動選択
        if strategy is None:
            strategy = self._auto_select_strategy()
        self.strategy = strategy
        
        # 初期化
        self._initialize_filters()
        self._temporal_state: Optional[np.ndarray] = None
        
        logger.info(f"Unified filter pipeline initialized with strategy: {strategy.value}")
    
    def _auto_select_strategy(self) -> FilterStrategy:
        """最適な戦略を自動選択"""
        # CUDA利用可能性チェック
        cuda_available = self._check_cuda_availability()
        
        if cuda_available and HAS_NUMBA:
            logger.info("CUDA + Numba available - using unified CUDA strategy")
            return FilterStrategy.UNIFIED_CUDA
        elif HAS_NUMBA:
            logger.info("Numba available - using unified Numba strategy")
            return FilterStrategy.UNIFIED_NUMBA
        elif cuda_available:
            logger.info("CUDA available - using sequential CUDA strategy")
            return FilterStrategy.SEQUENTIAL_CUDA
        else:
            logger.info("Fallback to sequential CPU strategy")
            return FilterStrategy.SEQUENTIAL_CPU
    
    def _check_cuda_availability(self) -> bool:
        """CUDA利用可能性をチェック"""
        try:
            cv2.cuda.getCudaEnabledDeviceCount()
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0 and HAS_NUMBA:
                # Numba CUDA も確認
                try:
                    cuda.detect()
                    return True
                except Exception:
                    return False
            return device_count > 0
        except (cv2.error, AttributeError):
            return False
    
    def _initialize_filters(self):
        """フィルタコンポーネント初期化"""
        if self.strategy == FilterStrategy.UNIFIED_CUDA:
            self._initialize_unified_cuda()
        elif self.strategy == FilterStrategy.UNIFIED_NUMBA:
            self._initialize_unified_numba()
        elif self.strategy == FilterStrategy.SEQUENTIAL_CUDA:
            self._initialize_sequential_cuda()
        else:
            self._initialize_sequential_cpu()
    
    def _initialize_unified_cuda(self):
        """統合CUDAカーネル初期化"""
        logger.info("Initializing unified CUDA kernels...")
        # GPU メモリプール
        self._gpu_memory_pool = {}
        self._cuda_stream = cv2.cuda.Stream()
    
    def _initialize_unified_numba(self):
        """統合Numbaカーネル初期化"""
        logger.info("Initializing unified Numba kernels...")
        # JIT関数のwarmup
        if HAS_NUMBA:
            self._warmup_numba_kernels()
    
    def _initialize_sequential_cuda(self):
        """逐次CUDA初期化"""
        logger.info("Initializing sequential CUDA filters...")
        self._cuda_bilateral = cv2.cuda.createBilateralFilter(
            -1, self.config.bilateral_d, 
            self.config.bilateral_sigma_color, 
            self.config.bilateral_sigma_space
        )
    
    def _initialize_sequential_cpu(self):
        """逐次CPU初期化"""
        logger.info("Initializing sequential CPU filters...")
        # 特別な初期化は不要
        pass
    
    def _warmup_numba_kernels(self):
        """Numbaカーネルのwarmup"""
        if not HAS_NUMBA:
            return
        
        # ダミーデータでkernelをコンパイル
        dummy_data = np.random.randint(500, 2000, (240, 424), dtype=np.uint16)
        dummy_prev = dummy_data.copy().astype(np.float32)
        
        try:
            # 統合カーネルをwarmup
            _ = _unified_filter_numba(dummy_data, dummy_prev, 0.8, 500, 5000)
            logger.info("Numba kernels warmed up successfully")
        except Exception as e:
            logger.warning(f"Numba warmup failed: {e}")
    
    @profile_phase("filter_pipeline")
    def apply_filters(self, depth_image: np.ndarray) -> np.ndarray:
        """
        統合フィルタパイプラインを適用
        
        Args:
            depth_image: 入力深度画像
            
        Returns:
            フィルタ済み深度画像
        """
        with measure_performance("unified_filter_pipeline"):
            start_time = time.perf_counter()
            
            # 戦略別処理
            if self.strategy == FilterStrategy.UNIFIED_CUDA:
                filtered = self._apply_unified_cuda(depth_image)
                if filtered is not None:
                    self.stats.cuda_accelerated += 1
                    self.stats.unified_kernel_usage += 1
                else:
                    # フォールバック
                    filtered = self._apply_fallback(depth_image)
                    self.stats.cpu_fallback += 1
            
            elif self.strategy == FilterStrategy.UNIFIED_NUMBA:
                filtered = self._apply_unified_numba(depth_image)
                if filtered is not None:
                    self.stats.numba_accelerated += 1
                    self.stats.unified_kernel_usage += 1
                else:
                    # フォールバック
                    filtered = self._apply_fallback(depth_image)
                    self.stats.cpu_fallback += 1
            
            elif self.strategy == FilterStrategy.SEQUENTIAL_CUDA:
                filtered = self._apply_sequential_cuda(depth_image)
                self.stats.cuda_accelerated += 1
            
            else:
                filtered = self._apply_sequential_cpu(depth_image)
                self.stats.cpu_fallback += 1
            
            # 統計更新
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.stats.total_frames += 1
            self.stats.total_time_ms += elapsed_ms
            self.stats.avg_time_ms = self.stats.total_time_ms / self.stats.total_frames
            
            return filtered
    
    def _apply_unified_cuda(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """統合CUDAカーネル適用"""
        try:
            # TODO: カスタムCUDAカーネル実装
            # 現時点では逐次CUDA処理にフォールバック
            return self._apply_sequential_cuda(depth_image)
        except Exception as e:
            logger.debug(f"Unified CUDA failed: {e}")
            return None
    
    def _apply_unified_numba(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """統合Numbaカーネル適用"""
        if not HAS_NUMBA:
            return None
        
        try:
            # 前フレーム状態の初期化
            if self._temporal_state is None:
                self._temporal_state = depth_image.astype(np.float32)
                return depth_image.copy()
            
            # 統合カーネル実行
            filtered = _unified_filter_numba(
                depth_image, 
                self._temporal_state,
                self.config.temporal_alpha,
                int(self.config.min_depth * 1000),  # mm単位
                int(self.config.max_depth * 1000)   # mm単位
            )
            
            # 時間状態更新
            self._temporal_state = filtered.astype(np.float32)
            
            return filtered
            
        except Exception as e:
            logger.debug(f"Unified Numba failed: {e}")
            return None
    
    def _apply_sequential_cuda(self, depth_image: np.ndarray) -> np.ndarray:
        """逐次CUDA処理"""
        filtered = depth_image.copy()
        
        try:
            # GPU転送
            gpu_image = cv2.cuda.GpuMat()
            gpu_image.upload(filtered.astype(np.float32))
            
            # メディアンフィルタ（OpenCVのCUDA版は限定的）
            if self.config.enable_median:
                # CPU版メディアンフィルタ（CUDAサポートなし）
                filtered = cv2.medianBlur(filtered, self.config.median_kernel_size)
                gpu_image.upload(filtered.astype(np.float32))
            
            # バイラテラルフィルタ
            if self.config.enable_bilateral:
                gpu_filtered = cv2.cuda.bilateralFilter(
                    gpu_image, 
                    self.config.bilateral_d,
                    self.config.bilateral_sigma_color,
                    self.config.bilateral_sigma_space,
                    stream=self._cuda_stream
                )
                filtered = gpu_filtered.download().astype(np.uint16)
            
            # 時間フィルタ（CPU）
            if self.config.enable_temporal:
                filtered = self._apply_temporal_filter_cpu(filtered)
                
        except Exception as e:
            logger.debug(f"Sequential CUDA failed, using CPU: {e}")
            return self._apply_sequential_cpu(depth_image)
        
        return filtered
    
    def _apply_sequential_cpu(self, depth_image: np.ndarray) -> np.ndarray:
        """逐次CPU処理"""
        filtered = depth_image.copy()
        
        # メディアンフィルタ
        if self.config.enable_median:
            filtered = cv2.medianBlur(filtered, self.config.median_kernel_size)
        
        # バイラテラルフィルタ
        if self.config.enable_bilateral:
            depth_float = filtered.astype(np.float32) / 65535.0
            filtered_float = cv2.bilateralFilter(
                depth_float,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color / 255.0,
                self.config.bilateral_sigma_space
            )
            filtered = (filtered_float * 65535.0).astype(np.uint16)
        
        # 時間フィルタ
        if self.config.enable_temporal:
            filtered = self._apply_temporal_filter_cpu(filtered)
        
        return filtered
    
    def _apply_fallback(self, depth_image: np.ndarray) -> np.ndarray:
        """フォールバック処理"""
        return self._apply_sequential_cpu(depth_image)
    
    def _apply_temporal_filter_cpu(self, depth_image: np.ndarray) -> np.ndarray:
        """CPU時間フィルタ"""
        if self._temporal_state is None:
            self._temporal_state = depth_image.astype(np.float32)
            return depth_image
        
        # EMAフィルタ
        alpha = self.config.temporal_alpha
        current_float = depth_image.astype(np.float32)
        
        # 有効領域マスク
        valid_mask = (depth_image >= self.config.min_depth * 1000) & \
                     (depth_image <= self.config.max_depth * 1000)
        
        # EMA更新
        self._temporal_state[valid_mask] = (
            alpha * current_float[valid_mask] + 
            (1 - alpha) * self._temporal_state[valid_mask]
        )
        
        return self._temporal_state.astype(np.uint16)
    
    def get_stats(self) -> PipelineStats:
        """統計情報を取得"""
        return self.stats
    
    def reset_stats(self):
        """統計をリセット"""
        self.stats = PipelineStats()
    
    def print_stats(self):
        """統計情報をコンソール出力"""
        print("\n" + "="*50)
        print("UNIFIED FILTER PIPELINE STATS")
        print("="*50)
        print(f"Strategy: {self.strategy.value}")
        print(f"Total frames: {self.stats.total_frames}")
        print(f"Average time: {self.stats.avg_time_ms:.2f}ms")
        print(f"CUDA accelerated: {self.stats.cuda_accelerated}")
        print(f"Numba accelerated: {self.stats.numba_accelerated}")
        print(f"CPU fallback: {self.stats.cpu_fallback}")
        print(f"Unified kernel usage: {self.stats.unified_kernel_usage}")
        if self.stats.total_frames > 0:
            print(f"Acceleration rate: {(self.stats.cuda_accelerated + self.stats.numba_accelerated)/self.stats.total_frames*100:.1f}%")
        print("="*50)


# Numba統合カーネル実装
if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _unified_filter_numba(depth_image: np.ndarray, 
                             temporal_state: np.ndarray,
                             temporal_alpha: float,
                             min_depth_mm: int,
                             max_depth_mm: int) -> np.ndarray:
        """
        統合Numbaフィルタカーネル
        median + bilateral + temporal を統合処理
        """
        height, width = depth_image.shape
        result = np.zeros_like(depth_image, dtype=np.uint16)
        
        # メディアンフィルタ用カーネル（3x3）
        kernel_size = 3
        half_kernel = kernel_size // 2
        
        for y in range(height):
            for x in range(width):
                current_depth = depth_image[y, x]
                
                # 深度範囲チェック
                if current_depth < min_depth_mm or current_depth > max_depth_mm:
                    result[y, x] = 0
                    continue
                
                # メディアンフィルタ（3x3）
                neighborhood = []
                for dy in range(-half_kernel, half_kernel + 1):
                    for dx in range(-half_kernel, half_kernel + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbor_depth = depth_image[ny, nx]
                            if min_depth_mm <= neighbor_depth <= max_depth_mm:
                                neighborhood.append(neighbor_depth)
                
                if len(neighborhood) == 0:
                    result[y, x] = 0
                    continue
                
                # メディアン計算（挿入ソート）
                neighborhood.sort()
                median_depth = neighborhood[len(neighborhood) // 2]
                
                # 簡易バイラテラルフィルタ（空間重み付け）
                bilateral_sum = 0.0
                weight_sum = 0.0
                sigma_space = 2.0
                sigma_color = 30.0
                
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbor_depth = depth_image[ny, nx]
                            if min_depth_mm <= neighbor_depth <= max_depth_mm:
                                # 空間重み
                                spatial_dist2 = dx*dx + dy*dy
                                spatial_weight = np.exp(-spatial_dist2 / (2 * sigma_space * sigma_space))
                                
                                # 色（深度）重み
                                color_dist = abs(float(neighbor_depth) - float(median_depth))
                                color_weight = np.exp(-color_dist*color_dist / (2 * sigma_color * sigma_color))
                                
                                total_weight = spatial_weight * color_weight
                                bilateral_sum += float(neighbor_depth) * total_weight
                                weight_sum += total_weight
                
                if weight_sum > 0:
                    bilateral_depth = bilateral_sum / weight_sum
                else:
                    bilateral_depth = float(median_depth)
                
                # 時間フィルタ（EMA）
                if temporal_state[y, x] > 0:
                    filtered_depth = (temporal_alpha * bilateral_depth + 
                                    (1 - temporal_alpha) * float(temporal_state[y, x]))
                else:
                    filtered_depth = bilateral_depth
                
                # 結果格納
                result[y, x] = max(0, min(65535, int(filtered_depth)))
                
                # 時間状態更新
                temporal_state[y, x] = filtered_depth
        
        return result
else:
    def _unified_filter_numba(*args, **kwargs):
        raise NotImplementedError("Numba not available")


def create_filter_pipeline(config: Optional[FilterConfig] = None, 
                          strategy: Optional[FilterStrategy] = None) -> UnifiedFilterPipeline:
    """フィルタパイプラインファクトリー"""
    return UnifiedFilterPipeline(config, strategy)


def benchmark_filter_strategies(depth_image: np.ndarray, iterations: int = 50) -> Dict[str, Any]:
    """フィルタ戦略のベンチマーク"""
    results = {}
    
    strategies = [
        FilterStrategy.UNIFIED_NUMBA,
        FilterStrategy.SEQUENTIAL_CPU,
    ]
    
    # CUDA利用可能な場合は追加
    pipeline_test = UnifiedFilterPipeline()
    if pipeline_test._check_cuda_availability():
        strategies.extend([
            FilterStrategy.UNIFIED_CUDA,
            FilterStrategy.SEQUENTIAL_CUDA
        ])
    
    for strategy in strategies:
        try:
            pipeline = UnifiedFilterPipeline(strategy=strategy)
            
            # ベンチマーク実行
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = pipeline.apply_filters(depth_image.copy())
                times.append((time.perf_counter() - start) * 1000.0)
            
            results[strategy.value] = {
                'avg_ms': np.mean(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'std_ms': np.std(times)
            }
            
        except Exception as e:
            logger.warning(f"Benchmark failed for {strategy.value}: {e}")
            results[strategy.value] = {'error': str(e)}
    
    return results 