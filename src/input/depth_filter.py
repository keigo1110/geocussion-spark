#!/usr/bin/env python3
"""
深度画像ノイズフィルタモジュール
median / bilateral / temporal フィルタによるノイズ除去
残像低減・エッジ保持対応 + CUDA高速化対応
"""

import time
from typing import Optional, Dict, Any, Deque, List
from collections import deque
from enum import Enum
import numpy as np
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor

# メモリ最適化
from ..collision.optimization import optimize_array_operations, memory_efficient_context, InPlaceOperations
from .. import get_logger

logger = get_logger(__name__)

# CUDA利用可能性チェック
try:
    # OpenCV CUDA機能の確認
    cv2.cuda.getCudaEnabledDeviceCount()
    HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if HAS_CUDA:
        logger.info(f"CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount()} devices found")
except (cv2.error, AttributeError):
    HAS_CUDA = False
    logger.info("CUDA not available, using CPU fallback")


class FilterType(Enum):
    """フィルタタイプ"""
    MEDIAN = "median"
    BILATERAL = "bilateral"
    TEMPORAL = "temporal"
    COMBINED = "combined"


class CudaBilateralFilter:
    """CUDA対応バイラテラルフィルタ"""
    
    def __init__(self):
        self.gpu_mat_cache = {}
        self.stream = cv2.cuda.Stream() if HAS_CUDA else None
        self.initialized = False
        
    def initialize(self, image_shape: tuple):
        """GPU行列の初期化"""
        if not HAS_CUDA:
            return False
            
        try:
            h, w = image_shape
            # GPU行列をキャッシュ
            self.gpu_mat_cache['input'] = cv2.cuda.GpuMat(h, w, cv2.CV_32F)
            self.gpu_mat_cache['output'] = cv2.cuda.GpuMat(h, w, cv2.CV_32F)
            self.initialized = True
            logger.debug(f"CUDA bilateral filter initialized for {w}x{h}")
            return True
        except Exception as e:
            logger.warning(f"CUDA bilateral filter initialization failed: {e}")
            return False
    
    def apply(self, depth_image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
        """CUDA バイラテラルフィルタ適用"""
        if not HAS_CUDA or not self.initialized:
            return None
            
        try:
            # CPU → GPU転送
            gpu_input = self.gpu_mat_cache['input']
            gpu_output = self.gpu_mat_cache['output']
            
            gpu_input.upload(depth_image)
            
            # CUDA バイラテラルフィルタ実行
            cv2.cuda.bilateralFilter(
                gpu_input, 
                gpu_output, 
                d, 
                sigma_color, 
                sigma_space,
                stream=self.stream
            )
            
            # GPU → CPU転送
            result = gpu_output.download()
            
            if self.stream:
                self.stream.waitForCompletion()
                
            return result
            
        except Exception as e:
            logger.warning(f"CUDA bilateral filter error: {e}")
            return None


class DepthFilter:
    """深度画像ノイズフィルタクラス（CUDA高速化対応）"""
    
    def __init__(
        self,
        filter_types: Optional[List[FilterType]] = None,
        median_kernel_size: int = 5,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        temporal_alpha: float = 0.3,
        temporal_history_size: int = 3,
        min_valid_depth: float = 0.1,
        max_valid_depth: float = 10.0,
        use_cuda: bool = True,
        enable_multiscale: bool = True,
        enable_async: bool = True
    ):
        """
        初期化
        
        Args:
            use_cuda: CUDA加速を使用するか
            enable_multiscale: マルチスケール処理を有効にするか
            enable_async: 非同期処理を有効にするか
        """
        self.filter_types = filter_types or [FilterType.COMBINED]
        self.median_kernel_size = median_kernel_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.temporal_alpha = temporal_alpha
        self.temporal_history_size = temporal_history_size
        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth
        
        # CUDA & パフォーマンス設定
        # Enable CUDA only if OpenCV has required bilateralFilter implementation
        cuda_available = use_cuda and HAS_CUDA and hasattr(cv2.cuda, "bilateralFilter")
        if use_cuda and HAS_CUDA and not cuda_available:
            logger.info("cv2.cuda.bilateralFilter not found – falling back to CPU version of DepthFilter")
        self.use_cuda = cuda_available
        self.enable_multiscale = enable_multiscale
        self.enable_async = enable_async
        
        # CUDA フィルタ初期化
        self.cuda_bilateral = CudaBilateralFilter() if self.use_cuda else None
        
        # EMA 時間フィルタ用（履歴バッファ削除）
        self.temporal_ema: Optional[np.ndarray] = None
        self.ema_initialized = False
        
        # 非同期処理用
        self.executor = ThreadPoolExecutor(max_workers=2) if enable_async else None
        
        # パフォーマンス計測
        self.processing_times: Dict[str, float] = {}
        
        logger.info(f"DepthFilter initialized: CUDA={self.use_cuda}, "
                   f"multiscale={enable_multiscale}, async={enable_async}")
        
    @optimize_array_operations
    def apply_filter(self, depth_image: np.ndarray) -> np.ndarray:
        """
        深度画像にフィルタを適用（メモリ最適化版）
        
        Args:
            depth_image: 入力深度画像 (H, W)
            
        Returns:
            フィルタ済み深度画像
        """
        start_time = time.perf_counter()
        
        # メモリ効率的なコンテキストで処理
        with memory_efficient_context() as ctx:
            pool = ctx['pool']
            inplace_ops = ctx['inplace_ops']
            
            # 入力検証
            if depth_image.dtype != np.uint16:
                depth_image = depth_image.astype(np.uint16)
            
            # 深度範囲マスク作成（一時配列使用）
            with pool.temporary_array(depth_image.shape, 'float32') as depth_float:
                np.copyto(depth_float, depth_image.astype(np.float32) / 1000.0)
                valid_mask = (depth_float >= self.min_valid_depth) & (depth_float <= self.max_valid_depth)
            
            # 結果配列をプールから取得
            with pool.temporary_array(depth_image.shape, depth_image.dtype) as filtered_image:
                np.copyto(filtered_image, depth_image)
                
                # フィルタタイプに応じた処理
                for filter_type in self.filter_types:
                    if filter_type == FilterType.MEDIAN:
                        filtered_image[:] = self._apply_median_filter(filtered_image, valid_mask)
                    elif filter_type == FilterType.BILATERAL:
                        filtered_image[:] = self._apply_bilateral_filter(filtered_image, valid_mask)
                    elif filter_type == FilterType.TEMPORAL:
                        filtered_image[:] = self._apply_temporal_filter(filtered_image, valid_mask)
                    elif filter_type == FilterType.COMBINED:
                        filtered_image[:] = self._apply_combined_filter(filtered_image, valid_mask)
                
                # 無効領域をマスク（インプレース）
                filtered_image[~valid_mask] = 0
                
                self.processing_times['total'] = (time.perf_counter() - start_time) * 1000
                return filtered_image.copy()  # 最終的にコピーして返す
    
    def _apply_median_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        メディアンフィルタを適用
        
        Args:
            depth_image: 深度画像
            valid_mask: 有効ピクセルマスク
            
        Returns:
            フィルタ済み画像
        """
        start_time = time.perf_counter()
        
        # 有効領域のみにフィルタ適用
        filtered = depth_image.copy()
        if np.any(valid_mask):
            # OpenCVのメディアンフィルタを使用
            filtered = cv2.medianBlur(depth_image, self.median_kernel_size)
        
        self.processing_times['median'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
    def _apply_bilateral_filter_cuda(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        CUDA バイラテラルフィルタを適用（高速版）
        
        Args:
            depth_image: 深度画像
            valid_mask: 有効ピクセルマスク
            
        Returns:
            フィルタ済み画像
        """
        start_time = time.perf_counter()
        
        filtered = depth_image.copy()
        
        if not np.any(valid_mask):
            self.processing_times['bilateral_cuda'] = (time.perf_counter() - start_time) * 1000
            return filtered
        
        try:
            # CUDA フィルタ初期化（初回のみ）
            if not self.cuda_bilateral.initialized:
                if not self.cuda_bilateral.initialize(depth_image.shape):
                    # CUDA初期化失敗時はCPU版にフォールバック
                    return self._apply_bilateral_filter_cpu(depth_image, valid_mask)
            
            # マルチスケール処理
            if self.enable_multiscale and min(depth_image.shape) > 400:
                # 1/2解像度で前処理
                h, w = depth_image.shape
                small_h, small_w = h // 2, w // 2
                
                # ダウンサンプリング
                depth_small = cv2.resize(depth_image, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                depth_small_float = depth_small.astype(np.float32) / 65535.0
                
                # 小解像度でバイラテラルフィルタ
                filtered_small_result = self.cuda_bilateral.apply(
                    depth_small_float, 
                    self.bilateral_d // 2,  # 解像度に応じてパラメータ調整
                    self.bilateral_sigma_color / 255.0,
                    self.bilateral_sigma_space / 2
                )
                
                if filtered_small_result is not None:
                    # アップサンプリング
                    filtered_small_up = cv2.resize(
                        filtered_small_result, (w, h), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    filtered = (filtered_small_up * 65535.0).astype(np.uint16)
                else:
                    # CUDA失敗時はCPU版フォールバック
                    filtered = self._apply_bilateral_filter_cpu(depth_image, valid_mask)
            else:
                # フル解像度処理
                depth_float = depth_image.astype(np.float32) / 65535.0
                filtered_float_result = self.cuda_bilateral.apply(
                    depth_float,
                    self.bilateral_d,
                    self.bilateral_sigma_color / 255.0,
                    self.bilateral_sigma_space
                )
                
                if filtered_float_result is not None:
                    filtered = (filtered_float_result * 65535.0).astype(np.uint16)
                else:
                    # CUDA失敗時はCPU版フォールバック
                    filtered = self._apply_bilateral_filter_cpu(depth_image, valid_mask)
                    
        except Exception as e:
            logger.warning(f"CUDA bilateral filter error, falling back to CPU: {e}")
            filtered = self._apply_bilateral_filter_cpu(depth_image, valid_mask)
        
        self.processing_times['bilateral_cuda'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
    def _apply_bilateral_filter_cpu(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        CPU バイラテラルフィルタを適用（従来版）
        """
        start_time = time.perf_counter()
        
        filtered = depth_image.copy()
        if np.any(valid_mask):
            depth_float = depth_image.astype(np.float32) / 65535.0
            
            filtered_float = cv2.bilateralFilter(
                depth_float,
                self.bilateral_d,
                self.bilateral_sigma_color / 255.0,
                self.bilateral_sigma_space
            )
            
            filtered = (filtered_float * 65535.0).astype(np.uint16)
        
        self.processing_times['bilateral_cpu'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
    def _apply_bilateral_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        バイラテラルフィルタを適用（CUDA/CPU自動切替）
        """
        if self.use_cuda:
            return self._apply_bilateral_filter_cuda(depth_image, valid_mask)
        else:
            return self._apply_bilateral_filter_cpu(depth_image, valid_mask)
    
    def _apply_temporal_ema_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        EMA 時間フィルタを適用（履歴バッファ不要版）
        
        Args:
            depth_image: 深度画像
            valid_mask: 有効ピクセルマスク
            
        Returns:
            フィルタ済み画像
        """
        start_time = time.perf_counter()
        
        if not self.ema_initialized:
            # 初回はそのまま使用
            self.temporal_ema = depth_image.astype(np.float32)
            self.ema_initialized = True
            filtered = depth_image.copy()
        else:
            # EMA更新: ema = alpha * current + (1-alpha) * ema
            cv2.accumulateWeighted(
                depth_image.astype(np.float32),
                self.temporal_ema,
                self.temporal_alpha,
                mask=valid_mask.astype(np.uint8)
            )
            filtered = self.temporal_ema.astype(np.uint16)
        
        self.processing_times['temporal_ema'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
    def _apply_temporal_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        時間フィルタを適用（EMA版使用）
        """
        return self._apply_temporal_ema_filter(depth_image, valid_mask)
    
    def _apply_combined_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        複合フィルタを適用（最適化された順序で全フィルタを組み合わせ）
        
        Args:
            depth_image: 深度画像
            valid_mask: 有効ピクセルマスク
            
        Returns:
            フィルタ済み画像
        """
        start_time = time.perf_counter()
        
        # 1. 時間フィルタ（最初に適用して全体的なノイズを減らす）
        filtered = self._apply_temporal_filter(depth_image, valid_mask)
        
        # 2. メディアンフィルタ（塩胡椒ノイズ除去）
        filtered = self._apply_median_filter(filtered, valid_mask)
        
        # 3. バイラテラルフィルタ（エッジ保持平滑化）
        filtered = self._apply_bilateral_filter(filtered, valid_mask)
        
        self.processing_times['combined'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
    def reset_temporal_history(self) -> None:
        """時間フィルタの履歴をリセット"""
        self.temporal_ema = None
        self.ema_initialized = False
    
    def get_performance_stats(self) -> Dict[str, float]:
        """パフォーマンス統計を取得"""
        return self.processing_times.copy()
    
    def update_parameters(self, **kwargs) -> None:
        """フィルタパラメータを動的更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['temporal_alpha', 'temporal_history_size']:
                    # 時間フィルタパラメータ変更時は履歴リセット
                    self.reset_temporal_history()

    def _estimate_noise_level(self, depth_image: np.ndarray) -> float:
        """
        深度画像のノイズレベルを推定
        ラプラシアンフィルタの分散をノイズ指標として利用
        """
        # ゼロ以外の深度値のみを対象
        valid_depth = depth_image[depth_image > 0]
        if valid_depth.size < 100:
            return 0.0  # 十分なデータがない
        
        # 8bitに変換してからラプラシアン適用
        depth_8bit = cv2.convertScaleAbs(valid_depth, alpha=255.0/valid_depth.max())
        laplacian_var = cv2.Laplacian(depth_8bit, cv2.CV_64F).var()
        
        return float(laplacian_var)
    
    def _adapt_parameters(self, noise_level: float) -> None:
        """ノイズレベルに応じてフィルタパラメータを調整"""
        if noise_level > 100:  # 高ノイズ
            self.temporal_alpha = 0.2  # 強い平滑化
            self.bilateral_sigma_color = 80.0
            self.median_kernel_size = 7
        elif noise_level > 50:  # 中ノイズ
            self.temporal_alpha = 0.3
            self.bilateral_sigma_color = 50.0
            self.median_kernel_size = 5
        else:  # 低ノイズ
            self.temporal_alpha = 0.5  # 応答性重視
            self.bilateral_sigma_color = 30.0
            self.median_kernel_size = 3


class AdaptiveDepthFilter(DepthFilter):
    """適応的深度フィルタ（動的パラメータ調整）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noise_level_history: Deque[float] = deque(maxlen=10)
        self.enable_adaptive = True
        
    def apply_filter(self, depth_image: np.ndarray) -> np.ndarray:
        """
        適応的フィルタ適用（ノイズレベルに応じてパラメータ自動調整）
        
        Args:
            depth_image: 入力深度画像
            
        Returns:
            フィルタ済み深度画像
        """
        if self.enable_adaptive:
            noise_level = self._estimate_noise_level(depth_image)
            self.noise_level_history.append(noise_level)
            
            # 平均ノイズレベルに基づくパラメータ調整
            if len(self.noise_level_history) >= 3:
                avg_noise = float(np.mean(self.noise_level_history))
                self._adapt_parameters(avg_noise)
        
        return super().apply_filter(depth_image)
    
    def _estimate_noise_level(self, depth_image: np.ndarray) -> float:
        """
        ノイズレベルを推定
        
        Args:
            depth_image: 深度画像
            
        Returns:
            推定ノイズレベル
        """
        # Laplacianによるエッジ強度計算
        depth_float = depth_image.astype(np.float32)
        laplacian = cv2.Laplacian(depth_float, cv2.CV_32F)
        noise_estimate = float(np.std(laplacian[depth_image > 0]))
        return noise_estimate
    
    def _adapt_parameters(self, noise_level: float) -> None:
        """
        ノイズレベルに応じてフィルタパラメータを調整
        
        Args:
            noise_level: 推定ノイズレベル
        """
        if noise_level > 100:  # 高ノイズ
            self.temporal_alpha = 0.2  # 強い平滑化
            self.bilateral_sigma_color = 80.0
            self.median_kernel_size = 7
        elif noise_level > 50:  # 中ノイズ
            self.temporal_alpha = 0.3
            self.bilateral_sigma_color = 50.0
            self.median_kernel_size = 5
        else:  # 低ノイズ
            self.temporal_alpha = 0.5  # 応答性重視
            self.bilateral_sigma_color = 30.0
            self.median_kernel_size = 3 