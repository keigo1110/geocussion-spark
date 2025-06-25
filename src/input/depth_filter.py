#!/usr/bin/env python3
"""
深度画像ノイズフィルタモジュール
median / bilateral / temporal フィルタによるノイズ除去
残像低減・エッジ保持対応
"""

import time
from typing import Optional, Dict, Any, Deque, List
from collections import deque
from enum import Enum
import numpy as np
import cv2

# メモリ最適化
from ..collision.optimization import optimize_array_operations, memory_efficient_context, InPlaceOperations


class FilterType(Enum):
    """フィルタタイプ"""
    MEDIAN = "median"
    BILATERAL = "bilateral"
    TEMPORAL = "temporal"
    COMBINED = "combined"


class DepthFilter:
    """深度画像ノイズフィルタクラス"""
    
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
        max_valid_depth: float = 10.0
    ):
        """
        初期化
        
        Args:
            filter_types: 適用するフィルタタイプのリスト
            median_kernel_size: メディアンフィルタのカーネルサイズ
            bilateral_d: バイラテラルフィルタの近傍径
            bilateral_sigma_color: バイラテラルフィルタの色空間標準偏差
            bilateral_sigma_space: バイラテラルフィルタの座標空間標準偏差
            temporal_alpha: 時間フィルタの重み（0.0-1.0、小さいほど平滑化強）
            temporal_history_size: 時間フィルタの履歴サイズ
            min_valid_depth: 有効深度の最小値
            max_valid_depth: 有効深度の最大値
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
        
        # 時間フィルタ用の履歴バッファ
        self.depth_history: Deque[np.ndarray] = deque(maxlen=temporal_history_size)
        self.temporal_avg: Optional[np.ndarray] = None
        
        # パフォーマンス計測
        self.processing_times: Dict[str, float] = {}
        
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
    
    def _apply_bilateral_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        バイラテラルフィルタを適用（エッジ保持平滑化）
        
        Args:
            depth_image: 深度画像
            valid_mask: 有効ピクセルマスク
            
        Returns:
            フィルタ済み画像
        """
        start_time = time.perf_counter()
        
        filtered = depth_image.copy()
        if np.any(valid_mask):
            # バイラテラルフィルタ（エッジ保持）
            # uint16のままではcv2.bilateralFilterが期待通りに動作しないことがあるため、
            # float32に正規化してから処理し、精度を維持する。
            depth_float = depth_image.astype(np.float32) / 65535.0
            
            filtered_float = cv2.bilateralFilter(
                depth_float,
                self.bilateral_d,
                self.bilateral_sigma_color / 255.0,  # sigmaColorをfloatスケールに調整
                self.bilateral_sigma_space
            )
            
            # 16bitに戻す
            filtered = (filtered_float * 65535.0).astype(np.uint16)
        
        self.processing_times['bilateral'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
    def _apply_temporal_filter(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        時間フィルタを適用（残像低減）
        
        Args:
            depth_image: 深度画像
            valid_mask: 有効ピクセルマスク
            
        Returns:
            フィルタ済み画像
        """
        start_time = time.perf_counter()
        
        # 履歴に追加（コピーは必要最小限）
        self.depth_history.append(depth_image.copy())
        
        filtered = depth_image.copy()
        
        if len(self.depth_history) > 1:
            if self.temporal_avg is None:
                # 初回は単純平均
                self.temporal_avg = np.mean(
                    [img.astype(np.float32) for img in self.depth_history],
                    axis=0
                ).astype(np.uint16)
            else:
                # 指数移動平均（EMA）による平滑化
                current_float = depth_image.astype(np.float32)
                avg_float = self.temporal_avg.astype(np.float32)
                
                # 動きが大きい領域は追従性を高める
                diff = np.abs(current_float - avg_float)
                adaptive_alpha = np.clip(
                    self.temporal_alpha + diff / 500.0,  # 差が大きいほど追従性向上
                    0.1, 0.9
                )
                
                new_avg = adaptive_alpha * current_float + (1 - adaptive_alpha) * avg_float
                self.temporal_avg = new_avg.astype(np.uint16)
            
            assert self.temporal_avg is not None
            filtered = self.temporal_avg.copy()
        
        self.processing_times['temporal'] = (time.perf_counter() - start_time) * 1000
        return filtered
    
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
        self.depth_history.clear()
        self.temporal_avg = None
    
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