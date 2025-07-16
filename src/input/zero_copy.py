#!/usr/bin/env python3
"""
ゼロコピー最適化モジュール
メモリコピーを最小化した高速データ転送
speedup.md対応: 2.6ms → 0.8ms の改善目標
"""

import time
from typing import Optional, Tuple, Any, Union
import numpy as np
import cv2
from dataclasses import dataclass

from src import get_logger
from src.performance.profiler import measure_performance

logger = get_logger(__name__)


@dataclass
class ZeroCopyStats:
    """ゼロコピー統計"""
    total_calls: int = 0
    total_time_saved_ms: float = 0.0
    avg_time_per_call_ms: float = 0.0
    memory_saved_mb: float = 0.0
    copy_avoided_count: int = 0


class ZeroCopyFrameExtractor:
    """ゼロコピーフレーム抽出器"""
    
    def __init__(self):
        self.stats = ZeroCopyStats()
        self._buffer_cache = {}  # バッファキャッシュ
        
        logger.info("Zero-copy frame extractor initialized")
    
    def extract_depth_zero_copy(self, depth_frame: Any) -> Optional[np.ndarray]:
        """
        深度フレームをゼロコピーで抽出
        
        Args:
            depth_frame: Orbbec深度フレーム
            
        Returns:
            深度画像（ゼロコピー版）またはNone
        """
        with measure_performance("depth_zero_copy"):
            start_time = time.perf_counter()
            
            try:
                # フレーム情報取得
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                
                # 1. ゼロコピー抽出を試行
                depth_array = self._try_zero_copy_extraction(depth_frame, width, height)
                
                if depth_array is not None:
                    # 成功時の統計更新
                    self._update_success_stats(start_time, width, height)
                    return depth_array
                
                # 2. フォールバック: 従来方式
                logger.debug("Zero-copy failed, falling back to traditional copy")
                return self._fallback_copy_extraction(depth_frame, width, height)
                
            except Exception as e:
                logger.warning(f"Zero-copy extraction failed: {e}")
                return None
    
    def _try_zero_copy_extraction(self, depth_frame: Any, width: int, height: int) -> Optional[np.ndarray]:
        """ゼロコピー抽出を試行"""
        try:
            # Method 1: memoryview からのゼロコピー
            if hasattr(depth_frame, 'get_data_ptr'):
                # ポインタ直接アクセス（理想的）
                data_ptr = depth_frame.get_data_ptr()
                if data_ptr:
                    # ctypes 経由でのゼロコピーアクセス
                    import ctypes
                    buffer_size = width * height * 2  # uint16 = 2 bytes
                    buffer_type = ctypes.c_uint8 * buffer_size
                    buffer = buffer_type.from_address(data_ptr)
                    
                    # numpy array as memoryview (zero-copy)
                    array = np.frombuffer(buffer, dtype=np.uint16).reshape((height, width))
                    self.stats.copy_avoided_count += 1
                    return array
            
            # Method 2: get_data() のmemoryview化
            raw_data = depth_frame.get_data()
            if hasattr(raw_data, '__array_interface__') or hasattr(raw_data, '__array__'):
                # NumPy互換オブジェクト
                array = np.asarray(raw_data, dtype=np.uint16)
                if array.size == width * height:
                    self.stats.copy_avoided_count += 1
                    return array.reshape((height, width))
            
            # Method 3: バッファプロトコル経由
            if hasattr(raw_data, '__buffer__'):
                try:
                    # Python 3.12+ buffer protocol
                    mv = memoryview(raw_data)
                    array = np.asarray(mv, dtype=np.uint16).reshape((height, width))
                    self.stats.copy_avoided_count += 1
                    return array
                except (ValueError, TypeError):
                    pass
            
            # Method 4: frombuffer with copy=False試行
            try:
                # NumPy 1.21+ supports copy parameter
                array = np.frombuffer(raw_data, dtype=np.uint16)
                if array.size == width * height:
                    reshaped = array.reshape((height, width))
                    # 書き込み可能性チェック（真のゼロコピーの場合は read-only）
                    if not reshaped.flags.writeable:
                        # ゼロコピー成功
                        self.stats.copy_avoided_count += 1
                        return reshaped
            except Exception:
                pass
                
            return None
            
        except Exception as e:
            logger.debug(f"Zero-copy method failed: {e}")
            return None
    
    def _fallback_copy_extraction(self, depth_frame: Any, width: int, height: int) -> np.ndarray:
        """従来方式フォールバック"""
        raw_data = depth_frame.get_data()
        depth_data = np.frombuffer(raw_data, dtype=np.uint16)
        return depth_data.reshape((height, width))
    
    def _update_success_stats(self, start_time: float, width: int, height: int):
        """成功時の統計更新"""
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        self.stats.total_calls += 1
        self.stats.total_time_saved_ms += elapsed_ms
        self.stats.avg_time_per_call_ms = self.stats.total_time_saved_ms / self.stats.total_calls
        
        # 推定メモリ節約量（uint16 = 2bytes）
        frame_size_mb = (width * height * 2) / (1024 * 1024)
        self.stats.memory_saved_mb += frame_size_mb
    
    def extract_color_zero_copy(self, color_frame: Any) -> Optional[np.ndarray]:
        """
        カラーフレームをゼロコピーで抽出
        
        Args:
            color_frame: Orbbecカラーフレーム
            
        Returns:
            カラー画像（ゼロコピー版）またはNone
        """
        with measure_performance("color_zero_copy"):
            try:
                # フォーマット取得
                from src.data_types import OBFormat
                color_format = color_frame.get_format()
                
                # RGB/BGR の場合のみゼロコピー対応
                if color_format in [OBFormat.RGB, OBFormat.BGR]:
                    raw_data = color_frame.get_data()
                    
                    # ゼロコピー試行
                    try:
                        color_array = np.asarray(raw_data, dtype=np.uint8)
                        if color_array.size > 0:
                            # 解像度推定（一般的な解像度から）
                            height, width = self._estimate_color_resolution(color_array.size)
                            if height > 0 and width > 0:
                                color_image = color_array.reshape((height, width, 3))
                                
                                # BGR → RGB 変換（必要な場合）
                                if color_format == OBFormat.BGR:
                                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                                
                                self.stats.copy_avoided_count += 1
                                return color_image
                    except Exception:
                        pass
                
                # フォールバック: 従来方式
                return self._fallback_color_extraction(color_frame)
                
            except Exception as e:
                logger.debug(f"Color zero-copy extraction failed: {e}")
                return None
    
    def _estimate_color_resolution(self, total_pixels: int) -> Tuple[int, int]:
        """カラー画像解像度を推定"""
        # 一般的な解像度パターンをチェック
        common_resolutions = [
            (720, 1280),   # 720p
            (480, 640),    # VGA
            (480, 848),    # Orbbec一般的
            (1080, 1920),  # 1080p
        ]
        
        for height, width in common_resolutions:
            if total_pixels == height * width * 3:  # RGB = 3 channels
                return height, width
        
        return 0, 0
    
    def _fallback_color_extraction(self, color_frame: Any) -> Optional[np.ndarray]:
        """カラーフレーム従来方式フォールバック"""
        try:
            from src.data_types import OBFormat
            color_format = color_frame.get_format()
            raw_data = color_frame.get_data()
            
            if color_format == OBFormat.MJPG:
                # JPEG デコード
                color_data = np.frombuffer(raw_data, dtype=np.uint8)
                decoded_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if decoded_image is not None:
                    return cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
            else:
                # RGB/BGR
                color_data = np.frombuffer(raw_data, dtype=np.uint8)
                height, width = self._estimate_color_resolution(color_data.size)
                if height > 0 and width > 0:
                    color_image = color_data.reshape((height, width, 3))
                    if color_format == OBFormat.BGR:
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    return color_image
                    
            return None
            
        except Exception as e:
            logger.debug(f"Color fallback extraction failed: {e}")
            return None
    
    def get_stats(self) -> ZeroCopyStats:
        """統計情報を取得"""
        return self.stats
    
    def reset_stats(self):
        """統計をリセット"""
        self.stats = ZeroCopyStats()
    
    def print_stats(self):
        """統計情報をコンソール出力"""
        print("\n" + "="*50)
        print("ZERO-COPY OPTIMIZATION STATS")
        print("="*50)
        print(f"Total calls: {self.stats.total_calls}")
        print(f"Copy avoided: {self.stats.copy_avoided_count}")
        print(f"Success rate: {self.stats.copy_avoided_count/self.stats.total_calls*100:.1f}%" if self.stats.total_calls > 0 else "Success rate: 0%")
        print(f"Avg time per call: {self.stats.avg_time_per_call_ms:.3f}ms")
        print(f"Total time saved: {self.stats.total_time_saved_ms:.1f}ms")
        print(f"Memory saved: {self.stats.memory_saved_mb:.1f}MB")
        print("="*50)


# 型ヒント用
ZeroCopyResult = Tuple[Optional[np.ndarray], bool]  # (data, is_zero_copy)


def extract_frame_optimized(frame: Any, frame_type: str = "depth") -> ZeroCopyResult:
    """
    最適化されたフレーム抽出
    
    Args:
        frame: フレームオブジェクト
        frame_type: "depth" または "color"
        
    Returns:
        (extracted_data, is_zero_copy): 抽出されたデータとゼロコピー成功フラグ
    """
    extractor = ZeroCopyFrameExtractor()
    
    if frame_type == "depth":
        data = extractor.extract_depth_zero_copy(frame)
        is_zero_copy = extractor.stats.copy_avoided_count > 0
    elif frame_type == "color":
        data = extractor.extract_color_zero_copy(frame)
        is_zero_copy = extractor.stats.copy_avoided_count > 0
    else:
        logger.warning(f"Unsupported frame type: {frame_type}")
        return None, False
    
    return data, is_zero_copy


def benchmark_zero_copy_vs_traditional(frame: Any, iterations: int = 100) -> dict:
    """
    ゼロコピーと従来方式のベンチマーク比較
    
    Args:
        frame: テスト用フレーム
        iterations: 反復回数
        
    Returns:
        ベンチマーク結果
    """
    from src.performance.profiler import benchmark_function
    
    extractor = ZeroCopyFrameExtractor()
    
    # ゼロコピー版
    def zero_copy_test():
        return extractor.extract_depth_zero_copy(frame)
    
    # 従来版
    def traditional_test():
        raw_data = frame.get_data()
        depth_data = np.frombuffer(raw_data, dtype=np.uint16)
        return depth_data.reshape((frame.get_height(), frame.get_width()))
    
    zero_copy_results = benchmark_function(zero_copy_test, iterations)
    traditional_results = benchmark_function(traditional_test, iterations)
    
    speedup = traditional_results['avg_ms'] / zero_copy_results['avg_ms'] if zero_copy_results['avg_ms'] > 0 else 1.0
    
    return {
        'zero_copy': zero_copy_results,
        'traditional': traditional_results,
        'speedup': speedup,
        'time_saved_ms': traditional_results['avg_ms'] - zero_copy_results['avg_ms'],
        'zero_copy_success_rate': extractor.stats.copy_avoided_count / iterations * 100.0
    }


# グローバルインスタンス
_global_extractor: Optional[ZeroCopyFrameExtractor] = None


def get_zero_copy_extractor() -> ZeroCopyFrameExtractor:
    """グローバルゼロコピー抽出器を取得"""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = ZeroCopyFrameExtractor()
    return _global_extractor 