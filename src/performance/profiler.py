#!/usr/bin/env python3
"""
パフォーマンス測定とプロファイリング基盤
リアルタイムボトルネック解析とFPS最適化
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
from contextlib import contextmanager
import gc
import psutil
import sys

from src import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    frame_time_ms: float = 0.0
    fps: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    # フェーズ別時間
    depth_processing_ms: float = 0.0
    pointcloud_generation_ms: float = 0.0
    hand_detection_ms: float = 0.0
    mesh_generation_ms: float = 0.0
    collision_detection_ms: float = 0.0
    audio_synthesis_ms: float = 0.0
    # 詳細統計
    total_allocations: int = 0
    peak_memory_mb: float = 0.0
    gc_collections: int = 0


@dataclass
class PhaseTimer:
    """フェーズ別タイマー"""
    name: str
    start_time: float = 0.0
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    def start(self):
        """タイマー開始"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """タイマー停止と時間記録"""
        if self.start_time == 0.0:
            return 0.0
        
        elapsed = (time.perf_counter() - self.start_time) * 1000.0
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.start_time = 0.0
        return elapsed
    
    def get_average(self) -> float:
        """平均時間を取得"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def reset(self):
        """統計リセット"""
        self.total_time = 0.0
        self.call_count = 0
        self.min_time = float('inf')
        self.max_time = 0.0


class PerformanceProfiler:
    """リアルタイムパフォーマンスプロファイラー"""
    
    def __init__(self, history_size: int = 100, enable_detailed_profiling: bool = True):
        self.history_size = history_size
        self.enable_detailed_profiling = enable_detailed_profiling
        
        # メトリクス履歴
        self.metrics_history: deque = deque(maxlen=history_size)
        self.frame_times: deque = deque(maxlen=history_size)
        
        # フェーズタイマー
        self.phase_timers: Dict[str, PhaseTimer] = {}
        
        # システム監視
        self.process = psutil.Process()
        self.last_frame_time = time.perf_counter()
        
        # GPU監視（オプション）
        self.gpu_available = self._check_gpu_availability()
        
        # リアルタイム統計
        self.current_metrics = PerformanceMetrics()
        
        # スレッドセーフティ
        self.lock = threading.RLock()
        
        logger.info(f"Performance profiler initialized (GPU: {'available' if self.gpu_available else 'unavailable'})")
    
    def _check_gpu_availability(self) -> bool:
        """GPU監視可能性をチェック"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            return False
    
    def create_timer(self, phase_name: str) -> PhaseTimer:
        """フェーズタイマーを作成"""
        with self.lock:
            if phase_name not in self.phase_timers:
                self.phase_timers[phase_name] = PhaseTimer(phase_name)
            return self.phase_timers[phase_name]
    
    @contextmanager
    def measure_phase(self, phase_name: str):
        """フェーズ実行時間を測定"""
        timer = self.create_timer(phase_name)
        timer.start()
        try:
            yield timer
        finally:
            timer.stop()
    
    def start_frame(self):
        """フレーム開始時刻を記録"""
        self.last_frame_time = time.perf_counter()
    
    def end_frame(self) -> PerformanceMetrics:
        """フレーム終了とメトリクス更新"""
        current_time = time.perf_counter()
        frame_time = (current_time - self.last_frame_time) * 1000.0
        
        with self.lock:
            # フレーム時間履歴更新
            self.frame_times.append(frame_time)
            
            # 現在のメトリクス更新
            self.current_metrics.frame_time_ms = frame_time
            self.current_metrics.fps = 1000.0 / frame_time if frame_time > 0 else 0.0
            
            # システムリソース監視
            if self.enable_detailed_profiling:
                self._update_system_metrics()
            
            # フェーズ別時間更新
            self._update_phase_metrics()
            
            # 履歴に追加
            metrics_copy = PerformanceMetrics(
                frame_time_ms=self.current_metrics.frame_time_ms,
                fps=self.current_metrics.fps,
                cpu_percent=self.current_metrics.cpu_percent,
                memory_mb=self.current_metrics.memory_mb,
                gpu_memory_mb=self.current_metrics.gpu_memory_mb,
                depth_processing_ms=self.current_metrics.depth_processing_ms,
                pointcloud_generation_ms=self.current_metrics.pointcloud_generation_ms,
                hand_detection_ms=self.current_metrics.hand_detection_ms,
                mesh_generation_ms=self.current_metrics.mesh_generation_ms,
                collision_detection_ms=self.current_metrics.collision_detection_ms,
                audio_synthesis_ms=self.current_metrics.audio_synthesis_ms
            )
            self.metrics_history.append(metrics_copy)
            
        return self.current_metrics
    
    def _update_system_metrics(self):
        """システムメトリクス更新"""
        try:
            # CPU使用率
            self.current_metrics.cpu_percent = self.process.cpu_percent()
            
            # メモリ使用量
            memory_info = self.process.memory_info()
            self.current_metrics.memory_mb = memory_info.rss / 1024 / 1024
            
            # ピークメモリ更新
            self.current_metrics.peak_memory_mb = max(
                self.current_metrics.peak_memory_mb,
                self.current_metrics.memory_mb
            )
            
            # GPU メモリ（可能な場合）
            if self.gpu_available:
                self.current_metrics.gpu_memory_mb = self._get_gpu_memory()
            
            # GC統計
            gc_stats = gc.get_stats()
            if gc_stats:
                self.current_metrics.gc_collections = sum(stat['collections'] for stat in gc_stats)
                
        except Exception as e:
            logger.debug(f"System metrics update failed: {e}")
    
    def _get_gpu_memory(self) -> float:
        """GPU メモリ使用量を取得"""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 / 1024  # MB
        except Exception:
            return 0.0
    
    def _update_phase_metrics(self):
        """フェーズ別メトリクス更新"""
        phase_mapping = {
            'depth_processing': 'depth_processing_ms',
            'pointcloud_generation': 'pointcloud_generation_ms',
            'hand_detection': 'hand_detection_ms',
            'mesh_generation': 'mesh_generation_ms',
            'collision_detection': 'collision_detection_ms',
            'audio_synthesis': 'audio_synthesis_ms'
        }
        
        for phase_name, metric_name in phase_mapping.items():
            if phase_name in self.phase_timers:
                timer = self.phase_timers[phase_name]
                if timer.call_count > 0:
                    setattr(self.current_metrics, metric_name, timer.get_average())
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """現在のメトリクスを取得"""
        with self.lock:
            return self.current_metrics
    
    def get_average_fps(self, window_size: int = 30) -> float:
        """指定ウィンドウでの平均FPSを取得"""
        with self.lock:
            if len(self.frame_times) < window_size:
                return 0.0
            
            recent_times = list(self.frame_times)[-window_size:]
            avg_frame_time = sum(recent_times) / len(recent_times)
            return 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def identify_bottlenecks(self, threshold_ms: float = 10.0) -> List[str]:
        """ボトルネックを特定"""
        bottlenecks = []
        
        with self.lock:
            for phase_name, timer in self.phase_timers.items():
                if timer.get_average() > threshold_ms:
                    bottlenecks.append(f"{phase_name}: {timer.get_average():.1f}ms")
        
        return bottlenecks
    
    def get_performance_report(self) -> Dict[str, Any]:
        """詳細パフォーマンスレポートを生成"""
        with self.lock:
            report = {
                'summary': {
                    'current_fps': self.current_metrics.fps,
                    'avg_fps_30': self.get_average_fps(30),
                    'frame_time_ms': self.current_metrics.frame_time_ms,
                    'cpu_percent': self.current_metrics.cpu_percent,
                    'memory_mb': self.current_metrics.memory_mb,
                    'peak_memory_mb': self.current_metrics.peak_memory_mb
                },
                'phases': {},
                'bottlenecks': self.identify_bottlenecks(),
                'system': {
                    'total_frames': len(self.frame_times),
                    'gc_collections': self.current_metrics.gc_collections
                }
            }
            
            # フェーズ詳細
            for phase_name, timer in self.phase_timers.items():
                if timer.call_count > 0:
                    report['phases'][phase_name] = {
                        'avg_ms': timer.get_average(),
                        'min_ms': timer.min_time,
                        'max_ms': timer.max_time,
                        'total_calls': timer.call_count,
                        'total_time_ms': timer.total_time
                    }
            
            return report
    
    def reset_statistics(self):
        """統計をリセット"""
        with self.lock:
            for timer in self.phase_timers.values():
                timer.reset()
            self.metrics_history.clear()
            self.frame_times.clear()
            self.current_metrics = PerformanceMetrics()
            logger.info("Performance statistics reset")
    
    def print_report(self):
        """パフォーマンスレポートをコンソール出力"""
        report = self.get_performance_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # サマリー
        summary = report['summary']
        print(f"FPS: {summary['current_fps']:.1f} (avg: {summary['avg_fps_30']:.1f})")
        print(f"Frame Time: {summary['frame_time_ms']:.1f}ms")
        print(f"CPU: {summary['cpu_percent']:.1f}%")
        print(f"Memory: {summary['memory_mb']:.1f}MB (peak: {summary['peak_memory_mb']:.1f}MB)")
        
        # ボトルネック
        if report['bottlenecks']:
            print(f"\n🚨 BOTTLENECKS (>{10.0}ms):")
            for bottleneck in report['bottlenecks']:
                print(f"  • {bottleneck}")
        
        # フェーズ詳細
        print(f"\n📊 PHASE BREAKDOWN:")
        phases = report['phases']
        for phase_name, stats in phases.items():
            print(f"  {phase_name}: {stats['avg_ms']:.1f}ms "
                  f"(min: {stats['min_ms']:.1f}, max: {stats['max_ms']:.1f}, "
                  f"calls: {stats['total_calls']})")
        
        print("="*60)


# グローバルプロファイラーインスタンス
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """グローバルプロファイラーを取得"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_phase(phase_name: str):
    """フェーズプロファイリングデコレータ"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.measure_phase(phase_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def measure_performance(phase_name: str):
    """パフォーマンス測定コンテキストマネージャー"""
    profiler = get_profiler()
    with profiler.measure_phase(phase_name):
        yield


def benchmark_function(func: Callable, iterations: int = 100) -> Dict[str, float]:
    """関数のベンチマーク実行"""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    
    return {
        'avg_ms': np.mean(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'std_ms': np.std(times),
        'iterations': iterations
    } 