#!/usr/bin/env python3
"""
リアルタイム パフォーマンス プロファイラ

Geocussion-SP アプリケーションの各フェーズを監視し、
ボトルネックの自動特定とリアルタイム最適化提案を行います。

主要機能:
- フレーム毎の処理時間測定
- CPU/GPU/メモリ使用率監視
- ボトルネック自動検出
- 最適化提案システム
- JSON レポート出力
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import psutil

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .. import get_logger

logger = get_logger(__name__)


@dataclass
class PhaseMetrics:
    """フェーズ計測結果"""
    name: str
    execution_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    timestamp: float = 0.0
    frame_number: int = 0


@dataclass 
class BottleneckAnalysis:
    """ボトルネック分析結果"""
    phase_name: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    issue_type: str  # "CPU", "GPU", "MEMORY", "TIME"
    description: str
    recommendation: str
    impact_score: float  # 0-100


@dataclass
class PerformanceReport:
    """性能レポート"""
    timestamp: float
    total_fps: float
    target_fps: float
    total_frame_time_ms: float
    phase_metrics: List[PhaseMetrics]
    bottlenecks: List[BottleneckAnalysis]
    optimization_suggestions: List[str]
    system_info: Dict[str, Any]


class PerformanceProfiler:
    """リアルタイム パフォーマンス プロファイラ"""
    
    def __init__(
        self,
        target_fps: float = 30.0,
        history_size: int = 60,
        analysis_interval: float = 2.0,
        enable_real_time_display: bool = True,
        enable_json_logging: bool = True
    ):
        """
        初期化
        
        Args:
            target_fps: 目標FPS
            history_size: 履歴保持フレーム数
            analysis_interval: 分析実行間隔 (秒)
            enable_real_time_display: リアルタイム表示の有効化
            enable_json_logging: JSON ログの有効化
        """
        self.target_fps = target_fps
        self.history_size = history_size
        self.analysis_interval = analysis_interval
        self.enable_real_time_display = enable_real_time_display
        self.enable_json_logging = enable_json_logging
        
        # 計測状態
        self.is_profiling = False
        self.current_frame = 0
        self.frame_start_time = 0.0
        self.current_phase = None
        self.phase_start_time = 0.0
        
        # データ蓄積
        self.frame_history: deque = deque(maxlen=history_size)
        self.phase_history: Dict[str, deque] = {}
        self.current_frame_phases: List[PhaseMetrics] = []
        
        # システム監視
        self.system_monitor = SystemMonitor()
        
        # 分析システム
        self.bottleneck_analyzer = BottleneckAnalyzer(target_fps)
        
        # 表示スレッド
        self.display_thread: Optional[threading.Thread] = None
        self.stop_display = threading.Event()
        
        # レポート
        self.latest_report: Optional[PerformanceReport] = None
    
    def start_profiling(self):
        """プロファイリング開始"""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.current_frame = 0
        logger.info("Performance profiling started")
        
        # リアルタイム表示スレッド開始
        if self.enable_real_time_display:
            self.stop_display.clear()
            self.display_thread = threading.Thread(
                target=self._display_loop, 
                daemon=True
            )
            self.display_thread.start()
    
    def stop_profiling(self) -> Optional[PerformanceReport]:
        """プロファイリング停止"""
        if not self.is_profiling:
            return None
        
        self.is_profiling = False
        
        # 表示スレッド停止
        if self.display_thread:
            self.stop_display.set()
            self.display_thread.join(timeout=1.0)
        
        # 最終レポート生成
        final_report = self.generate_report()
        
        logger.info("Performance profiling stopped")
        
        if self.enable_json_logging and final_report:
            self._save_json_report(final_report)
        
        return final_report
    
    def start_frame(self):
        """フレーム計測開始"""
        if not self.is_profiling:
            return
        
        self.frame_start_time = time.perf_counter()
        self.current_frame += 1
        self.current_frame_phases.clear()
    
    def end_frame(self):
        """フレーム計測終了"""
        if not self.is_profiling:
            return
        
        frame_time = (time.perf_counter() - self.frame_start_time) * 1000
        current_fps = 1000.0 / max(frame_time, 1.0)
        
        # システム使用率取得
        cpu_usage = self.system_monitor.get_cpu_usage()
        memory_usage = self.system_monitor.get_memory_usage()
        gpu_usage, gpu_memory = self.system_monitor.get_gpu_usage()
        
        # フレーム記録
        frame_data = {
            "frame_number": self.current_frame,
            "frame_time_ms": frame_time,
            "fps": current_fps,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage,
            "gpu_memory": gpu_memory,
            "timestamp": time.time(),
            "phases": self.current_frame_phases.copy()
        }
        
        self.frame_history.append(frame_data)
        
        # 定期的分析実行
        if len(self.frame_history) >= 30:  # 最低30フレーム蓄積後
            self._perform_analysis()
    
    def start_phase(self, phase_name: str):
        """フェーズ計測開始"""
        if not self.is_profiling:
            return
        
        self.current_phase = phase_name
        self.phase_start_time = time.perf_counter()
        
        if phase_name not in self.phase_history:
            self.phase_history[phase_name] = deque(maxlen=self.history_size)
    
    def end_phase(self):
        """フェーズ計測終了"""
        if not self.is_profiling or not self.current_phase:
            return
        
        phase_time = (time.perf_counter() - self.phase_start_time) * 1000
        
        # システム使用率取得
        cpu_usage = self.system_monitor.get_cpu_usage()
        memory_usage = self.system_monitor.get_memory_usage()
        gpu_usage, gpu_memory = self.system_monitor.get_gpu_usage()
        
        # フェーズ記録
        phase_metrics = PhaseMetrics(
            name=self.current_phase,
            execution_time_ms=phase_time,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            gpu_usage_percent=gpu_usage,
            gpu_memory_mb=gpu_memory,
            timestamp=time.time(),
            frame_number=self.current_frame
        )
        
        self.current_frame_phases.append(phase_metrics)
        self.phase_history[self.current_phase].append(phase_metrics)
        
        self.current_phase = None
    
    def _perform_analysis(self):
        """パフォーマンス分析実行"""
        if len(self.frame_history) < 10:
            return
        
        # 最新レポート生成
        self.latest_report = self.generate_report()
    
    def generate_report(self) -> PerformanceReport:
        """パフォーマンスレポート生成"""
        if not self.frame_history:
            return PerformanceReport(
                timestamp=time.time(),
                total_fps=0.0,
                target_fps=self.target_fps,
                total_frame_time_ms=0.0,
                phase_metrics=[],
                bottlenecks=[],
                optimization_suggestions=[],
                system_info=self.system_monitor.get_system_info()
            )
        
        # 統計計算
        recent_frames = list(self.frame_history)[-30:]  # 最新30フレーム
        
        avg_fps = np.mean([f["fps"] for f in recent_frames])
        avg_frame_time = np.mean([f["frame_time_ms"] for f in recent_frames])
        
        # フェーズ統計
        phase_stats = []
        for phase_name, phase_data in self.phase_history.items():
            if phase_data:
                recent_phase_data = list(phase_data)[-10:]  # 最新10エントリ
                avg_phase_time = np.mean([p.execution_time_ms for p in recent_phase_data])
                
                avg_metrics = PhaseMetrics(
                    name=phase_name,
                    execution_time_ms=avg_phase_time,
                    cpu_usage_percent=np.mean([p.cpu_usage_percent for p in recent_phase_data]),
                    memory_usage_mb=np.mean([p.memory_usage_mb for p in recent_phase_data]),
                    gpu_usage_percent=np.mean([p.gpu_usage_percent for p in recent_phase_data]),
                    gpu_memory_mb=np.mean([p.gpu_memory_mb for p in recent_phase_data]),
                    timestamp=time.time(),
                    frame_number=self.current_frame
                )
                phase_stats.append(avg_metrics)
        
        # ボトルネック分析
        bottlenecks = self.bottleneck_analyzer.analyze(recent_frames, phase_stats)
        
        # 最適化提案生成
        suggestions = self._generate_optimization_suggestions(bottlenecks, phase_stats)
        
        return PerformanceReport(
            timestamp=time.time(),
            total_fps=avg_fps,
            target_fps=self.target_fps,
            total_frame_time_ms=avg_frame_time,
            phase_metrics=phase_stats,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            system_info=self.system_monitor.get_system_info()
        )
    
    def _generate_optimization_suggestions(
        self, 
        bottlenecks: List[BottleneckAnalysis],
        phase_stats: List[PhaseMetrics]
    ) -> List[str]:
        """最適化提案生成"""
        suggestions = []
        
        # FPS不足の場合
        if hasattr(self, 'latest_report') and self.latest_report:
            if self.latest_report.total_fps < self.target_fps * 0.8:  # 80%未満
                suggestions.append(f"FPS目標未達成: {self.latest_report.total_fps:.1f}/{self.target_fps} FPS")
        
        # ボトルネック基盤提案
        for bottleneck in bottlenecks:
            if bottleneck.severity in ["CRITICAL", "HIGH"]:
                suggestions.append(f"{bottleneck.phase_name}: {bottleneck.recommendation}")
        
        # フェーズ別提案
        for phase in phase_stats:
            if phase.execution_time_ms > 20:  # 20ms超過
                suggestions.append(f"{phase.name}フェーズが遅い ({phase.execution_time_ms:.1f}ms)")
            
            if phase.cpu_usage_percent > 80:
                suggestions.append(f"{phase.name}でCPU高負荷 ({phase.cpu_usage_percent:.1f}%)")
            
            if phase.gpu_usage_percent > 90:
                suggestions.append(f"{phase.name}でGPU高負荷 ({phase.gpu_usage_percent:.1f}%)")
        
        return suggestions
    
    def _display_loop(self):
        """リアルタイム表示ループ"""
        while not self.stop_display.wait(self.analysis_interval):
            if self.latest_report:
                self._display_real_time_stats(self.latest_report)
    
    def _display_real_time_stats(self, report: PerformanceReport):
        """リアルタイム統計表示"""
        print(f"\n==== Performance Stats (Frame {self.current_frame}) ====")
        print(f"FPS: {report.total_fps:.1f}/{report.target_fps} ({report.total_frame_time_ms:.1f}ms)")
        
        if report.phase_metrics:
            print("Phase Times:")
            for phase in sorted(report.phase_metrics, key=lambda p: p.execution_time_ms, reverse=True):
                print(f"  {phase.name}: {phase.execution_time_ms:.1f}ms")
        
        if report.bottlenecks:
            critical_bottlenecks = [b for b in report.bottlenecks if b.severity in ["CRITICAL", "HIGH"]]
            if critical_bottlenecks:
                print("Critical Issues:")
                for bottleneck in critical_bottlenecks:
                    print(f"  {bottleneck.phase_name}: {bottleneck.description}")
        
        print("=" * 50)
    
    def _save_json_report(self, report: PerformanceReport):
        """JSON レポート保存"""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp_str}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            logger.info(f"Performance report saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


class SystemMonitor:
    """システムリソース監視"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_cpu_usage(self) -> float:
        """CPU使用率取得"""
        return self.process.cpu_percent()
    
    def get_memory_usage(self) -> float:
        """メモリ使用量取得 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_gpu_usage(self) -> tuple[float, float]:
        """GPU使用率とメモリ取得"""
        if not GPU_AVAILABLE:
            return 0.0, 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 最初のGPU
                return gpu.load * 100, gpu.memoryUsed
        except:
            pass
        
        return 0.0, 0.0
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "gpu_available": GPU_AVAILABLE,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }


class BottleneckAnalyzer:
    """ボトルネック分析器"""
    
    def __init__(self, target_fps: float):
        self.target_fps = target_fps
        self.target_frame_time = 1000.0 / target_fps
    
    def analyze(
        self, 
        frame_data: List[Dict], 
        phase_stats: List[PhaseMetrics]
    ) -> List[BottleneckAnalysis]:
        """ボトルネック分析実行"""
        bottlenecks = []
        
        # フレーム時間分析
        avg_frame_time = np.mean([f["frame_time_ms"] for f in frame_data])
        if avg_frame_time > self.target_frame_time * 1.2:  # 20%超過
            severity = "CRITICAL" if avg_frame_time > self.target_frame_time * 1.5 else "HIGH"
            bottlenecks.append(BottleneckAnalysis(
                phase_name="Overall",
                severity=severity,
                issue_type="TIME",
                description=f"Frame time {avg_frame_time:.1f}ms exceeds target {self.target_frame_time:.1f}ms",
                recommendation="Optimize slowest phases or reduce quality settings",
                impact_score=min(100, (avg_frame_time / self.target_frame_time - 1) * 100)
            ))
        
        # フェーズ別分析
        for phase in phase_stats:
            # 時間ボトルネック
            if phase.execution_time_ms > 15:  # 15ms閾値
                severity = "CRITICAL" if phase.execution_time_ms > 25 else "HIGH"
                bottlenecks.append(BottleneckAnalysis(
                    phase_name=phase.name,
                    severity=severity,
                    issue_type="TIME", 
                    description=f"Phase takes {phase.execution_time_ms:.1f}ms",
                    recommendation=self._get_phase_recommendation(phase.name, "TIME"),
                    impact_score=min(100, phase.execution_time_ms * 2)
                ))
            
            # CPU ボトルネック
            if phase.cpu_usage_percent > 80:
                bottlenecks.append(BottleneckAnalysis(
                    phase_name=phase.name,
                    severity="HIGH",
                    issue_type="CPU",
                    description=f"High CPU usage {phase.cpu_usage_percent:.1f}%",
                    recommendation=self._get_phase_recommendation(phase.name, "CPU"),
                    impact_score=phase.cpu_usage_percent - 50
                ))
            
            # GPU ボトルネック
            if phase.gpu_usage_percent > 90:
                bottlenecks.append(BottleneckAnalysis(
                    phase_name=phase.name,
                    severity="HIGH",
                    issue_type="GPU",
                    description=f"High GPU usage {phase.gpu_usage_percent:.1f}%",
                    recommendation=self._get_phase_recommendation(phase.name, "GPU"),
                    impact_score=phase.gpu_usage_percent - 60
                ))
        
        return sorted(bottlenecks, key=lambda b: b.impact_score, reverse=True)
    
    def _get_phase_recommendation(self, phase_name: str, issue_type: str) -> str:
        """フェーズ別推奨事項"""
        recommendations = {
            ("hand_detection", "TIME"): "MediaPipe confidence threshold increase or skip frames",
            ("hand_detection", "CPU"): "Reduce input resolution or detection frequency",
            ("mesh_generation", "TIME"): "Use incremental mesh updates or simplify",
            ("mesh_generation", "CPU"): "Enable mesh caching or reduce triangle count",
            ("collision_detection", "TIME"): "Use broadphase optimization with KD-Tree",
            ("collision_detection", "CPU"): "Enable GPU acceleration or reduce precision",
            ("distance_calculation", "TIME"): "Enable Numba JIT or GPU calculation",
            ("distance_calculation", "GPU"): "Optimize batch size or use mixed precision"
        }
        
        return recommendations.get((phase_name.lower(), issue_type), 
                                 f"Optimize {phase_name} for {issue_type} usage")


# 便利関数とコンテキストマネージャ

class ProfiledPhase:
    """フェーズプロファイリング用コンテキストマネージャ"""
    
    def __init__(self, profiler: PerformanceProfiler, phase_name: str):
        self.profiler = profiler
        self.phase_name = phase_name
    
    def __enter__(self):
        self.profiler.start_phase(self.phase_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_phase()


def profile_function(profiler: PerformanceProfiler, phase_name: str):
    """関数プロファイリング用デコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ProfiledPhase(profiler, phase_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# グローバルプロファイラインスタンス
global_profiler: Optional[PerformanceProfiler] = None

def get_global_profiler() -> PerformanceProfiler:
    """グローバルプロファイラ取得"""
    global global_profiler
    if global_profiler is None:
        global_profiler = PerformanceProfiler()
    return global_profiler

def start_global_profiling():
    """グローバルプロファイリング開始"""
    get_global_profiler().start_profiling()

def stop_global_profiling() -> Optional[PerformanceReport]:
    """グローバルプロファイリング停止"""
    if global_profiler:
        return global_profiler.stop_profiling()
    return None 