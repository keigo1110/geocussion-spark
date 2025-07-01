#!/usr/bin/env python3
"""
Geocussion-SP リアルタイムデモ（衝突検出統合版）
点群→メッシュ→衝突検出→音響合成の全パイプライン実装

使用方法:
  python3 demo_collision_detection.py                 # 通常実行
  python3 demo_collision_detection.py --no-audio      # 音響無効化
  python3 demo_collision_detection.py --test          # プリプロセッシング最適化テスト
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import signal
import numpy as np
import cv2

# Constants
DEFAULT_SPHERE_RADIUS = 0.05  # 5cm
DEFAULT_MESH_UPDATE_INTERVAL = 15
DEFAULT_MAX_MESH_SKIP_FRAMES = 60
DEFAULT_AUDIO_COOLDOWN_TIME = 0.3  # 300ms (debounce)
DEFAULT_VOXEL_SIZE = 0.005  # 5mm
DEFAULT_AUDIO_POLYPHONY = 16
DEFAULT_MASTER_VOLUME = 0.7
LOW_RESOLUTION = (424, 240)
HIGH_RESOLUTION = (848, 480)
ESTIMATED_HIGH_RES_POINTS = 300000

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 動的インポートとモックの定義
# =============================================================================

# OrbbecSDKの動的インポート
HAS_ORBBEC_SDK = False
try:
    # type: ignore[import]
    from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBError, OBFormat  # type: ignore  # pylint: disable=import-error
    HAS_ORBBEC_SDK = True
    logger.info("OrbbecSDK is available")
except ImportError:
    logger.warning("OrbbecSDK is not available. Using mock implementations.")
    # Mock classes
    class Pipeline:
        def __init__(self): pass
        def start(self, config): pass
        def stop(self): pass
        def wait_for_frames(self, timeout): return None
    
    class FrameSet:
        def get_depth_frame(self): return None
        def get_color_frame(self): return None
    
    class Config:
        def enable_stream(self, stream, width, height, fmt, fps): pass
    
    class OBSensorType:
        DEPTH = "depth"
        COLOR = "color"
    
    class OBError(Exception):
        pass
    
    class OBFormat:
        RGB = "rgb"
        BGR = "bgr"
        MJPG = "mjpg"

# MediaPipeの動的インポート
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    logger.info("MediaPipe is available")
except ImportError:
    logger.warning("MediaPipe is not available. Hand detection will be disabled.")

# Open3Dの動的インポート
HAS_OPEN3D = False
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    logger.warning("Open3D is not available. 3D visualization will be disabled.")

# 音響ライブラリの動的インポート
HAS_AUDIO = False
try:
    import pyo
    HAS_AUDIO = True
    logger.info("Pyo audio engine is available")
except ImportError:
    logger.warning("Pyo audio engine is not available. Audio synthesis will be disabled.")

# Numba JIT最適化の初期化
def initialize_numba_optimization():
    """Numba JIT最適化を初期化"""
    try:
        sys.path.insert(0, str(project_root / "src"))
        from src.numba_config import initialize_numba, get_numba_status, warmup_basic_functions
        
        logger.info("Starting Numba initialization...")
        success = initialize_numba(verbose=True, force_retry=True)
        if success:
            status = get_numba_status()
            logger.info(f"Numba JIT acceleration enabled (v{status['version']})")
            logger.info("Warming up JIT functions...")
            warmup_basic_functions()
            logger.info("JIT functions warmed up - maximum performance ready")
        else:
            logger.warning("Numba JIT acceleration disabled (falling back to NumPy)")
    except Exception as e:
        logger.warning(f"Numba configuration error: {e}")
        logger.warning("Using NumPy fallback for all computations")

# Numba初期化を実行
initialize_numba_optimization()

# 必要なクラスのimport
from src.detection.tracker import TrackedHand
from src.sound.mapping import ScaleType, InstrumentType
from src.mesh.projection import PointCloudProjector, ProjectionMethod
from src.mesh.delaunay import DelaunayTriangulator
from src.mesh.simplify import MeshSimplifier
from src.mesh.index import SpatialIndex, IndexType
from src.collision.search import CollisionSearcher
from src.collision.sphere_tri import SphereTriangleCollision
from src.collision.events import CollisionEventQueue
from src.sound.mapping import AudioMapper
from src.detection.hands2d import MediaPipeHandsWrapper
from src.input.stream import OrbbecCamera
from src.detection.hands3d import Hand3DProjector
from src.detection.tracker import Hand3DTracker
from src.sound.synth import AudioSynthesizer, AudioConfig, EngineState, create_audio_synthesizer
from src.sound.voice_mgr import VoiceManager, StealStrategy, SpatialMode, SpatialConfig, create_voice_manager, allocate_and_play
from src.debug.dual_viewer import DualViewer
from src.input.depth_filter import DepthFilter, FilterType
from src.input.pointcloud import PointCloudConverter
from src.config import get_config, InputConfig
from src.mesh.pipeline import create_mesh_pipeline  # unified mesh pipeline
from src.mesh.manager import PipelineManager  # NEW pipeline manager

# GPU加速コンポーネントの動的インポート
HAS_GPU_ACCELERATION = False
try:
    from src.collision.distance_gpu import GPUDistanceCalculator, create_gpu_distance_calculator
    from src.mesh.delaunay_gpu import GPUDelaunayTriangulator, create_gpu_triangulator
    HAS_GPU_ACCELERATION = True
    logger.info("GPU acceleration modules loaded (CuPy available)")
except ImportError:
    logger.warning("GPU acceleration unavailable (CuPy not installed)")

# =============================================================================
# テスト関数
# =============================================================================

def run_preprocessing_optimization_test():
    """前処理最適化効果測定テスト"""
    def mock_mediapipe_process(image):
        """MediaPipe処理のモック"""
        time.sleep(0.015)  # ~15ms処理時間シミュレーション
        return []
    
    print("=" * 70)
    print("前処理最適化効果 測定テスト")
    print("=" * 70)
    
    # モック深度データ作成
    depth_low = np.random.randint(500, 2000, LOW_RESOLUTION[::-1], dtype=np.uint16)
    depth_high = np.random.randint(500, 2000, (480, 848), dtype=np.uint16)
    color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # ケース1: 高解像度 + MediaPipe重複実行
    print("🔍 ケース1: 848x480 + MediaPipe重複実行")
    fps_case1 = _measure_fps_for_test_case(
        depth_high, color_image, mock_mediapipe_process, 
        num_mediapipe_calls=2, target_frame_time=0.075
    )
    
    # ケース2: 低解像度 + MediaPipe1回実行
    print("🔍 ケース2: 424x240 + MediaPipe1回実行")
    fps_case2 = _measure_fps_for_test_case(
        depth_low, color_image, mock_mediapipe_process,
        num_mediapipe_calls=1, target_frame_time=0.036
    )
    
    # 結果表示
    _display_test_results("前処理最適化効果", fps_case1, fps_case2)

def run_headless_fps_comparison_test():
    """ヘッドレスモードFPS効果測定テスト"""
    def mock_core_processing():
        """コア処理のシミュレーション"""
        time.sleep(0.025)  # 25ms処理時間
    
    def mock_gui_rendering():
        """GUI描画処理のシミュレーション"""
        time.sleep(0.008)  # Open3D 3D描画
        time.sleep(0.003)  # OpenCV RGB表示
        time.sleep(0.002)  # UI更新
    
    print("=" * 70)
    print("ヘッドレスモード FPS効果 測定テスト")
    print("=" * 70)
    
    # GUI有りモード測定
    print("🖥️  GUI有りモード測定中...")
    fps_gui = _measure_fps_with_processing(
        mock_core_processing, mock_gui_rendering, num_frames=100
    )
    
    # ヘッドレスモード測定
    print("⚡ ヘッドレスモード測定中...")
    fps_headless = _measure_fps_with_processing(
        mock_core_processing, None, num_frames=100
    )
    
    # 結果表示
    _display_test_results("ヘッドレスモード FPS効果", fps_gui, fps_headless)

def _measure_fps_for_test_case(depth_data, color_image, mediapipe_func, 
                               num_mediapipe_calls, target_frame_time):
    """テストケースのFPSを測定"""
    start_time = time.time()
    frames = 0
    
    for _ in range(50):  # 50フレーム測定
        frame_start = time.time()
        
        # 点群処理シミュレーション
        points = depth_data.reshape(-1)
        valid_points = points[points > 0]
        
        # MediaPipe実行
        for _ in range(num_mediapipe_calls):
            mediapipe_func(color_image)
        
        frame_time = time.time() - frame_start
        frames += 1
        
        # フレーム時間調整
        if frame_time < target_frame_time:
            time.sleep(target_frame_time - frame_time)
    
    elapsed = time.time() - start_time
    return frames / elapsed

def _measure_fps_with_processing(core_func, gui_func, num_frames):
    """処理関数のFPSを測定"""
    start_time = time.time()
    
    for _ in range(num_frames):
        core_func()
        if gui_func:
            gui_func()
    
    elapsed = time.time() - start_time
    return num_frames / elapsed

def _display_test_results(test_name, fps1, fps2):
    """テスト結果を表示"""
    print(f"\n📊 {test_name} 結果")
    print("=" * 50)
    print(f"ケース1: {fps1:.1f} FPS")
    print(f"ケース2: {fps2:.1f} FPS")
    print(f"改善倍率: {fps2/fps1:.1f}x")
    print(f"FPS向上: +{fps2-fps1:.1f} FPS")
    print(f"フレーム時間短縮: {(1/fps1-1/fps2)*1000:.1f}ms")

# =============================================================================
# メインビューワークラス
# =============================================================================

class FullPipelineViewer(DualViewer):
    """全フェーズ統合拡張DualViewer（手検出+メッシュ生成+衝突検出+音響生成）"""
    
    def __init__(self, **kwargs):
        # 設定値の抽出
        self._extract_configuration(kwargs)
        
        # 親クラス初期化
        super().__init__(**kwargs)
        
        # コンポーネントの初期化
        self._initialize_components()
        
        # 状態管理の初期化
        self._initialize_state()
        
        # パフォーマンス統計の初期化
        self._initialize_performance_stats()
        
        # GPU加速の初期化
        self._initialize_gpu_acceleration()
        
        # ヘルプテキストの更新
        self.update_help_text()
        
        # 初期化完了フラグ
        self._components_initialized = False
        
        self._display_initialization_info()
    
    def _extract_configuration(self, kwargs):
        """設定値を抽出"""
        # 音響関連パラメータ
        self.enable_audio_synthesis = kwargs.pop('enable_audio_synthesis', True)
        self.audio_scale = kwargs.pop('audio_scale', ScaleType.PENTATONIC)
        self.audio_instrument = kwargs.pop('audio_instrument', InstrumentType.MARIMBA)
        self.audio_polyphony = kwargs.pop('audio_polyphony', DEFAULT_AUDIO_POLYPHONY)
        self.audio_master_volume = kwargs.pop('audio_master_volume', DEFAULT_MASTER_VOLUME)
        
        # ヘッドレスモード設定
        self.headless_mode = kwargs.pop('headless_mode', False)
        self.headless_duration = kwargs.pop('headless_duration', 30)
        self.pure_headless_mode = kwargs.pop('pure_headless_mode', False)
        
        # 衝突検出パラメータ
        self.enable_mesh_generation = kwargs.pop('enable_mesh_generation', True)
        self.enable_collision_detection = kwargs.pop('enable_collision_detection', True)
        self.enable_collision_visualization = kwargs.pop('enable_collision_visualization', True)
        self.sphere_radius = kwargs.pop('sphere_radius', DEFAULT_SPHERE_RADIUS)
        
        # メッシュ更新間隔制御
        self.mesh_update_interval = kwargs.pop('mesh_update_interval', DEFAULT_MESH_UPDATE_INTERVAL)
        self.max_mesh_skip_frames = kwargs.pop('max_mesh_skip_frames', DEFAULT_MAX_MESH_SKIP_FRAMES)
        
        # ボクセルダウンサンプリングパラメータ
        self.voxel_downsampling_enabled = kwargs.pop('enable_voxel_downsampling', True)
        self.voxel_size = kwargs.pop('voxel_size', DEFAULT_VOXEL_SIZE)
    
        # MediaPipe GPU 使用設定
        self.use_gpu_mediapipe = kwargs.pop('use_gpu_mediapipe', False)
    
    def _initialize_components(self):
        """コンポーネントを初期化"""
        # ヘルプテキスト
        self.help_text = ""
        
        # 地形メッシュ生成コンポーネント
        self.projector = PointCloudProjector(
            resolution=0.01,  # 1cm解像度
            method=ProjectionMethod.MEDIAN_HEIGHT,
            fill_holes=True,
            plane="xz",  # カメラ座標系に合わせて XZ 平面投影
        )
        
        # LODメッシュ生成器
        self._initialize_mesh_generators()
        
        # メッシュ簡略化
        self.simplifier = MeshSimplifier(
            target_reduction=0.7,  # 70%削減
            preserve_boundary=True
        )
        
        # 衝突検出コンポーネント
        self.spatial_index: Optional[SpatialIndex] = None
        self.collision_searcher: Optional[CollisionSearcher] = None
        self.collision_tester: Optional[SphereTriangleCollision] = None
        self.event_queue: CollisionEventQueue = CollisionEventQueue()
        
        # 音響生成コンポーネント
        self.audio_mapper: Optional[AudioMapper] = None
        self.audio_synthesizer: Optional[AudioSynthesizer] = None
        self.voice_manager: Optional[VoiceManager] = None
        self.audio_enabled = False
        
        # 音響クールダウン管理
        self.audio_cooldown_time = DEFAULT_AUDIO_COOLDOWN_TIME
        self.last_audio_trigger_time = {}
        # 衝突デバウンス用: (hand_id, triangle_idx) -> last trigger time
        from typing import Tuple as _Tuple
        self._last_contact_trigger_time: Dict[_Tuple[str, int, int, int], float] = {}
    
        # 手検出コンポーネント
        self.enable_hand_detection = True
        self.enable_hand_tracking = True
        self.enable_tracking = True
        self.min_detection_confidence = 0.2
        self._initialize_hand_detection()
        
        # 3Dコンポーネント（後で初期化）
        self.projector_3d = None
        self.tracker = None
    
    def _initialize_mesh_generators(self):
        """メッシュ生成器を初期化 (T-MESH-101)"""

        # LOD メッシュ生成器（MeshPipeline 内部でも使用）
        from src.mesh.lod_mesh import create_lod_mesh_generator  # 遅延 import で循環回避
        self.lod_mesh_generator = create_lod_mesh_generator(
            high_radius=0.20,
            medium_radius=0.50,
            enable_gpu=True,
        )

        # 従来 Triangulator – 直接呼び出し箇所残存のため保持
        self.triangulator = DelaunayTriangulator(
            adaptive_sampling=True,
            boundary_points=True,
            quality_threshold=0.3,
            use_gpu=True,
        )

        # 統合 MeshPipeline
        self.mesh_pipeline = create_mesh_pipeline(enable_incremental=False)
        self.pipeline_manager = PipelineManager(self.mesh_pipeline)
        self._mesh_version = -1  # track version for viewer refresh
    
    def _initialize_hand_detection(self):
        """手検出コンポーネントを初期化"""
        self.hands_2d = MediaPipeHandsWrapper(
            use_gpu=self.use_gpu_mediapipe,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
            max_num_hands=2,
            # ROI トラッキング設定（効率化）
            enable_roi_tracking=True,
            tracker_type="KCF",           # KCFトラッカーで高速化
            skip_interval=4,              # 4フレームに1回MediaPipe実行
            roi_confidence_threshold=0.6,
            max_tracking_age=15
        )
        return True
    
    def _initialize_state(self):
        """状態管理を初期化"""
        self.current_mesh = None
        self.current_collision_points = []
        self.current_tracked_hands = []
        self.current_hands_2d = []
        self.current_hands_3d = []
        self.frame_counter = 0
        self.last_mesh_update = -999  # 初回メッシュ生成を確実にする
        self.force_mesh_update_requested = False
        
        # メッシュとコリジョンの可視化オブジェクト
        self.mesh_geometries = []
        self.collision_geometries = []
        
        # 直近の点群 / 深度フレームを保持
        self._last_points_3d: Optional[np.ndarray] = None
        self._latest_depth_image: Optional[np.ndarray] = None
    
    def _initialize_performance_stats(self):
        """パフォーマンス統計を初期化"""
        self.perf_stats = {
            'frame_count': 0,
            'mesh_generation_time': 0.0,
            'collision_detection_time': 0.0,
            'audio_synthesis_time': 0.0,
            'collision_events_count': 0,
            'audio_notes_played': 0,
            'total_pipeline_time': 0.0
        }
        
        # 基本パフォーマンス統計（親クラス互換）
        self.performance_stats = {
            'fps': 0.0,
            'frame_time': 0.0,
            'filter_time': 0.0,
            'pointcloud_time': 0.0,
            'hand_detection_time': 0.0,
            # DualViewer のオーバーレイで参照される追加キー（初期化しておく）
            'hand_projection_time': 0.0,
            'hand_tracking_time': 0.0,
            'display_time': 0.0,
            'hands_detected': 0,
            'hands_tracked': 0
        }
        
        # GPU統計
        self.gpu_stats = {
            'distance_calculations': 0,
            'triangulations': 0,
            'gpu_time_total_ms': 0.0,
            'cpu_fallbacks': 0
        }
    
    def _initialize_gpu_acceleration(self):
        """GPU加速を初期化"""
        self.gpu_distance_calc = None
        self.gpu_triangulator = None
        self.gpu_acceleration_enabled = False
        
        if not HAS_GPU_ACCELERATION:
            return
        
        try:
            # GPU距離計算器
            self.gpu_distance_calc = create_gpu_distance_calculator(
                use_gpu=True,
                batch_size=10000,
                memory_limit_ratio=0.8
            )
            
            # GPU三角分割器
            self.gpu_triangulator = create_gpu_triangulator(
                use_gpu=True,
                quality_threshold=0.2,
                enable_caching=True
            )
            
            # 実際にGPUが使えるかチェック
            gpu_calc_available = (hasattr(self.gpu_distance_calc, 'gpu_available') and 
                                self.gpu_distance_calc.gpu_available)
            gpu_tri_available = (hasattr(self.gpu_triangulator, 'use_gpu') and 
                               self.gpu_triangulator.use_gpu)
            
            self.gpu_acceleration_enabled = gpu_calc_available or gpu_tri_available
            
            if self.gpu_acceleration_enabled:
                logger.info("🚀 GPU acceleration initialized successfully")
                logger.info(f"  - Distance Calculator: {'GPU' if gpu_calc_available else 'CPU fallback'}")
                logger.info(f"  - Triangulator: {'GPU' if gpu_tri_available else 'CPU fallback'}")
            else:
                logger.warning("GPU acceleration components initialized but GPU not available")
                
        except Exception as e:
            logger.warning(f"GPU acceleration initialization failed: {e}")
            logger.warning("Falling back to CPU-only processing")
    
    def _display_initialization_info(self):
        """初期化情報を表示"""
        print("全フェーズ統合ビューワーが初期化されました")
        print(f"  - メッシュ生成: {'有効' if self.enable_mesh_generation else '無効'}")
        print(f"  - 衝突検出: {'有効' if self.enable_collision_detection else '無効'}")
        print(f"  - 接触点可視化: {'有効' if self.enable_collision_visualization else '無効'}")
        print(f"  - 球半径: {self.sphere_radius*100:.1f}cm")
        print(f"  - 音響合成: {'有効' if self.enable_audio_synthesis else '無効'}")
        
        if self.enable_audio_synthesis:
            print(f"    - 音階: {self.audio_scale.value}")
            print(f"    - 楽器: {self.audio_instrument.value}")
            print(f"    - ポリフォニー: {self.audio_polyphony}")
            print(f"    - 音量: {self.audio_master_volume:.1f}")
            print(f"    - エンジン状態: {'動作中' if self.audio_enabled else '停止中'}")
        
        # 音響システム初期化
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
    
    def update_help_text(self):
        """ヘルプテキストを更新"""
        help_sections = [
            ("Basic Controls", [
                "Q/ESC: Exit",
                "F: Toggle filter",
                "H: Toggle hand detection",
                "T: Toggle tracking",
                "R: Reset filter",
                "Y: Reset tracker"
            ]),
            ("Point Cloud Optimization", [
                "X: Toggle voxel downsampling",
                "Z/Shift+Z: Voxel size -/+ (1mm-10cm)",
                "B: Print voxel performance stats"
            ]),
            ("衝突検出制御", [
                "M: メッシュ生成 ON/OFF",
                "C: 衝突検出 ON/OFF",
                "V: 衝突可視化 ON/OFF",
                "N: メッシュ強制更新",
                "+/-: 球半径調整",
                "P: パフォーマンス統計表示"
            ]),
            ("音響生成制御", [
                "A: 音響合成 ON/OFF",
                "S: 音階切り替え",
                "I: 楽器切り替え",
                "1/2: 音量調整",
                "R: 音響エンジン再起動",
                "Q: 全音声停止"
            ])
        ]
        
        self.help_text = ""
        for section_name, commands in help_sections:
            self.help_text += f"\n=== {section_name} ===\n"
            for command in commands:
                self.help_text += f"{command}\n"
    
    def handle_key_event(self, key: int) -> bool:
        """キーイベント処理"""
        key_handlers = {
            # 基本操作
            ord('q'): lambda: False,
            27: lambda: False,  # ESC
            ord('f'): self._toggle_filter,
            ord('h'): self._toggle_hand_detection,
            ord('t'): self._toggle_tracking,
            ord('r'): self._reset_filter,
            ord('y'): self._reset_tracker,
            
            # ボクセルダウンサンプリング
            ord('x'): self._toggle_voxel_downsampling,
            ord('X'): self._toggle_voxel_downsampling,
            ord('z'): self._decrease_voxel_size,
            ord('Z'): self._increase_voxel_size,
            ord('b'): self._print_voxel_stats,
            ord('B'): self._print_voxel_stats,
            
            # 衝突検出
            ord('m'): self._toggle_mesh_generation,
            ord('M'): self._toggle_mesh_generation,
            ord('c'): self._toggle_collision_detection,
            ord('C'): self._toggle_collision_detection,
            ord('v'): self._toggle_collision_visualization,
            ord('V'): self._toggle_collision_visualization,
            ord('n'): self._force_mesh_update_request,
            ord('N'): self._force_mesh_update_request,
            ord('+'): self._increase_sphere_radius,
            ord('='): self._increase_sphere_radius,
            ord('-'): self._decrease_sphere_radius,
            ord('_'): self._decrease_sphere_radius,
            ord('p'): self._print_performance_stats,
            ord('P'): self._print_performance_stats,
            
            # 音響生成
            ord('a'): self._toggle_audio_synthesis,
            ord('A'): self._toggle_audio_synthesis,
            ord('s'): self._cycle_audio_scale,
            ord('S'): self._cycle_audio_scale,
            ord('i'): self._cycle_audio_instrument,
            ord('I'): self._cycle_audio_instrument,
            ord('1'): self._decrease_volume,
            ord('2'): self._increase_volume,
        }
        
        # 特殊キーの処理
        if key == ord('r') or key == ord('R'):
            if self.enable_audio_synthesis and (key == ord('R')):
                print("音響エンジンを再起動中...")
                self._restart_audio_system()
                return True
            else:
                return self._reset_filter()
        
        if key == ord('q') or key == ord('Q'):
            if self.enable_audio_synthesis and self.voice_manager and (key == ord('Q')):
                self.voice_manager.stop_all_voices()
                print("全音声を停止しました")
                return True
            else:
                return False
        
        # 通常のキーハンドラー実行
        handler = key_handlers.get(key)
        if handler:
            return handler()
        
        return True
    
    # キーハンドラーメソッド群
    def _toggle_filter(self) -> bool:
        self.enable_filter = not self.enable_filter
        print(f"Depth filter: {'Enabled' if self.enable_filter else 'Disabled'}")
        return True
    
    def _toggle_hand_detection(self) -> bool:
        self.enable_hand_detection = not self.enable_hand_detection
        print(f"Hand detection: {'Enabled' if self.enable_hand_detection else 'Disabled'}")
        return True
    
    def _toggle_tracking(self) -> bool:
        self.enable_tracking = not self.enable_tracking
        print(f"Hand tracking: {'Enabled' if self.enable_tracking else 'Disabled'}")
        return True
    
    def _reset_filter(self) -> bool:
        if self.depth_filter is not None:
            self.depth_filter.reset_temporal_history()
            print("Filter history reset")
        return True
    
    def _reset_tracker(self) -> bool:
        if self.tracker is not None:
            self.tracker.reset()
            print("Hand tracker reset")
        return True
    
    def _toggle_voxel_downsampling(self) -> bool:
        if self.pointcloud_converter:
            self.pointcloud_converter.toggle_voxel_downsampling()
        return True
    
    def _decrease_voxel_size(self) -> bool:
        if self.pointcloud_converter:
            current_size = self.pointcloud_converter.voxel_size
            new_size = max(0.001, current_size - 0.001)
            self.pointcloud_converter.set_voxel_size(new_size)
        return True
    
    def _increase_voxel_size(self) -> bool:
        if self.pointcloud_converter:
            current_size = self.pointcloud_converter.voxel_size
            new_size = min(0.05, current_size + 0.001)
            self.pointcloud_converter.set_voxel_size(new_size)
        return True
    
    def _print_voxel_stats(self) -> bool:
        if self.pointcloud_converter:
            self.pointcloud_converter.print_performance_stats()
        return True
    
    def _toggle_mesh_generation(self) -> bool:
        self.enable_mesh_generation = not self.enable_mesh_generation
        print(f"メッシュ生成: {'有効' if self.enable_mesh_generation else '無効'}")
        return True
    
    def _toggle_collision_detection(self) -> bool:
        self.enable_collision_detection = not self.enable_collision_detection
        print(f"衝突検出: {'有効' if self.enable_collision_detection else '無効'}")
        return True
    
    def _toggle_collision_visualization(self) -> bool:
        self.enable_collision_visualization = not self.enable_collision_visualization
        print(f"衝突可視化: {'有効' if self.enable_collision_visualization else '無効'}")
        self._update_visualization()
        return True
    
    def _force_mesh_update_request(self) -> bool:
        print("メッシュを強制更新中...")
        self._force_mesh_update()
        return True
    
    def _increase_sphere_radius(self) -> bool:
        self.sphere_radius = min(self.sphere_radius + 0.01, 0.2)
        print(f"球半径: {self.sphere_radius*100:.1f}cm")
        return True
    
    def _decrease_sphere_radius(self) -> bool:
        self.sphere_radius = max(self.sphere_radius - 0.01, 0.01)
        print(f"球半径: {self.sphere_radius*100:.1f}cm")
        return True
    
    def _toggle_audio_synthesis(self) -> bool:
        self.enable_audio_synthesis = not self.enable_audio_synthesis
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
        else:
            self._shutdown_audio_system()
        print(f"音響合成: {'有効' if self.enable_audio_synthesis else '無効'}")
        return True
    
    def _decrease_volume(self) -> bool:
        if self.enable_audio_synthesis and self.audio_synthesizer:
            self.audio_master_volume = max(0.0, self.audio_master_volume - 0.1)
            self.audio_synthesizer.update_master_volume(self.audio_master_volume)
            print(f"音量: {self.audio_master_volume:.1f}")
        return True
    
    def _increase_volume(self) -> bool:
        if self.enable_audio_synthesis and self.audio_synthesizer:
            self.audio_master_volume = min(1.0, self.audio_master_volume + 0.1)
            self.audio_synthesizer.update_master_volume(self.audio_master_volume)
            print(f"音量: {self.audio_master_volume:.1f}")
        return True
    
    def _update_terrain_mesh(self, points_3d: np.ndarray) -> None:
        """地形メッシュを更新 – MeshPipeline 版 (T-MESH-101)"""
        if points_3d is None or len(points_3d) < 100:
            return

        try:
            import time

            start_t = time.perf_counter()

            # MeshPipeline に委譲
            mesh_res = self.mesh_pipeline.generate_mesh(
                points_3d,
                self.current_tracked_hands,
                force_update=getattr(self, 'force_mesh_update_requested', False),
            )

            if mesh_res.mesh is None:
                return  # 生成失敗 / ポイント不足

            simplified_mesh = mesh_res.mesh  # MeshPipeline 内で簡略化済み

            gen_ms = (time.perf_counter() - start_t) * 1000.0
            if self.frame_counter % 50 == 0:
                logger.debug("[MESH-PIPELINE] %d points -> %d tris in %.1fms", len(points_3d), simplified_mesh.num_triangles, gen_ms)

            # 空間インデックスは「メッシュが変わったとき」のみ再構築
            if mesh_res.changed or self.spatial_index is None:
                self.spatial_index = SpatialIndex(simplified_mesh, index_type=IndexType.BVH)
                # 衝突検出コンポーネント初期化 / 更新
                self.collision_searcher = CollisionSearcher(self.spatial_index)
                self.collision_tester = SphereTriangleCollision(simplified_mesh)
            
            # メッシュ保存
            self.current_mesh = simplified_mesh
            
            # メッシュ範囲のログ出力
            self._log_mesh_info(simplified_mesh)
            
            # 可視化: mesh_res.needs_refresh または mesh_res.changed
            if mesh_res.needs_refresh or mesh_res.changed:
                self._update_mesh_visualization(simplified_mesh)
            
            # 強制更新フラグをリセット
            if hasattr(self, 'force_mesh_update_requested'):
                self.force_mesh_update_requested = False
            
            print(f"メッシュ更新完了: {simplified_mesh.num_triangles}三角形")
            
        except Exception as e:
            logger.error(f"メッシュ生成中にエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _log_lod_mesh_generation(self, points_3d: np.ndarray, triangle_mesh: Any, time_ms: float) -> None:
        """LODメッシュ生成のログ"""
        if hasattr(self, 'frame_counter') and self.frame_counter % 50 == 0:
            print(f"[LOD-MESH] {len(points_3d)} points -> {triangle_mesh.num_vertices} vertices, "
                  f"{triangle_mesh.num_triangles} triangles in {time_ms:.1f}ms")
    
    def _log_mesh_info(self, mesh: Any) -> None:
        """メッシュ情報のログ"""
        if mesh.vertices.size > 0:
            mesh_min = np.min(mesh.vertices, axis=0)
            mesh_max = np.max(mesh.vertices, axis=0)
            logger.info(f"[MESH-INFO] Vertex range: X[{mesh_min[0]:.3f}, {mesh_max[0]:.3f}], "
                       f"Y[{mesh_min[1]:.3f}, {mesh_max[1]:.3f}], Z[{mesh_min[2]:.3f}, {mesh_max[2]:.3f}]")
    
    def _detect_collisions(self, tracked_hands: List[TrackedHand]) -> List[Any]:
        """衝突検出を実行"""
        if not self.collision_searcher:
            logger.debug("No collision searcher available")
            return []
        
        events = []
        self.current_collision_points = []
        logger.debug(f"Processing {len(tracked_hands)} hands")
        
        # GPU加速距離計算の準備
        use_gpu_distance = (
            self.gpu_acceleration_enabled and 
            self.gpu_distance_calc is not None and
            len(tracked_hands) > 0
        )
        
        for i, hand in enumerate(tracked_hands):
            if hand.position is None:
                logger.debug(f"Hand {i} has no position")
                continue
            
            # Prepare list of positions to test: current + historical (predictive)
            pos_history = self._hand_position_history.get(hand.id, [])
            positions_to_test: List[np.ndarray] = [np.array(hand.position)] + [np.array(p) for p in pos_history]

            # Remove duplicates while preserving order
            seen = set()
            unique_positions = []
            for p in positions_to_test:
                tup = tuple(np.round(p, 4))
                if tup not in seen:
                    seen.add(tup)
                    unique_positions.append(p)

            hand_pos_np = unique_positions[0]
            
            logger.debug(f"Hand {i} position: ({hand_pos_np[0]:.3f}, {hand_pos_np[1]:.3f}, {hand_pos_np[2]:.3f})")
            
            try:
                for ptest in unique_positions:
                    # Adaptive radius: enlarge slightly based on velocity magnitude
                    vel_mag = float(np.linalg.norm(getattr(hand, "velocity", np.zeros(3)))) if hasattr(hand, "velocity") else 0.0
                    adaptive_radius = self.sphere_radius + min(0.03, vel_mag * 0.05)

                    # Direct sphere query for the specific test point
                    res = self.collision_searcher._search_point(ptest, adaptive_radius)

                    if not res.triangle_indices:
                        continue

                    info = self._perform_collision_test(
                        ptest, adaptive_radius, res, use_gpu_distance, i
                    )

                    if info and info.has_collision:
                        event = self._create_collision_event(hand, ptest, info)
                        if event:
                            events.append(event)
                            self._update_collision_points(info)
                        # Found collision, break predictive loop
                        break

            except Exception as e:
                logger.error(f"Error processing hand {i}: {e}")
        
        logger.debug(f"Total collision events: {len(events)}")
        return events
    
    def _perform_collision_test(self, hand_pos: np.ndarray, radius: float, 
                               search_result: Any, use_gpu: bool, hand_index: int) -> Any:
        """衝突テストを実行"""
        if use_gpu and len(search_result.triangle_indices) > 5:
            # GPU加速距離計算
            info = self._gpu_collision_testing(hand_pos, radius, search_result)
            self.gpu_stats['distance_calculations'] += 1
            logger.debug(f"[GPU-DISTANCE] Hand {hand_index} collision test using GPU acceleration")
        else:
            # CPU衝突検出
            if self.collision_tester is not None:
                info = self.collision_tester.test_sphere_collision(hand_pos, radius, search_result)
                if use_gpu:
                    self.gpu_stats['cpu_fallbacks'] += 1
                    logger.debug(f"[CPU-FALLBACK] Hand {hand_index} using CPU collision")
            else:
                return None
        
        return info
    
    def _create_collision_event(self, hand: TrackedHand, hand_pos: np.ndarray, info: Any) -> Any:
        """衝突イベントを生成"""
        velocity = np.array(hand.velocity) if hasattr(hand, 'velocity') and hand.velocity is not None else np.zeros(3)
        return self.event_queue.create_event(info, hand.id, hand_pos, velocity)
    
    def _update_collision_points(self, info: Any) -> None:
        """衝突点を更新（統一形式）"""
        for cp in info.contact_points:
            # ContactPointオブジェクトをそのまま保持
            self.current_collision_points.append(cp)
    
    def _gpu_collision_testing(self, hand_pos: np.ndarray, radius: float, search_result: Any) -> Any:
        """GPU加速衝突テスト"""
        try:
            import time
            start_time = time.perf_counter()
            
            # Type safety hints for static analysers
            assert self.gpu_distance_calc is not None, "GPU distance calculator not initialized"

            # 三角形データを抽出
            if self.current_mesh is None or not hasattr(self.current_mesh, 'vertices') or not hasattr(self.current_mesh, 'triangles'):
                # GPU処理失敗時はCPU処理にフォールバック
                if self.collision_tester is not None:
                    return self.collision_tester.test_sphere_collision(hand_pos, radius, search_result)
                from src.collision.sphere_tri import CollisionInfo
                import numpy as _np
                return CollisionInfo(
                    has_collision=False,
                    contact_points=[],
                    closest_point=None,
                    total_penetration_depth=0.0,
                    collision_normal=_np.array([0.0, 0.0, 1.0]),
                    collision_time_ms=0.0,
                )
            
            vertices = np.asarray(self.current_mesh.vertices)
            triangles = np.asarray(self.current_mesh.triangles)
            
            # 対象三角形のみ抽出
            target_triangles = triangles[search_result.triangle_indices]
            
            # 手位置を配列に変換
            hand_points = hand_pos.reshape(1, 3)
            
            # GPU距離計算
            distances = self.gpu_distance_calc.point_to_triangle_distance_batch(
                hand_points, target_triangles, vertices
            )
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.gpu_stats['gpu_time_total_ms'] += elapsed
            
            if distances is not None and distances.size > 0:
                # 衝突結果を生成
                return self._create_collision_info_from_distances(
                    distances, hand_pos, radius, vertices, triangles, search_result, elapsed
                )
            
            # 衝突なしの場合
            from src.collision.sphere_tri import CollisionInfo
            import numpy as _np
            return CollisionInfo(
                has_collision=False,
                contact_points=[],
                closest_point=None,
                total_penetration_depth=0.0,
                collision_normal=_np.array([0.0, 0.0, 1.0]),
                collision_time_ms=elapsed,
            )
            
        except Exception as e:
            logger.error(f"GPU collision testing failed: {e}, falling back to CPU")
            self.gpu_stats['cpu_fallbacks'] += 1
            if self.collision_tester is not None:
                return self.collision_tester.test_sphere_collision(hand_pos, radius, search_result)
            from src.collision.sphere_tri import CollisionInfo
            import numpy as _np
            return CollisionInfo(
                has_collision=False,
                contact_points=[],
                closest_point=None,
                total_penetration_depth=0.0,
                collision_normal=_np.array([0.0, 0.0, 1.0]),
                collision_time_ms=0.0,
            )
    
    def _create_collision_info_from_distances(self, distances: np.ndarray, hand_pos: np.ndarray, 
                                            radius: float, vertices: np.ndarray, 
                                            triangles: np.ndarray, search_result: Any,
                                            computation_time_ms: float) -> Any:
        """距離情報から衝突情報を生成"""
        from src.collision.sphere_tri import CollisionInfo, ContactPoint
        from src.types import CollisionType
        
        # 衝突判定（半径内の距離）
        collision_mask = distances[0] <= radius
        collision_triangle_indices = np.array(search_result.triangle_indices)[collision_mask]
        collision_distances = distances[0][collision_mask]
        
        contact_points = []
        if len(collision_triangle_indices) > 0:
            for i, tri_idx in enumerate(collision_triangle_indices):
                # 三角形の重心を接触点として使用（簡易版）
                tri_vertices = vertices[triangles[tri_idx]]
                centroid = np.mean(tri_vertices, axis=0)
                
                # 法線計算（簡易版）
                v1 = tri_vertices[1] - tri_vertices[0]
                v2 = tri_vertices[2] - tri_vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                from numpy import array as _arr
                contact_point = ContactPoint(
                    position=centroid,
                    normal=normal,
                    depth=float(radius - collision_distances[i]),
                    triangle_index=int(tri_idx),
                    barycentric=_arr([1.0 / 3, 1.0 / 3, 1.0 / 3]),
                    collision_type=CollisionType.FACE_COLLISION,
                )
                contact_points.append(contact_point)
        
        if not contact_points:
            import numpy as _np
            return CollisionInfo(
                has_collision=False,
                contact_points=[],
                closest_point=None,
                total_penetration_depth=0.0,
                collision_normal=_np.array([0.0, 0.0, 1.0]),
                collision_time_ms=computation_time_ms,
            )

        # Determine aggregate metrics
        closest_point = max(contact_points, key=lambda cp: cp.depth)
        total_penetration = sum(cp.depth for cp in contact_points)
        import numpy as _np
        avg_normal = _np.mean([cp.normal for cp in contact_points], axis=0)
        avg_normal = avg_normal / (_np.linalg.norm(avg_normal) + 1e-8)

        return CollisionInfo(
            has_collision=True,
            contact_points=contact_points,
            closest_point=closest_point,
            total_penetration_depth=total_penetration,
            collision_normal=avg_normal,
            collision_time_ms=computation_time_ms,
        )
    
    def _update_mesh_visualization(self, mesh: Any) -> None:
        """メッシュ可視化を更新"""
        if not hasattr(self, 'vis') or self.vis is None:
            return
        
        # 既存のメッシュジオメトリを削除
        for geom in self.mesh_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.mesh_geometries.clear()
        
        try:
            # Open3Dメッシュを作成
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
            
            # 法線計算
            o3d_mesh.compute_vertex_normals()
            
            # 半透明のマテリアル設定
            o3d_mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 薄青色
            
            # ワイヤーフレーム表示
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
            wireframe.paint_uniform_color([0.3, 0.3, 0.7])  # 青色
            
            # ジオメトリを追加
            self.vis.add_geometry(o3d_mesh, reset_bounding_box=False)
            self.vis.add_geometry(wireframe, reset_bounding_box=False)
            
            self.mesh_geometries.extend([o3d_mesh, wireframe])
            
            # -- Open3D バッファ更新 (T-MESH-106) --
            try:
                self.vis.update_geometry(o3d_mesh)
                self.vis.update_geometry(wireframe)
            except Exception as _exc:  # pylint: disable=broad-except
                logger.debug("Open3D update_geometry failed: %s", _exc)
            
        except Exception as e:
            logger.error(f"メッシュ可視化エラー: {e}")
    
    def _update_collision_visualization(self) -> None:
        """衝突可視化を更新"""
        if not hasattr(self, 'vis') or self.vis is None:
            return
        
        # 既存の衝突ジオメトリを削除
        for geom in self.collision_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.collision_geometries.clear()
        
        if not self.enable_collision_visualization:
            return
        
        try:
            # 接触点を可視化
            self._visualize_contact_points()
            
            # 衝突球を可視化
            self._visualize_collision_spheres()
        
        except Exception as e:
            logger.error(f"衝突可視化エラー: {e}")
    
    def _visualize_contact_points(self) -> None:
        """接触点を可視化"""
        if self.vis is None:
            return
        for contact in self.current_collision_points:
            # ContactPointオブジェクトから直接アクセス
            position = contact.position
            normal = contact.normal
            
            # 接触点（球）
            contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            contact_sphere.translate(position)
            contact_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
            
            # 法線ベクトル（線分）
            normal_end = position + normal * 0.05
            normal_line = o3d.geometry.LineSet()
            normal_line.points = o3d.utility.Vector3dVector([position, normal_end])
            normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            normal_line.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
            
            self.vis.add_geometry(contact_sphere, reset_bounding_box=False)
            self.vis.add_geometry(normal_line, reset_bounding_box=False)
            
            self.collision_geometries.extend([contact_sphere, normal_line])
    
    def _visualize_collision_spheres(self) -> None:
        """衝突球を可視化"""
        if self.vis is None:
            return
        if self.current_tracked_hands:
            for tracked in self.current_tracked_hands:
                if tracked.position is not None:
                    hand_sphere = o3d.geometry.TriangleMesh.create_sphere(
                        radius=self.sphere_radius
                    )
                    hand_sphere.translate(tracked.position)
                    hand_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 緑色
                    
                    # ワイヤーフレーム表示
                    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(hand_sphere)
                    wireframe.paint_uniform_color([0.0, 0.8, 0.0])
                    
                    self.vis.add_geometry(wireframe, reset_bounding_box=False)
                    self.collision_geometries.append(wireframe)
    
    def _update_visualization(self) -> None:
        """可視化全体を更新"""
        if self.current_mesh and self.enable_collision_visualization:
            self._update_mesh_visualization(self.current_mesh)
        self._update_collision_visualization()

        # Open3D ビューワーへ再描画を通知 (MS3 – Viewer refresh)
        self._refresh_viewer()

    # ------------------------------------------------------------------
    # 3D Viewer helpers
    # ------------------------------------------------------------------

    def _refresh_viewer(self) -> None:
        """Force Open3D visualizer to redraw the scene."""
        if hasattr(self, "vis") and self.vis is not None:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Visualizer refresh failed: %s", exc)
    
    def _force_mesh_update(self) -> None:
        """メッシュ強制更新をリクエスト"""
        self.force_mesh_update_requested = True
    
    def _print_performance_stats(self) -> bool:
        """パフォーマンス統計を印刷"""
        print("\n" + "="*50)
        print("パフォーマンス統計")
        print("="*50)
        
        # 基本統計
        self._print_basic_stats()
        
        # ボクセルダウンサンプリング統計
        self._print_voxel_stats_details()
        
        # メッシュと衝突検出統計
        self._print_mesh_collision_stats()
        
        # 音響統計
        self._print_audio_stats()
        
        print("="*50)
        return True
    
    def _print_basic_stats(self) -> None:
        """基本統計を印刷"""
        print(f"総フレーム数: {self.perf_stats['frame_count']}")
        print(f"現在のフレーム: {self.frame_counter}")
        print(f"パイプライン時間: {self.perf_stats['total_pipeline_time']:.2f}ms")
        print(f"メッシュ生成時間: {self.perf_stats['mesh_generation_time']:.2f}ms")
        print(f"衝突検出時間: {self.perf_stats['collision_detection_time']:.2f}ms")
        print(f"音響生成時間: {self.perf_stats['audio_synthesis_time']:.2f}ms")
        print(f"総衝突イベント数: {self.perf_stats['collision_events_count']}")
        print(f"総音響ノート数: {self.perf_stats['audio_notes_played']}")
    
    def _print_voxel_stats_details(self) -> None:
        """ボクセル統計の詳細を印刷"""
        if self.pointcloud_converter:
            voxel_stats = self.pointcloud_converter.get_performance_stats()
            print(f"\n--- Point Cloud Optimization ---")
            print(f"ボクセルダウンサンプリング: {'有効' if voxel_stats.get('voxel_downsampling_enabled', False) else '無効'}")
            if voxel_stats.get('voxel_downsampling_enabled', False):
                print(f"  - ボクセルサイズ: {voxel_stats.get('current_voxel_size_mm', 0):.1f}mm")
                print(f"  - 最新入力点数: {voxel_stats.get('last_input_points', 0):,}")
                print(f"  - 最新出力点数: {voxel_stats.get('last_output_points', 0):,}")
                print(f"  - ダウンサンプリング率: {voxel_stats.get('last_downsampling_ratio', 0)*100:.1f}%")
                print(f"  - 平均処理時間: {voxel_stats.get('average_time_ms', 0):.2f}ms")
    
    def _print_mesh_collision_stats(self) -> None:
        """メッシュと衝突検出統計を印刷"""
        if self.current_mesh:
            print(f"現在のメッシュ: {self.current_mesh.num_triangles}三角形")
        print(f"球半径: {self.sphere_radius*100:.1f}cm")
    
    def _print_audio_stats(self) -> None:
        """音響統計を印刷"""
        if self.enable_audio_synthesis:
            print(f"音響合成: {'有効' if self.audio_enabled else '無効'}")
            if self.audio_enabled:
                print(f"  - 音階: {self.audio_scale.value}")
                print(f"  - 楽器: {self.audio_instrument.value}")
                print(f"  - 音量: {self.audio_master_volume:.1f}")
                if self.voice_manager:
                    voice_stats = self.voice_manager.get_performance_stats()
                    print(f"  - アクティブボイス: {voice_stats['current_active_voices']}/{self.audio_polyphony}")
                    print(f"  - 総作成ボイス: {voice_stats['total_voices_created']}")
                    print(f"  - ボイススティール: {voice_stats['total_voices_stolen']}")
    
    def _process_frame(self) -> bool:
        """1フレーム処理（統一版）"""
        frame_start_time = time.perf_counter()
        
        # 3D手検出コンポーネントの遅延初期化
        self._lazy_initialize_3d_components()
        
        # フレーム取得
        frame_data = self._get_frame_data()
        if frame_data is None:
            return True
        
        # 深度・カラー画像の抽出
        depth_image, color_image = self._extract_images_from_frame(frame_data)
        if depth_image is None:
            return True
        
        # カラー画像をキャッシュ（手検出用）
        self._last_color_frame = color_image
        
        # フィルタ適用
        depth_image = self._apply_depth_filter(depth_image)
        
        # ---------------- Hand masking (P-HAND-002) ----------------
        from src.detection.hand_mask import HandMasker  # type: ignore
        if not hasattr(self, "_hand_masker"):
            self._hand_masker = HandMasker()

        depth_image_masked, centers_3d, radii_arr = self._hand_masker.apply_mask(
            depth_image,
            self.current_hands_2d,
            self.current_tracked_hands,
        )

        # 最新深度をキャッシュ
        self._latest_depth_image = depth_image_masked

        # 点群生成
        points_3d = self._generate_point_cloud_if_needed(depth_image_masked)

        # 手検出処理はマスク前の深度画像を使用

        hands_2d, hands_3d, tracked_hands = self._perform_hand_detection(depth_image)
        self._save_hand_detection_results(hands_2d, hands_3d, tracked_hands)

        # ----- adjust further references -----
        # Use adaptive exclusion when we regenerate pointcloud on subsequent calls
        self._exclude_centers_cached = (centers_3d, radii_arr)

        # Continue pipeline below so we need to skip duplicated code (return later edits)
        
        # 衝突検出とメッシュ生成のパイプライン
        collision_events = self._process_collision_pipeline(points_3d, tracked_hands)
        
        # 表示処理（ヘッドレスモードでは表示をスキップ）
        if not self.headless_mode:
            # RGB表示処理
            if not self._process_rgb_display(frame_data, collision_events):
                return False
            
            # 点群表示処理
            if self.frame_count % self.update_interval == 0:
                if not self._process_pointcloud_display(frame_data):
                    return False
        
        self.frame_count += 1
        
        # パフォーマンス統計更新
        frame_time = (time.perf_counter() - frame_start_time) * 1000
        self.performance_stats['frame_time'] = frame_time
        self.performance_stats['fps'] = 1000.0 / frame_time if frame_time > 0 else 0.0
        
        return True
    
    def _get_frame_data(self) -> Optional[Any]:
        """フレームデータを取得（カメラまたはモック）"""
        if self.camera is None:
            # ヘッドレスモード用モックデータ
            if self.headless_mode:
                # 遅延でモックカメラを生成
                from typing import cast, Any
                from src.input.mock_camera import MockCamera

                if not hasattr(self, "_mock_camera"):
                    self._mock_camera = MockCamera(LOW_RESOLUTION[0], LOW_RESOLUTION[1])

                # 型チェッカー対策で OrbbecCamera 互換として扱う
                self.camera = cast(Any, self._mock_camera)
                return self._mock_camera.get_frame()
            logger.warning("Camera not available")
            return None
        
        return self.camera.get_frame(timeout_ms=100)
    
    def _extract_images_from_frame(self, frame_data: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """フレームデータから深度・カラー画像を抽出"""
        depth_image = None
        color_image = None
        
        if frame_data.depth_frame is not None:
            depth_image = self._extract_depth_image(frame_data)
        
        if frame_data.color_frame is not None:
            color_image = self._extract_color_image(frame_data)
        
        return depth_image, color_image
    
    def _lazy_initialize_3d_components(self) -> None:
        """3D手検出コンポーネントの遅延初期化"""
        if not self._components_initialized and hasattr(self, 'camera') and self.camera is not None:
            try:
                logger.info("Setting up 3D hand detection components...")
                if self.camera.depth_intrinsics is not None:
                    # 3D投影器の初期化
                    self.projector_3d = Hand3DProjector(
                        self.camera.depth_intrinsics,
                        min_confidence_3d=0.1
                    )
                    # トラッカーの初期化
                    self.tracker = Hand3DTracker()
                    
                    # 点群コンバーターの初期化（まだない場合）
                    if not hasattr(self, 'pointcloud_converter') or self.pointcloud_converter is None:
                        self.pointcloud_converter = PointCloudConverter(
                            self.camera.depth_intrinsics,
                            enable_voxel_downsampling=self.voxel_downsampling_enabled,
                            voxel_size=self.voxel_size
                        )
                    
                    self._components_initialized = True
                    logger.info("3D hand detection components initialized")
                else:
                    logger.warning("Camera depth intrinsics not available")
            except Exception as e:
                logger.error(f"3D component initialization error: {e}")
    
    def _extract_depth_image(self, frame_data: Any) -> Optional[np.ndarray]:
        """フレームデータから深度画像を抽出"""
        try:
            if self.camera is None or self.camera.depth_intrinsics is None:
                # --------------------------------------------------------------
                # Determine frame dimensions in a robust manner
                # --------------------------------------------------------------
                df = frame_data.depth_frame
                width = None
                height = None

                # Newer pyorbbecsdk versions expose width/height attributes
                if hasattr(df, "width") and hasattr(df, "height"):
                    width = int(df.width)
                    height = int(df.height)
                # Older versions expose getter functions
                elif hasattr(df, "get_width") and hasattr(df, "get_height"):
                    try:
                        width = int(df.get_width())
                        height = int(df.get_height())
                    except Exception:  # pragma: no cover – defensive
                        width, height = None, None

                if width is None or height is None:
                    logger.warning("DepthFrame size unavailable – defaulting to 424x240")
                    width, height = 424, 240

                from src.input.pointcloud import _create_default_intrinsics  # local import
                logger.warning("depth_intrinsics is None – using fallback intrinsics (%dx%d)", width, height)
                fallback_intr = _create_default_intrinsics(width, height)

                if self.camera is not None:
                    self.camera.depth_intrinsics = fallback_intr  # type: ignore[attr-defined]

                intr = fallback_intr
            else:
                intr = self.camera.depth_intrinsics

            depth_data = np.frombuffer(frame_data.depth_frame.get_data(), dtype=np.uint16)
            return depth_data.reshape((intr.height, intr.width))
        except Exception as e:
            logger.error(f"Failed to extract depth image: {e}")
            return None
    
    def _extract_color_image(self, frame_data: Any) -> Optional[np.ndarray]:
        """フレームデータからカラー画像を抽出"""
        try:
            if (
                frame_data.color_frame is None
                or self.camera is None
                or not getattr(self.camera, "has_color", False)
            ):
                return None
            
            color_data = np.frombuffer(frame_data.color_frame.get_data(), dtype=np.uint8)
            color_format = frame_data.color_frame.get_format()
            
            # フォーマット変換
            return self._convert_color_format(color_data, color_format)
        except Exception as e:
            logger.error(f"Failed to extract color image: {e}")
            return None
    
    def _apply_depth_filter(self, depth_image: np.ndarray) -> np.ndarray:
        """深度フィルタを適用"""
        if self.depth_filter is not None and self.enable_filter:
            filter_start_time = time.perf_counter()
            filtered = self.depth_filter.apply_filter(depth_image)
            self.performance_stats['filter_time'] = (time.perf_counter() - filter_start_time) * 1000
            return filtered
        else:
            self.performance_stats['filter_time'] = 0.0
            return depth_image
    
    def _generate_point_cloud_if_needed(self, depth_image: np.ndarray, *, force: bool = False) -> Optional[np.ndarray]:
        """必要に応じて点群を生成"""
        need_points_for_mesh = force or (
            self.enable_mesh_generation and 
            (self.frame_count - self.last_mesh_update >= self.mesh_update_interval)
        )
        
        if self.pointcloud_converter and (self.frame_count % self.update_interval == 0 or need_points_for_mesh):
            pointcloud_start = time.perf_counter()
            # Exclude hand vicinity points – adaptive radii (P-HAND-002)
            if hasattr(self, "_exclude_centers_cached"):
                exc_centers, exc_radii = self._exclude_centers_cached
            else:
                exc_centers, exc_radii = None, 0.03

            points_3d, _ = self.pointcloud_converter.numpy_to_pointcloud(
                depth_image,
                exclude_centers=exc_centers,
                exclude_radii=exc_radii,
            )
            self.performance_stats['pointcloud_time'] = (time.perf_counter() - pointcloud_start) * 1000
            
            # キャッシュ
            if points_3d is not None:
                self._last_points_3d = points_3d
            
            if need_points_for_mesh and points_3d is not None:
                logger.info(f"[MESH-PREP] Frame {self.frame_count}: Generated {len(points_3d)} points for mesh update")
            return points_3d
        elif self.headless_mode and not self.camera and need_points_for_mesh:
            # ヘッドレスモード用モック点群
            mock_points = np.random.rand(5000, 3).astype(np.float32)
            mock_points[:, 2] += 0.5  # Z座標調整
            return mock_points
        
        return None
    
    def _process_hand_detection(self, depth_image: np.ndarray) -> Tuple[List, List, List]:
        """手検出処理の実装（親クラスからオーバーライド）"""
        if not self.enable_hand_detection:
            return [], [], []
        
        hands_2d = []
        hands_3d = []
        tracked_hands = []
        
        # カラー画像取得（RGB手検出用）
        if hasattr(self, 'camera') and self.camera and self.camera.has_color:
            try:
                # 最新のカラーフレームを取得
                color_frame = getattr(self, '_last_color_frame', None)
                if color_frame is not None:
                    # 2D手検出（MediaPipe）
                    if self.hands_2d is not None:
                        hands_2d = self.hands_2d.detect_hands(color_frame)
                    
                    # 3D投影
                    if hands_2d and self.projector_3d is not None:
                        for hand_2d in hands_2d:
                            # 関数名修正: Hand3DProjector.project_hand_to_3d に合わせる
                            hand_3d = self.projector_3d.project_hand_to_3d(hand_2d, depth_image)
                            if hand_3d is not None:
                                hands_3d.append(hand_3d)
                    
                    # トラッキング
                    if self.enable_tracking and self.tracker is not None and hands_3d:
                        tracked_hands = self.tracker.update(hands_3d)
            except Exception as e:
                logger.error(f"Hand detection error: {e}")
        
        return hands_2d, hands_3d, tracked_hands
    
    def _perform_hand_detection(self, depth_image: np.ndarray) -> Tuple[List, List, List]:
        """手検出処理を実行（パフォーマンス計測付き）"""
        if self.enable_hand_detection and self.hands_2d is not None:
            hand_start_time = time.perf_counter()
            hands_2d, hands_3d, tracked_hands = self._process_hand_detection(depth_image)
            self.performance_stats['hand_detection_time'] = (time.perf_counter() - hand_start_time) * 1000
            
            if self.frame_count % 10 == 0 and any([hands_2d, hands_3d, tracked_hands]):
                logger.info(f"[HAND-DETECT] Frame {self.frame_count}: "
                           f"2D:{len(hands_2d)}, 3D:{len(hands_3d)}, Tracked:{len(tracked_hands)} "
                           f"({self.performance_stats['hand_detection_time']:.1f}ms)")
            
            return hands_2d, hands_3d, tracked_hands
        else:
            self.performance_stats['hand_detection_time'] = 0.0
            return [], [], []
    
    def _save_hand_detection_results(self, hands_2d: List, hands_3d: List, tracked_hands: List) -> None:
        """手検出結果を保存"""
        self.current_hands_2d = hands_2d
        self.current_hands_3d = hands_3d
        self.current_tracked_hands = tracked_hands
    
        # --- Maintain short history for predictive collision (tap detection) ---
        if not hasattr(self, "_hand_position_history"):
            self._hand_position_history: Dict[str, List[np.ndarray]] = {}

        # Update history per hand
        active_ids = set()
        for th in tracked_hands:
            hid = th.id
            active_ids.add(hid)
            if hid not in self._hand_position_history:
                self._hand_position_history[hid] = []
            if th.position is not None:
                self._hand_position_history[hid].append(np.array(th.position, dtype=float))
            # Keep last 3 positions max
            if len(self._hand_position_history[hid]) > 3:
                self._hand_position_history[hid].pop(0)

        # Remove stale hands from history
        stale_ids = [hid for hid in self._hand_position_history.keys() if hid not in active_ids]
        for hid in stale_ids:
            del self._hand_position_history[hid]
    
    def _process_collision_pipeline(self, points_3d: Optional[np.ndarray], 
                                   tracked_hands: List[TrackedHand]) -> List[Any]:
        """衝突検出とメッシュ生成のパイプライン処理"""
        pipeline_start = time.perf_counter()
        self.frame_counter = self.frame_count
        collision_events = []
        
        # Mesh update via PipelineManager (asynchronous-ready)
        res = self.pipeline_manager.update_if_needed(points_3d, tracked_hands)

        if res.mesh is not None and (
            self._mesh_version != self.pipeline_manager.get_version()
        ):
            # Mesh changed or needs refresh
            self.current_mesh = res.mesh
            self._mesh_version = self.pipeline_manager.get_version()

            # Rebuild index only when res.changed True
            if res.changed or self.spatial_index is None:
                self.spatial_index = SpatialIndex(res.mesh, index_type=IndexType.BVH)
                self.collision_searcher = CollisionSearcher(self.spatial_index)
                self.collision_tester = SphereTriangleCollision(res.mesh)

            self._update_mesh_visualization(res.mesh)
        
        # 衝突検出
        if self.enable_collision_detection and self.current_mesh is not None and tracked_hands:
            collision_events = self._perform_collision_detection(tracked_hands)
        
        # 音響生成
        if self.enable_audio_synthesis and self.audio_enabled and collision_events:
            self._perform_audio_synthesis(collision_events)
        
        self.perf_stats['total_pipeline_time'] = (time.perf_counter() - pipeline_start) * 1000
        
        # ---- Memory hygiene: purge processed collision events (T-MEM-001) ----
        if hasattr(self, 'event_queue') and self.event_queue is not None:
            try:
                self.event_queue.pop_processed(max_length=256)
            except AttributeError:
                # Older versions of CollisionEventQueue may lack this method
                # (e.g. when running mixed code revisions). Fallback: clear if too big.
                if len(self.event_queue.event_queue) > 512:
                    while len(self.event_queue.event_queue) > 256:
                        self.event_queue.event_queue.popleft()

        return collision_events
    
    def _should_update_mesh(self, tracked_hands: List[TrackedHand], 
                           points_3d: Optional[np.ndarray]) -> bool:
        """メッシュ更新が必要かチェック"""
        hands_present = len(tracked_hands) > 0
        frame_diff = self.frame_count - self.last_mesh_update
        
        mesh_condition = (
            self.enable_mesh_generation and (
                self.force_mesh_update_requested or
                (not hands_present and frame_diff >= self.mesh_update_interval) or
                (frame_diff >= self.max_mesh_skip_frames) or
                (self.current_mesh is None)
            ) and
            (
                points_3d is not None
                or self._last_points_3d is not None
            )
        )
        
        # 診断ログ
        if self.frame_count % 10 == 0:
            logger.debug(f"[MESH-DIAG] Frame {self.frame_count}: enable_mesh={self.enable_mesh_generation}, "
                        f"frame_diff={frame_diff}, points={len(points_3d) if points_3d is not None else 'None'}, "
                        f"condition={mesh_condition}")
        
        return mesh_condition
    
    def _perform_collision_detection(self, tracked_hands: List[TrackedHand]) -> List[Any]:
        """衝突検出を実行"""
        logger.info(f"[COLLISION] Frame {self.frame_count}: *** CHECKING COLLISIONS *** "
                   f"with {len(tracked_hands)} hands and mesh available")
        
        collision_start = time.perf_counter()
        collision_events = self._detect_collisions(tracked_hands)
        
        self.perf_stats['collision_detection_time'] = (time.perf_counter() - collision_start) * 1000
        self.perf_stats['collision_events_count'] += len(collision_events)
        
        if len(collision_events) > 0:
            logger.info(f"[COLLISION] Frame {self.frame_count}: *** COLLISION DETECTED! *** "
                       f"{len(collision_events)} events")
        else:
            logger.info(f"[COLLISION] Frame {self.frame_count}: No collisions detected")
        
        return collision_events
    
    def _perform_audio_synthesis(self, collision_events: List[Any]) -> None:
        """音響合成を実行"""
        audio_start = time.perf_counter()
        logger.info(f"[AUDIO] Frame {self.frame_count}: *** GENERATING AUDIO *** "
                   f"for {len(collision_events)} collision events")
        
        audio_notes = self._generate_audio(collision_events)
        self.perf_stats['audio_notes_played'] += audio_notes
        self.perf_stats['audio_synthesis_time'] = (time.perf_counter() - audio_start) * 1000
        
        logger.info(f"[AUDIO] Frame {self.frame_count}: Generated {audio_notes} audio notes in "
                   f"{self.perf_stats['audio_synthesis_time']:.1f}ms")
    
    # 音響システム関連メソッド
    def _initialize_audio_system(self) -> None:
        """音響システムを初期化"""
        try:
            logger.info("音響システムを初期化中...")
            
            # 音響マッパー初期化
            self.audio_mapper = AudioMapper(
                scale=self.audio_scale,
                default_instrument=self.audio_instrument,
                pitch_range=(48, 84),  # C3-C6
                enable_adaptive_mapping=True
            )
            
            # 音響シンセサイザー初期化
            self.audio_synthesizer = create_audio_synthesizer(
                sample_rate=44100,
                buffer_size=256,
                max_polyphony=self.audio_polyphony
            )
            
            # 音響エンジン開始
            if self.audio_synthesizer.start_engine():
                # ボイス管理システム初期化
                self.voice_manager = create_voice_manager(
                    self.audio_synthesizer,
                    max_polyphony=self.audio_polyphony,
                    steal_strategy=StealStrategy.OLDEST
                )
                
                # マスターボリューム設定
                self.audio_synthesizer.update_master_volume(self.audio_master_volume)
                
                self.audio_enabled = True
                logger.info("音響システム初期化完了")
            else:
                logger.error("音響エンジンの開始に失敗しました")
                self.audio_enabled = False
        
        except Exception as e:
            logger.error(f"音響システム初期化エラー: {e}")
            self.audio_enabled = False
    
    def _shutdown_audio_system(self) -> None:
        """音響システムを停止"""
        try:
            logger.info("[AUDIO-SHUTDOWN] 音響システムを停止中...")
            self.audio_enabled = False
            
            # ボイス管理システムの停止
            if self.voice_manager:
                try:
                    self.voice_manager.stop_all_voices(fade_out_time=0.01)
                    time.sleep(0.05)
                    self.voice_manager = None
                except Exception as e:
                    logger.error(f"[AUDIO-SHUTDOWN] VoiceManager停止エラー: {e}")
            
            # シンセサイザーエンジンの停止
            if self.audio_synthesizer:
                try:
                    self.audio_synthesizer.stop_engine()
                    time.sleep(0.05)
                    self.audio_synthesizer = None
                except Exception as e:
                    logger.error(f"[AUDIO-SHUTDOWN] Synthesizer停止エラー: {e}")
            
            # 音響マッパーもクリア
            self.audio_mapper = None
            
            logger.info("[AUDIO-SHUTDOWN] 音響システムを停止しました")
        
        except Exception as e:
            logger.error(f"[AUDIO-SHUTDOWN] 音響システム停止エラー: {e}")
            self.audio_enabled = False
    
    def _restart_audio_system(self) -> None:
        """音響システムを再起動"""
        self._shutdown_audio_system()
        time.sleep(0.1)
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
    
    def _generate_audio(self, collision_events: List[Any]) -> int:
        """衝突イベントから音響を生成"""
        if not self.audio_enabled or not self.audio_mapper or not self.voice_manager:
            return 0
        
        notes_played = 0
        current_time = time.perf_counter()
        
        for event in collision_events:
            try:
                # クールダウンチェック
                if not self._check_audio_cooldown(event.hand_id, current_time):
                    continue
                
                # デバウンス
                if not self._check_contact_debounce(event):
                    continue
                
                # 音響パラメータマッピング
                audio_params = self.audio_mapper.map_collision_event(event)
                
                # 空間位置設定
                spatial_position = self._get_spatial_position(event)
                
                # 音響再生
                voice_id = allocate_and_play(
                    self.voice_manager,
                    audio_params,
                    priority=7,
                    spatial_position=spatial_position
                )
                
                if voice_id:
                    notes_played += 1
                    self.last_audio_trigger_time[event.hand_id] = current_time
                    logger.debug(f"[AUDIO-TRIGGER] Hand {event.hand_id}: Note triggered")
                    # 記録 – debounce と同じキー形式 (hand_id, gx, gy, gz)
                    gx = int(round(event.contact_position[0] * 50))
                    gy = int(round(event.contact_position[1] * 50))
                    gz = int(round(event.contact_position[2] * 50))
                    self._last_contact_trigger_time[(event.hand_id, gx, gy, gz)] = current_time

                    # Keep the debounce map bounded (T-MEM-001)
                    if len(self._last_contact_trigger_time) > 500:
                        # Remove ~20% oldest entries to avoid frequent churn
                        for _ in range(int(0.2 * len(self._last_contact_trigger_time))):
                            try:
                                self._last_contact_trigger_time.pop(next(iter(self._last_contact_trigger_time)))
                            except Exception:  # pragma: no cover
                                break
            
            except Exception as e:
                logger.error(f"音響生成エラー（イベント: {event.event_id}）: {e}")
        
        # ボイスクリーンアップ
        if self.voice_manager and self.frame_count % 10 == 0:
            try:
                self.voice_manager.cleanup_finished_voices()
            except Exception as e:
                logger.error(f"[AUDIO-CLEANUP] Error during cleanup: {e}")
        
        return notes_played
    
    def _check_audio_cooldown(self, hand_id: int, current_time: float) -> bool:
        """音響クールダウンをチェック"""
        last_trigger = self.last_audio_trigger_time.get(hand_id, 0)
        time_since_last = current_time - last_trigger
        
        if time_since_last < self.audio_cooldown_time:
            logger.debug(f"[AUDIO-COOLDOWN] Hand {hand_id}: {time_since_last*1000:.1f}ms since last trigger")
            return False
        
        return True
    
    def _check_contact_debounce(self, event: Any) -> bool:
        """手ID + 接触位置グリッド (≈2 cm) で 250 ms デバウンス"""
        gx = int(round(event.contact_position[0] * 50))  # 1/50 m = 2 cm
        gy = int(round(event.contact_position[1] * 50))
        gz = int(round(event.contact_position[2] * 50))
        key = (event.hand_id, gx, gy, gz)
        last_t = self._last_contact_trigger_time.get(key, 0.0)
        if (time.perf_counter() - last_t) < 0.25:  # 250 ms
            return False
        return True
    
    def _get_spatial_position(self, event: Any) -> np.ndarray:
        """イベントから空間位置を取得"""
        return np.array([
            float(event.contact_position[0]),
            0.0,
            float(event.contact_position[2])
        ], dtype=float)
    
    def _cycle_audio_scale(self) -> bool:
        """音階を循環切り替え"""
        scales = list(ScaleType)
        current_index = scales.index(self.audio_scale)
        next_index = (current_index + 1) % len(scales)
        self.audio_scale = scales[next_index]
        
        if self.audio_mapper:
            self.audio_mapper.set_scale(self.audio_scale)
        
        print(f"音階を切り替え: {self.audio_scale.value}")
        return True
    
    def _cycle_audio_instrument(self) -> bool:
        """楽器を循環切り替え"""
        instruments = list(InstrumentType)
        current_index = instruments.index(self.audio_instrument)
        next_index = (current_index + 1) % len(instruments)
        self.audio_instrument = instruments[next_index]
        
        if self.audio_mapper:
            self.audio_mapper.default_instrument = self.audio_instrument
        
        print(f"楽器を切り替え: {self.audio_instrument.value}")
        return True
    
    def __del__(self):
        """デストラクタ"""
        try:
            if hasattr(self, 'audio_enabled') and self.audio_enabled:
                logger.info("[DESTRUCTOR] 音響システムをクリーンアップ中...")
                self._shutdown_audio_system()
        except Exception as e:
            logger.error(f"[DESTRUCTOR] デストラクタでエラー: {e}")
    
    def cleanup(self):
        """明示的なクリーンアップメソッド"""
        try:
            if self.audio_enabled:
                self._shutdown_audio_system()
        except Exception as e:
            logger.error(f"[CLEANUP] クリーンアップエラー: {e}")
    
    def _process_rgb_display(self, frame_data: Any, collision_events: Optional[List[Any]] = None) -> bool:
        """RGB表示処理（衝突検出版）"""
        try:
            # 深度画像の可視化
            depth_image = self._extract_depth_image(frame_data)
            if depth_image is None:
                return True
            
            depth_colored = self._create_depth_visualization(depth_image)
            
            # 表示画像の準備
            display_images = []
            
            # 深度画像追加
            depth_resized = cv2.resize(depth_colored, self.rgb_window_size)
            cv2.putText(depth_resized, f"Depth (Frame: {self.frame_count})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_images.append(depth_resized)
            
            # RGB画像処理（手検出結果はキャッシュを使用）
            color_bgr = self._process_color_image(
                frame_data, 
                self.current_hands_2d,
                self.current_hands_3d,
                self.current_tracked_hands,
                collision_events
            )

            # ===== Flicker fix =====
            if not hasattr(self, "_last_color_bgr"):
                self._last_color_bgr = None  # 初期化

            if color_bgr is None:
                # カラーフレームが取得できなかった場合は前フレームを再利用
                if self._last_color_bgr is not None:
                    color_bgr = self._last_color_bgr
                else:
                    # まだキャッシュが無ければ黒画像で埋める
                    color_bgr = np.zeros((self.rgb_window_size[1], self.rgb_window_size[0], 3), dtype=np.uint8)
            else:
                # 正常に取得できた場合はキャッシュを更新
                self._last_color_bgr = color_bgr

            # カラー画像を表示用リストに追加
            display_images.append(color_bgr)
            
            # 画像を結合して表示
            combined_image = self._combine_display_images(display_images)
            
            # パフォーマンス情報をオーバーレイ
            if hasattr(super(), '_draw_performance_overlay'):
                super()._draw_performance_overlay(combined_image)
            
            # 衝突検出パフォーマンス情報を追加
            self._draw_collision_performance_info(combined_image, collision_events)
            
            cv2.imshow("Geocussion-SP Input Viewer", combined_image)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            return self.handle_key_event(key)
            
        except Exception as e:
            logger.error(f"RGB display error: {e}")
            return True
    
    def _create_depth_visualization(self, depth_image: np.ndarray) -> np.ndarray:
        """深度画像の可視化を作成"""
        # Manual normalization to avoid cv2 stub type issues
        assert depth_image is not None
        d_min = float(depth_image.min())
        d_ptp = float(depth_image.ptp()) if depth_image.ptp() > 0 else 1.0
        depth_normalized = ((depth_image.astype(np.float32) - d_min) / d_ptp * 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    def _process_color_image(self, frame_data: Any, hands_2d: List, hands_3d: List, 
                           tracked_hands: List, collision_events: Optional[List[Any]]) -> Optional[np.ndarray]:
        """カラー画像を処理"""
        if (
            frame_data.color_frame is None
            or self.camera is None
            or not getattr(self.camera, "has_color", False)
        ):
            return None
        
        color_data = np.frombuffer(frame_data.color_frame.get_data(), dtype=np.uint8)
        color_format = frame_data.color_frame.get_format()
        
        # フォーマット変換
        color_image = self._convert_color_format(color_data, color_format)
        if color_image is None:
            return None
        
        # 既に BGR 配列になっているので、そのままリサイズして使用する
        color_bgr = cv2.resize(color_image, self.rgb_window_size)
        
        # 手検出結果を描画
        if self.enable_hand_detection and hands_2d:
            color_bgr = self._draw_hand_detections(color_bgr, hands_2d, hands_3d, tracked_hands)
        
        # 衝突検出情報を描画
        if collision_events:
            self._draw_collision_info(color_bgr, collision_events)
        
        cv2.putText(color_bgr, f"RGB (FPS: {self.performance_stats['fps']:.1f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return color_bgr
    
    def _convert_color_format(self, color_data: np.ndarray, color_format: Any) -> Optional[np.ndarray]:
        """カラーフォーマットを変換"""
        try:
            # MediaPipe の入力は BGR を想定しているので、常に BGR 配列を返す
            if color_format == OBFormat.RGB:
                # RGB → BGR へ変換
                rgb_image = color_data.reshape((720, 1280, 3))
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif color_format == OBFormat.BGR:
                # そのまま reshape
                return color_data.reshape((720, 1280, 3))
            elif color_format == OBFormat.MJPG:
                # imdecode は BGR で返る
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                return color_image
        except Exception as e:
            logger.error(f"Color format conversion error: {e}")
        
        return None
    
    def _combine_display_images(self, display_images: List[np.ndarray]) -> np.ndarray:
        """表示画像を結合"""
        if len(display_images) > 1:
            return np.hstack(display_images)
        else:
            return display_images[0]
    
    def _draw_collision_info(self, image: np.ndarray, collision_events: List[Any]) -> None:
        """衝突情報をRGB画像に描画"""
        if not collision_events:
            return
        
        # 衝突イベント表示
        cv2.putText(image, f"COLLISION DETECTED! ({len(collision_events)} events)", 
                   (10, image.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 音響再生表示
        if self.enable_audio_synthesis and self.audio_enabled:
            cv2.putText(image, f"PLAYING AUDIO ({self.audio_instrument.value})", 
                       (10, image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _draw_collision_performance_info(self, image: np.ndarray, collision_events: Optional[List[Any]]) -> None:
        """衝突検出パフォーマンス情報を描画"""
        if not hasattr(self, 'perf_stats'):
            return
        
        # 情報リストを作成
        info_lines = [
            f"Mesh: {self.perf_stats.get('mesh_generation_time', 0):.1f}ms",
            f"Collision: {self.perf_stats.get('collision_detection_time', 0):.1f}ms",
            f"Audio: {self.perf_stats.get('audio_synthesis_time', 0):.1f}ms",
            f"Events: {len(collision_events) if collision_events else 0}",
            f"Sphere R: {self.sphere_radius*100:.1f}cm"
        ]
        
        # ボクセル情報追加
        self._add_voxel_info_to_lines(info_lines)
        
        # メッシュ情報追加
        if self.current_mesh:
            info_lines.append(f"Triangles: {self.current_mesh.num_triangles}")
        
        if self.current_collision_points:
            info_lines.append(f"Contacts: {len(self.current_collision_points)}")
        
        # 右側に描画
        x_offset = image.shape[1] - 200
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(image, line, (x_offset, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _add_voxel_info_to_lines(self, info_lines: List[str]) -> None:
        """ボクセル情報を行リストに追加"""
        if self.pointcloud_converter:
            voxel_stats = self.pointcloud_converter.get_performance_stats()
            if voxel_stats.get('voxel_downsampling_enabled', False):
                ratio = voxel_stats.get('last_downsampling_ratio', 0)
                voxel_size = voxel_stats.get('current_voxel_size_mm', 0)
                info_lines.append(f"Voxel: {ratio*100:.0f}% @ {voxel_size:.1f}mm")
            else:
                info_lines.append("Voxel: OFF")
    
    def run(self):
        """ビューワーを実行"""
        if self.headless_mode:
            self.run_headless()
        else:
            super().run()
    
    def run_headless(self):
        """ヘッドレスモード実行"""
        print("\n🖥️  ヘッドレスモード開始 - GUI無効化によるFPS最適化")
        print(f"⏱️  実行時間: {self.headless_duration}秒")
        print("=" * 50)
        
        # ヘッドレスモードフラグを設定
        self.headless_mode = True
        
        # カメラがない場合はモックモードを有効化
        if not self.camera:
            print("🔧 モックデータモードで実行")
        
        print("\n🎯 ヘッドレスモード フレーム処理開始...")
        print("=" * 50)
        
        # ヘッドレス実行ループ
        start_time = time.time()
        frame_count = 0
        fps_samples = []
        last_report_time = start_time
        
        try:
            while time.time() - start_time < self.headless_duration:
                frame_start = time.time()
                
                # 通常のフレーム処理を実行（表示はスキップされる）
                success = self._process_frame()
                
                if success:
                    frame_count += 1
                    frame_time = time.time() - frame_start
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_samples.append(current_fps)
                    
                    # 定期的な統計表示
                    elapsed = time.time() - start_time
                    if elapsed - (last_report_time - start_time) >= 5.0:
                        avg_fps = sum(fps_samples[-100:]) / len(fps_samples[-100:]) if fps_samples else 0
                        print(f"📊 [{elapsed:.1f}s] フレーム: {frame_count}, 平均FPS: {avg_fps:.1f}, 現在FPS: {current_fps:.1f}")
                        last_report_time = time.time()
                        
        except KeyboardInterrupt:
            print("\n⏹️  ユーザーによる中断")
        except Exception as e:
            logger.error(f"ヘッドレスモード実行エラー: {e}")
            import traceback
            traceback.print_exc()
        
        # 結果表示
        self._display_headless_results({
            'execution_time': time.time() - start_time,
            'frame_count': frame_count,
            'fps_samples': fps_samples
        })
    

    
    def _display_headless_results(self, results: Dict[str, Any]) -> None:
        """ヘッドレス実行結果を表示"""
        execution_time = results['execution_time']
        frame_count = results['frame_count']
        fps_samples = results.get('fps_samples', [])
        
        # 統計計算
        avg_fps = frame_count / execution_time if execution_time > 0 else 0
        max_fps = max(fps_samples) if fps_samples else 0
        min_fps = min(fps_samples) if fps_samples else 0
        
        # 結果表示
        print("\n" + "=" * 50)
        print("🏁 ヘッドレスモード 実行結果")
        print("=" * 50)
        print(f"⏱️  実行時間: {execution_time:.1f}秒")
        print(f"🎬 総フレーム数: {frame_count}")
        print(f"🚀 平均FPS: {avg_fps:.1f}")
        print(f"📈 最大FPS: {max_fps:.1f}")
        print(f"📉 最小FPS: {min_fps:.1f}")
        print(f"🎵 衝突イベント総数: {self.perf_stats.get('collision_events_count', 0)}")
        print(f"🔊 音響ノート総数: {self.perf_stats.get('audio_notes_played', 0)}")
        
        # ROI トラッキング統計
        if self.hands_2d is not None and hasattr(self.hands_2d, 'get_roi_tracking_stats'):
            self._display_roi_tracking_stats()
        
        print()
    
    def _display_roi_tracking_stats(self) -> None:
        """ROIトラッキング統計を表示"""
        if self.hands_2d is not None and hasattr(self.hands_2d, 'get_roi_tracking_stats'):
            roi_stats = self.hands_2d.get_roi_tracking_stats()
            print(f"\n📊 ROI トラッキング統計:")
            print(f"   MediaPipe 実行: {roi_stats.mediapipe_executions}/{roi_stats.total_frames}")
            print(f"   スキップ率: {roi_stats.skip_ratio*100:.1f}%")
            print(f"   トラッキング成功率: {roi_stats.success_rate*100:.1f}%")
            
            if roi_stats.mediapipe_executions > 0:
                avg_mediapipe_time = roi_stats.total_mediapipe_time_ms / roi_stats.mediapipe_executions
                print(f"   平均MediaPipe時間: {avg_mediapipe_time:.1f}ms")
            
            if roi_stats.tracking_successes > 0:
                avg_tracking_time = roi_stats.total_tracking_time_ms / roi_stats.tracking_successes
                print(f"   平均トラッキング時間: {avg_tracking_time:.1f}ms")
    

    
    def _draw_hand_detections(self, image: np.ndarray, hands_2d: List, hands_3d: List, 
                             tracked_hands: List) -> np.ndarray:
        """手検出結果を描画（親クラスのメソッドをオーバーライド）"""
        # 親クラスの実装があれば使用
        if hasattr(super(), '_draw_hand_detections'):
            return super()._draw_hand_detections(image, hands_2d, hands_3d, tracked_hands)  # type: ignore[attr-defined]
        
        # --- Custom implementation for our HandDetectionResult dataclass ---
        if hands_2d and self.hands_2d is not None:
            for hand_result in hands_2d:
                try:
                    image = self.hands_2d.draw_landmarks(image, hand_result)
                except Exception as e:
                    logger.debug(f"Landmark draw error: {e}")
        
        # トラッキング情報を描画
        if tracked_hands:
            for i, tracked in enumerate(tracked_hands):
                if tracked.position is not None:
                    info_text = f"Hand {tracked.id}: {tracked.position[2]:.2f}m"
                    cv2.putText(image, info_text, (10, 60 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image
    
    def _draw_performance_overlay(self, image: np.ndarray) -> None:
        """パフォーマンス情報をオーバーレイ（親クラスのメソッドをオーバーライド）"""
        # 親クラスの実装があれば使用
        if hasattr(super(), '_draw_performance_overlay'):
            super()._draw_performance_overlay(image)
            return
        
        # 簡易実装
        fps = self.performance_stats.get('fps', 0.0)
        frame_time = self.performance_stats.get('frame_time', 0.0)
        
        info_text = f"FPS: {fps:.1f} | Frame: {frame_time:.1f}ms"
        cv2.putText(image, info_text, (10, image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # ここで終了。追加の戻り値は不要 (-> None)
        return

    # ---------------------------------------------------------------------
    # Fallback: traditional mesh generation (used when LOD generator fails)
    # ---------------------------------------------------------------------
    def _generate_traditional_mesh(self, points_3d: np.ndarray):
        """従来方式でメッシュを生成（フォールバック用）"""
        if points_3d is None or len(points_3d) < 100:
            return None

        try:
            import time

            # 1) Point cloud projection to height map
            proj_start = time.perf_counter()
            height_map = self.projector.project_points(points_3d)
            proj_time = (time.perf_counter() - proj_start) * 1000.0

            # 2) Delaunay triangulation
            tri_start = time.perf_counter()
            triangle_mesh = self.triangulator.triangulate_heightmap(height_map)
            tri_time = (time.perf_counter() - tri_start) * 1000.0

            if triangle_mesh is None or triangle_mesh.num_triangles == 0:
                logger.warning("[TRADITIONAL-MESH] Triangulation failed (no triangles)")
                return None

            # 3) Mesh simplification
            simp_start = time.perf_counter()
            simplified_mesh = self.simplifier.simplify_mesh(triangle_mesh)
            simp_time = (time.perf_counter() - simp_start) * 1000.0

            if simplified_mesh is None:
                simplified_mesh = triangle_mesh

            # Logging every 50 frames for performance insight
            if hasattr(self, "frame_counter") and self.frame_counter % 50 == 0:
                total = proj_time + tri_time + simp_time
                logger.info(
                    f"[TRADITIONAL-MESH] {len(points_3d)} pts -> {simplified_mesh.num_triangles} tris "
                    f"(Proj {proj_time:.1f}ms, Tri {tri_time:.1f}ms, Simp {simp_time:.1f}ms, Total {total:.1f}ms)"
                )

            return simplified_mesh

        except Exception as e:
            logger.error(f"従来方式メッシュ生成エラー: {e}")
            return None


# =============================================================================
# メイン関数
# =============================================================================

def main():
    """メイン関数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 引数検証
    if not validate_arguments(args):
        return 1
    
    # 解像度設定
    depth_width, depth_height = determine_resolution(args)
    
    # 音階と楽器の変換
    audio_scale, audio_instrument = convert_audio_parameters(args)
    if audio_scale is None or audio_instrument is None:
        return 1
    
    # 情報表示
    display_configuration(args, depth_width, depth_height, audio_scale, audio_instrument)
    
    # テストモード
    if args.test:
        run_preprocessing_optimization_test()
        print("\n" + "=" * 70)
        run_headless_fps_comparison_test()
        return 0
    
    # 設定の適用
    apply_configuration(depth_width, depth_height, args)
    
    # ビューワー実行
    try:
        viewer = create_viewer(args, audio_scale, audio_instrument)
        
        print("\n全フェーズ統合ビューワーを開始します...")
        print("=" * 70)
        
        # ヘッドレスモード処理
        if args.headless:
            print("🖥️  ヘッドレスモード: カメラ初期化をスキップ")
            print("🎯 モックデータによるFPS測定を開始します...")
            viewer.run()
            print("\nビューワーが正常に終了しました")
            return 0
        
        # 通常モード処理
        print("カメラを初期化中...")
        if depth_width and depth_height:
            print(f"   深度解像度: {depth_width}x{depth_height} に設定")
        
        viewer.camera = OrbbecCamera(
            enable_color=True,
            depth_width=depth_width,
            depth_height=depth_height
        )
        
        # DualViewer の初期化は viewer.run() 内部で行われるため、ここでは呼び出さない
        viewer.run()
        
        print("\nビューワーが正常に終了しました")
        return 0
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました")
        return 0
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_argument_parser() -> argparse.ArgumentParser:
    """引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        description="Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=create_help_epilog()
    )
    
    add_basic_arguments(parser)
    add_collision_arguments(parser)
    add_audio_arguments(parser)
    add_detection_arguments(parser)
    add_display_arguments(parser)
    add_resolution_arguments(parser)
    add_window_arguments(parser)
    add_mode_arguments(parser)
    
    return parser


def create_help_epilog() -> str:
    """ヘルプのエピローグを作成"""
    return """
使用例:
    python demo_collision_detection.py                    # デフォルト設定（低解像度424x240）
    python demo_collision_detection.py --force-high-resolution # 高解像度848x480（低FPS注意）
    python demo_collision_detection.py --depth-width 640 --depth-height 360 # カスタム解像度
    python demo_collision_detection.py --no-collision     # 衝突検出無効
    python demo_collision_detection.py --no-mesh          # メッシュ生成無効
    python demo_collision_detection.py --no-audio         # 音響合成無効
    python demo_collision_detection.py --sphere-radius 0.08 # 球半径8cm
    python demo_collision_detection.py --audio-instrument BELL # ベル楽器

操作方法:
    RGB Window:
        Q/ESC: 終了
        F: 深度フィルタ ON/OFF
        H: 手検出 ON/OFF
        T: トラッキング ON/OFF
        
        M: メッシュ生成 ON/OFF
        C: 衝突検出 ON/OFF
        V: 衝突可視化 ON/OFF
        N: メッシュ強制更新
        +/-: 球半径調整
        P: パフォーマンス統計表示
        
        A: 音響合成 ON/OFF
        S: 音階切り替え
        I: 楽器切り替え
        1/2: 音量調整
        R: 音響エンジン再起動
        Q: 全音声停止
    
    3D Viewer:
        マウス: 回転/パン/ズーム
        R: 視点リセット
    """


def add_basic_arguments(parser: argparse.ArgumentParser):
    """基本的な引数を追加"""
    parser.add_argument('--no-filter', action='store_true', help='深度フィルタを無効にする')
    parser.add_argument('--no-hand-detection', action='store_true', help='手検出を無効にする')
    parser.add_argument('--no-tracking', action='store_true', help='トラッキングを無効にする')
    parser.add_argument('--gpu-mediapipe', action='store_true', help='MediaPipeでGPUを使用')


def add_collision_arguments(parser: argparse.ArgumentParser):
    """衝突検出関連の引数を追加"""
    parser.add_argument('--no-mesh', action='store_true', help='メッシュ生成を無効にする')
    parser.add_argument('--no-collision', action='store_true', help='衝突検出を無効にする')
    parser.add_argument('--no-collision-viz', action='store_true', help='衝突可視化を無効にする')
    parser.add_argument('--mesh-interval', type=int, default=DEFAULT_MESH_UPDATE_INTERVAL, 
                       help='メッシュ更新間隔（フレーム数） ※低解像度時は15frame推奨')
    parser.add_argument('--sphere-radius', type=float, default=DEFAULT_SPHERE_RADIUS, 
                       help='衝突検出球の半径（メートル）')
    parser.add_argument('--max-mesh-skip', type=int, default=DEFAULT_MAX_MESH_SKIP_FRAMES, 
                       help='手が写っている場合でもこのフレーム数経過で強制更新')


def add_audio_arguments(parser: argparse.ArgumentParser):
    """音響関連の引数を追加"""
    parser.add_argument('--no-audio', action='store_true', help='音響合成を無効にする')
    parser.add_argument('--audio-scale', type=str, default='PENTATONIC', 
                       choices=['PENTATONIC', 'MAJOR', 'MINOR', 'DORIAN', 'MIXOLYDIAN', 'CHROMATIC', 'BLUES'],
                       help='音階の種類')
    parser.add_argument('--audio-instrument', type=str, default='MARIMBA',
                       choices=['MARIMBA', 'SYNTH_PAD', 'BELL', 'PLUCK', 'BASS', 'LEAD', 'PERCUSSION', 'AMBIENT'],
                       help='楽器の種類')
    parser.add_argument('--audio-polyphony', type=int, default=DEFAULT_AUDIO_POLYPHONY, 
                       help='最大同時発音数')
    parser.add_argument('--audio-volume', type=float, default=DEFAULT_MASTER_VOLUME, 
                       help='マスター音量 (0.0-1.0)')


def add_detection_arguments(parser: argparse.ArgumentParser):
    """検出関連の引数を追加"""
    parser.add_argument('--min-confidence', type=float, default=0.7, 
                       help='最小検出信頼度 (0.0-1.0)')


def add_display_arguments(parser: argparse.ArgumentParser):
    """表示関連の引数を追加"""
    parser.add_argument('--update-interval', type=int, default=3, 
                       help='点群更新間隔（フレーム数）')
    parser.add_argument('--point-size', type=float, default=2.0, 
                       help='点群の点サイズ')
    parser.add_argument('--high-resolution', action='store_true', 
                       help='高解像度表示 (1280x720)')


def add_resolution_arguments(parser: argparse.ArgumentParser):
    """解像度関連の引数を追加"""
    parser.add_argument('--low-resolution', action='store_true', default=True, 
                       help='低解像度モード (424x240) ※FPS向上のため既定ON')
    parser.add_argument('--force-high-resolution', action='store_true', 
                       help='強制的に高解像度 (848x480) を使用 ※低FPS注意')
    parser.add_argument('--depth-width', type=int, help='深度解像度幅を直接指定')
    parser.add_argument('--depth-height', type=int, help='深度解像度高さを直接指定')


def add_window_arguments(parser: argparse.ArgumentParser):
    """ウィンドウ関連の引数を追加"""
    parser.add_argument('--window-width', type=int, default=640, help='RGBウィンドウの幅')
    parser.add_argument('--window-height', type=int, default=480, help='RGBウィンドウの高さ')


def add_mode_arguments(parser: argparse.ArgumentParser):
    """モード関連の引数を追加"""
    parser.add_argument('--test', action='store_true', help='テストモードで実行')
    parser.add_argument('--headless', action='store_true', help='ヘッドレスモード（GUI無効）※FPS大幅向上')
    parser.add_argument('--headless-duration', type=int, default=30, help='ヘッドレスモード実行時間（秒）')
    parser.add_argument('--headless-pure', action='store_true', help='純粋ヘッドレス（手検出無効、最大FPS測定）')


def validate_arguments(args) -> bool:
    """引数を検証"""
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("Error: --min-confidence must be between 0.0 and 1.0")
        return False
    
    if args.sphere_radius <= 0.0 or args.sphere_radius > 0.5:
        print("Error: --sphere-radius must be between 0.0 and 0.5")
        return False
    
    if args.audio_polyphony < 1 or args.audio_polyphony > 64:
        print("Error: --audio-polyphony must be between 1 and 64")
        return False
    
    if args.audio_volume < 0.0 or args.audio_volume > 1.0:
        print("Error: --audio-volume must be between 0.0 and 1.0")
        return False
    
    return True


def determine_resolution(args) -> Tuple[Optional[int], Optional[int]]:
    """解像度を決定"""
    if args.depth_width and args.depth_height:
        return args.depth_width, args.depth_height
    elif args.force_high_resolution:
        return HIGH_RESOLUTION
    elif args.low_resolution:
        return LOW_RESOLUTION
    else:
        return None, None


def convert_audio_parameters(args) -> Tuple[Optional[ScaleType], Optional[InstrumentType]]:
    """音響パラメータを変換"""
    try:
        audio_scale = ScaleType[args.audio_scale]
        audio_instrument = InstrumentType[args.audio_instrument]
        return audio_scale, audio_instrument
    except KeyError as e:
        print(f"Error: Invalid audio parameter: {e}")
        return None, None


def display_configuration(args, depth_width: Optional[int], depth_height: Optional[int], 
                         audio_scale: ScaleType, audio_instrument: InstrumentType):
    """設定を表示"""
    print("=" * 70)
    print("Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）")
    print("=" * 70)
    
    # 解像度情報
    display_resolution_info(depth_width, depth_height)
    
    # 機能状態
    print(f"深度フィルタ: {'無効' if args.no_filter else '有効'}")
    print(f"手検出: {'無効' if args.no_hand_detection else '有効'}")
    print(f"メッシュ生成: {'無効' if args.no_mesh else '有効'}")
    print(f"衝突検出: {'無効' if args.no_collision else '有効'}")
    
    if not args.no_collision:
        print(f"  - 球半径: {args.sphere_radius*100:.1f}cm")
        print(f"  - 可視化: {'無効' if args.no_collision_viz else '有効'}")
    
    print(f"音響合成: {'無効' if args.no_audio else '有効'}")
    if not args.no_audio:
        print(f"  - 音階: {audio_scale.value}")
        print(f"  - 楽器: {audio_instrument.value}")
        print(f"  - ポリフォニー: {args.audio_polyphony}")
        print(f"  - 音量: {args.audio_volume:.1f}")
    
    # ヘッドレスモード情報
    if args.headless:
        print(f"🖥️  ヘッドレスモード: 有効（GUI無効化でFPS向上）")
        print(f"⏱️  実行時間: {args.headless_duration}秒")
        print(f"🚀 予想FPS向上: +5-15 FPS (GUI負荷削除)")
    else:
        print(f"🖥️  表示モード: GUI有効")
    
    print("=" * 70)


def display_resolution_info(width: Optional[int], height: Optional[int]):
    """解像度情報を表示"""
    if width and height:
        resolution_mode = "低解像度" if width <= 424 else "高解像度"
        points_estimate = width * height
        print(f"🚀 解像度最適化: {resolution_mode} ({width}x{height})")
        print(f"   予想点群数: {points_estimate:,} points")
        fps_estimate = "25-30 FPS" if width <= 424 else "5-15 FPS"
        print(f"   予想FPS: {fps_estimate}")
        
        # 高解像度警告
        if points_estimate > ESTIMATED_HIGH_RES_POINTS:
            print(f"⚠️  Warning: High resolution ({width}x{height}) may cause low FPS")
            print(f"   Estimated points: {points_estimate:,}")
            print(f"   Consider using --low-resolution for better performance")
        else:
            print(f"✅ Optimized resolution: {width}x{height} (~{points_estimate:,} points)")
    else:
        print("📏 Using camera default resolution")


def apply_configuration(depth_width: Optional[int], depth_height: Optional[int], args):
    """設定を適用"""
    config = get_config()
    config.input.enable_low_resolution_mode = (depth_width == LOW_RESOLUTION[0] and 
                                               depth_height == LOW_RESOLUTION[1])
    config.input.depth_width = depth_width
    config.input.depth_height = depth_height
    
    # 解像度に基づく最適化
    if config.input.enable_low_resolution_mode:
        apply_low_resolution_optimizations(args)
    else:
        apply_high_resolution_optimizations(depth_width, depth_height, args, config)


def apply_low_resolution_optimizations(args):
    """低解像度時の最適化を適用"""
    if args.mesh_interval == DEFAULT_MESH_UPDATE_INTERVAL:
        args.mesh_interval = 20  # さらに間隔を空ける
    print(f"🔧 低解像度最適化: メッシュ更新間隔={args.mesh_interval}フレーム")


def apply_high_resolution_optimizations(width: Optional[int], height: Optional[int], args, config):
    """高解像度時の最適化を適用"""
    if width and height and (width >= HIGH_RESOLUTION[0] or height >= HIGH_RESOLUTION[1]):
        print(f"🚨 高解像度モード検出: {width}x{height}")
        print(f"⚡ 緊急FPS最適化を適用中...")
        
        # メッシュ更新間隔を大幅延長
        if args.mesh_interval <= 20:
            args.mesh_interval = 40
            print(f"🔧 緊急最適化: メッシュ更新間隔={args.mesh_interval}フレーム (40f間隔)")
        
        # 最大スキップフレームも延長
        if args.max_mesh_skip <= DEFAULT_MAX_MESH_SKIP_FRAMES:
            args.max_mesh_skip = 120
            print(f"🔧 緊急最適化: 最大メッシュスキップ={args.max_mesh_skip}フレーム")
        
        # 解像度ダウンサンプリングを有効化
        config.input.enable_resolution_downsampling = True
        config.input.resolution_target_width = LOW_RESOLUTION[0]
        config.input.resolution_target_height = LOW_RESOLUTION[1]
        print(f"🔧 緊急最適化: 解像度ダウンサンプリング有効 ({width}x{height} → {LOW_RESOLUTION[0]}x{LOW_RESOLUTION[1]})")
        
        print(f"⚡ 高解像度での予想FPS: 8-15 FPS (最適化適用済み)")
    elif width and height:
        print(f"🔧 中解像度最適化: メッシュ更新間隔={args.mesh_interval}フレーム")


def create_viewer(args, audio_scale: ScaleType, audio_instrument: InstrumentType) -> FullPipelineViewer:
    """ビューワーを作成"""
    return FullPipelineViewer(
        enable_filter=not args.no_filter,
        enable_hand_detection=not args.no_hand_detection,
        enable_tracking=not args.no_tracking,
        enable_mesh_generation=not args.no_mesh,
        enable_collision_detection=not args.no_collision,
        enable_collision_visualization=not args.no_collision_viz,
        enable_audio_synthesis=not args.no_audio,
        update_interval=args.update_interval,
        point_size=args.point_size,
        rgb_window_size=(args.window_width, args.window_height),
        min_detection_confidence=args.min_confidence,
        use_gpu_mediapipe=args.gpu_mediapipe,
        mesh_update_interval=args.mesh_interval,
        sphere_radius=args.sphere_radius,
        audio_scale=audio_scale,
        audio_instrument=audio_instrument,
        audio_polyphony=args.audio_polyphony,
        audio_master_volume=args.audio_volume,
        max_mesh_skip_frames=args.max_mesh_skip,
        headless_mode=args.headless,
        headless_duration=args.headless_duration,
        pure_headless_mode=args.headless_pure
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)