#!/usr/bin/env python3
"""
60fps手追跡・統一イベントシステム・統合コントローラーデモ
MediaPipe + 60fps Kalman Tracker + 統一イベント + 音響・衝突統合

使用方法:
    python3 demo_60fps_tracking.py                    # デフォルト設定
    python3 demo_60fps_tracking.py --no-audio         # 音響無効
    python3 demo_60fps_tracking.py --headless         # ヘッドレスモード
    python3 demo_60fps_tracking.py --high-resolution  # 高解像度モード
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 動的インポートとモック定義
# =============================================================================

# OrbbecSDKの動的インポート
HAS_ORBBEC_SDK = False
try:
    from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBError, OBFormat
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
    logger.info("Open3D is available")
except ImportError:
    logger.warning("Open3D is not available. 3D visualization will be disabled.")

# 音響ライブラリの動的インポート
HAS_AUDIO = False
try:
    import pygame
    HAS_AUDIO = True
    logger.info("Pygame audio engine is available")
except ImportError:
    logger.warning("Pygame audio engine is not available. Audio synthesis will be disabled.")

# GPU加速コンポーネントの動的インポート
HAS_GPU_ACCELERATION = False
try:
    from src.collision.distance_gpu import GPUDistanceCalculator, create_gpu_distance_calculator
    from src.mesh.delaunay_gpu import GPUDelaunayTriangulator, create_gpu_triangulator
    HAS_GPU_ACCELERATION = True
    logger.info("GPU acceleration modules loaded (CuPy available)")
except ImportError:
    logger.warning("GPU acceleration unavailable (CuPy not installed)")

# 共通定数のインポート
from src.constants import (
    DEMO_SPHERE_RADIUS_DEFAULT as DEFAULT_SPHERE_RADIUS,
    DEMO_MESH_UPDATE_INTERVAL as DEFAULT_MESH_UPDATE_INTERVAL,
    DEMO_MAX_MESH_SKIP_FRAMES as DEFAULT_MAX_MESH_SKIP_FRAMES,
    DEMO_AUDIO_COOLDOWN_TIME as DEFAULT_AUDIO_COOLDOWN_TIME,
    DEMO_VOXEL_SIZE as DEFAULT_VOXEL_SIZE,
    DEMO_AUDIO_POLYPHONY as DEFAULT_AUDIO_POLYPHONY,
    DEMO_MASTER_VOLUME as DEFAULT_MASTER_VOLUME,
    LOW_RESOLUTION,
    HIGH_RESOLUTION,
    ESTIMATED_HIGH_RES_POINTS,
)

# 必要なモジュールのインポート
try:
    from src.input.stream import OrbbecCamera
    from src.input.pointcloud import PointCloudConverter
    from src.input.depth_filter import DepthFilter, FilterType
    from src.input.mock_camera import MockCamera
    
    # 60fps追跡システム
    from src.detection.tracker_60fps import (
        HighFrequencyHand3DTracker,
        HighFrequencyKalmanConfig,
        create_high_frequency_tracker
    )
    from src.detection.unified_events import (
        UnifiedHandEventStream,
        UnifiedHandEvent,
        UnifiedHandEventType,
        create_unified_event_stream
    )
    from src.detection.integrated_controller import (
        IntegratedHandTrackingController,
        IntegratedControllerConfig,
        create_integrated_controller_from_instances as create_integrated_controller
    )
    
    # 従来システム
    from src.detection.hands2d import MediaPipeHandsWrapper
    from src.detection.hands3d import Hand3DProjector
    from src.detection.tracker import Hand3DTracker
    
    # メッシュ・衝突・音響システム
    from src.mesh.projection import PointCloudProjector, ProjectionMethod
    from src.mesh.delaunay import DelaunayTriangulator
    from src.mesh.simplify import MeshSimplifier
    from src.mesh.index import SpatialIndex, IndexType
    from src.mesh.manager import PipelineManager
    from src.mesh.pipeline import create_mesh_pipeline
    from src.mesh.update_scheduler import MeshUpdateScheduler
    from src.mesh.terrain_change import TerrainChangeDetector
    
    from src.collision.search import CollisionSearcher
    from src.collision.sphere_tri import SphereTriangleCollision
    from src.collision.events import CollisionEventQueue
    
    from src.sound.mapping import AudioMapper, ScaleType, InstrumentType
    from src.sound.simple_synth import create_simple_audio_synthesizer
    from src.sound.voice_mgr import VoiceManager, create_voice_manager
    
    from src.debug.dual_viewer import DualViewer
    from src.config import get_config
    from src.data_types import CameraIntrinsics
    
    logger.info("All required modules imported successfully")
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all dependencies are installed and the project is properly set up.")
    sys.exit(1)


# =============================================================================
# 60fps統合デモクラス
# =============================================================================

class Demo60FPSTracker:
    """60fps手追跡統合デモクラス"""
    
    def __init__(self, args):
        """初期化"""
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # 基本属性
        self.frame_count = 0
        self.last_mesh_update = -999
        self.current_mesh = None
        self.current_tracked_hands = []
        self.current_hands_2d = []
        self.current_hands_3d = []
        self.current_collision_points = []
        
        # コンポーネント初期化
        self.camera = None
        self.depth_filter = None
        self.pointcloud_converter = None
        self.integrated_controller = None
        self.dual_viewer = None
        
        # メッシュ・衝突システム
        self.mesh_pipeline = None
        self.pipeline_manager = None
        self.mesh_scheduler = None
        self.terrain_detector = None
        self.gpu_distance_calc = None
        self.gpu_triangulator = None
        self.gpu_acceleration_enabled = False
        
        # 衝突検出システム
        self.spatial_index = None
        self.collision_searcher = None
        self.collision_tester = None
        self.event_queue = None
        
        # 音響システム
        self.audio_enabled = False
        self.audio_synthesizer = None
        self.audio_mapper = None
        self.last_audio_trigger_time = {}
        
        # メッシュ生成フラグ
        self.enable_mesh_generation = not args.no_mesh
        self.force_mesh_update_requested = False
        
        # パフォーマンス統計
        self.performance_stats = {
            'fps': 0.0,
            'frame_time': 0.0,
            'filter_time': 0.0,
            'pointcloud_time': 0.0,
            'hand_detection_time': 0.0,
            'mesh_generation_time': 0.0,
            'collision_detection_time': 0.0,
            'audio_synthesis_time': 0.0,
            'collision_events_count': 0,
            'audio_notes_played': 0,
            'total_pipeline_time': 0.0
        }
        
        # 音響クールダウン管理
        self.audio_cooldown_time = DEFAULT_AUDIO_COOLDOWN_TIME
        
        # 初期化
        if not self.initialize_components():
            raise RuntimeError("Failed to initialize components")

    def initialize_components(self) -> bool:
        """全コンポーネントを初期化"""
        self.logger.info("Initializing 60fps tracking demo components...")
        
        try:
            # 1. カメラ初期化
            if not self._initialize_camera():
                return False
            
            # 2. 深度フィルタ初期化
            if not self._initialize_depth_filter():
                return False
            
            # 3. 点群コンバーター初期化
            if not self._initialize_pointcloud_converter():
                return False
            
            # 4. メッシュ・衝突システム初期化
            if not self._initialize_mesh_collision_system():
                return False
            
            # 5. 音響システム初期化
            if not self._initialize_audio_system():
                return False
            
            # 6. GPU加速初期化
            if not self._initialize_gpu_acceleration():
                return False
            
            # 7. 統合コントローラー初期化
            if not self._initialize_integrated_controller():
                return False
            
            # 8. 表示システム初期化
            if not self.args.headless:
                if not self._initialize_dual_viewer():
                    return False
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False

    def _initialize_camera(self) -> bool:
        """カメラ初期化"""
        try:
            if self.args.headless:
                # ヘッドレスモード用モックカメラ
                self.camera = MockCamera(
                    self.args.depth_width or LOW_RESOLUTION[0],
                    self.args.depth_height or LOW_RESOLUTION[1]
                )
                self.logger.info("Mock camera initialized for headless mode")
            else:
                # 実際のOrbbecカメラ
                self.camera = OrbbecCamera(
                    enable_color=True,
                    depth_width=self.args.depth_width or LOW_RESOLUTION[0],
                    depth_height=self.args.depth_height or LOW_RESOLUTION[1]
                )
                
                # カメラを初期化
                if not self.camera.initialize():
                    self.logger.error("Failed to initialize Orbbec camera")
                    return False
                
                self.logger.info(f"Orbbec camera initialized: {self.args.depth_width or LOW_RESOLUTION[0]}x{self.args.depth_height or LOW_RESOLUTION[1]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def _initialize_depth_filter(self) -> bool:
        """深度フィルタ初期化"""
        try:
            if self.args.no_filter:
                self.depth_filter = None
                self.logger.info("Depth filter disabled")
            else:
                self.depth_filter = DepthFilter(
                    filter_types=[FilterType.COMBINED],
                    bilateral_d=9,
                    bilateral_sigma_color=75,
                    bilateral_sigma_space=75,
                    temporal_alpha=0.3,
                    use_cuda=True
                )
                self.logger.info("Depth filter initialized with CUDA acceleration")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Depth filter initialization failed: {e}")
            return False

    def _initialize_pointcloud_converter(self) -> bool:
        """点群コンバーター初期化"""
        try:
            # カメラの内部パラメータを取得、なければデフォルトを作成
            if self.camera and self.camera.depth_intrinsics:
                intrinsics = self.camera.depth_intrinsics
                self.logger.info("Using camera intrinsics")
            else:
                # デフォルト内部パラメータを作成
                width = self.args.depth_width or LOW_RESOLUTION[0]
                height = self.args.depth_height or LOW_RESOLUTION[1]
                
                intrinsics = CameraIntrinsics(
                    fx=364.0,  # 424x240での典型的な値
                    fy=364.0,
                    cx=width / 2.0,
                    cy=height / 2.0,
                    width=width,
                    height=height
                )
                self.logger.warning(f"Using default camera intrinsics for {width}x{height}")
            
            self.pointcloud_converter = PointCloudConverter(
                intrinsics,
                enable_voxel_downsampling=True,
                voxel_size=DEFAULT_VOXEL_SIZE
            )
            self.logger.info("Point cloud converter initialized")
            return True
                
        except Exception as e:
            self.logger.error(f"Point cloud converter initialization failed: {e}")
            return False

    def _initialize_mesh_collision_system(self) -> bool:
        """メッシュ・衝突システム初期化"""
        try:
            if self.args.no_mesh:
                self.logger.info("Mesh generation disabled")
                return True
            
            # メッシュパイプライン
            self.mesh_pipeline = create_mesh_pipeline(enable_incremental=False)
            self.pipeline_manager = PipelineManager(self.mesh_pipeline, min_interval_sec=1.0)
            
            # メッシュ更新スケジューラー
            self.mesh_scheduler = MeshUpdateScheduler(
                base_interval_sec=1.0,
                grace_period_sec=2.0
            )
            
            # 地形変化検出器
            self.terrain_detector = TerrainChangeDetector()
            
            # 衝突検出コンポーネント
            if not self.args.no_collision:
                self.collision_searcher = None  # メッシュ生成後に初期化
                self.collision_tester = None
                self.spatial_index = None
                self.event_queue = None # イベントキューを初期化
                self.logger.info("Collision detection components will be initialized after mesh generation")
            
            self.logger.info("Mesh and collision system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh/collision system initialization failed: {e}")
            return False

    def _initialize_audio_system(self) -> bool:
        """音響システム初期化"""
        try:
            if self.args.no_audio or not HAS_AUDIO:
                self.logger.info("Audio synthesis disabled")
                return True
            
            # 音響マッパー
            self.audio_mapper = AudioMapper(
                scale=self.args.audio_scale,
                default_instrument=self.args.audio_instrument,
                pitch_range=(48, 84),
                enable_adaptive_mapping=True
            )
            
            # 音響シンセサイザー（pygame版）
            self.audio_synthesizer = create_simple_audio_synthesizer(
                sample_rate=44100,
                buffer_size=512,
                max_polyphony=self.args.audio_polyphony
            )
            
            # 音響エンジン開始
            if self.audio_synthesizer.start_engine():
                self.audio_synthesizer.update_master_volume(self.args.audio_volume)
                self.audio_enabled = True
                self.logger.info(f"Audio system initialized (pygame) - volume: {self.args.audio_volume}")
            else:
                self.logger.error("Failed to start audio engine")
                self.audio_enabled = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio system initialization failed: {e}")
            return False

    def _initialize_gpu_acceleration(self) -> bool:
        """GPU加速初期化"""
        try:
            if not HAS_GPU_ACCELERATION:
                self.logger.info("GPU acceleration not available")
                return True
            
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
            
            # GPU可用性チェック
            gpu_calc_available = (hasattr(self.gpu_distance_calc, 'gpu_available') and 
                                self.gpu_distance_calc.gpu_available)
            gpu_tri_available = (hasattr(self.gpu_triangulator, 'use_gpu') and 
                               self.gpu_triangulator.use_gpu)
            
            self.gpu_acceleration_enabled = gpu_calc_available or gpu_tri_available
            
            if self.gpu_acceleration_enabled:
                self.logger.info("GPU acceleration initialized successfully")
            else:
                self.logger.warning("GPU acceleration components initialized but GPU not available")
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPU acceleration initialization failed: {e}")
            return False

    def _initialize_integrated_controller(self) -> bool:
        """統合コントローラー初期化"""
        try:
            # カメラの内部パラメータを取得、なければデフォルトを作成
            if self.camera and self.camera.depth_intrinsics:
                intrinsics = self.camera.depth_intrinsics
                self.logger.info("Using camera intrinsics for integrated controller")
            else:
                # デフォルト内部パラメータを作成
                width = self.args.depth_width or LOW_RESOLUTION[0]
                height = self.args.depth_height or LOW_RESOLUTION[1]
                
                intrinsics = CameraIntrinsics(
                    fx=364.0,  # 424x240での典型的な値
                    fy=364.0,
                    cx=width / 2.0,
                    cy=height / 2.0,
                    width=width,
                    height=height
                )
                self.logger.warning(f"Using default camera intrinsics for integrated controller: {width}x{height}")
            
            # 統合コントローラー設定
            config = IntegratedControllerConfig(
                # 60fps追跡設定
                target_fps=60,
                mediapipe_detection_fps=15,
                collision_check_fps=60,
                audio_synthesis_fps=60,
                
                # MediaPipe設定
                mediapipe_confidence=0.7,
                mediapipe_tracking_confidence=0.5,
                use_gpu_mediapipe=self.args.gpu_mediapipe,
                
                # 統合システム設定
                collision_sphere_radius=self.args.sphere_radius,
                collision_enabled=not self.args.no_collision,
                audio_enabled=not self.args.no_audio,
                audio_cooldown_ms=50.0,
                
                # パフォーマンス設定
                max_concurrent_hands=4,
                enable_prediction=True,
                enable_interpolation=True,
                enable_performance_monitoring=True,
                stats_update_interval=60
            )
            
            # MediaPipe 2D検出器を作成
            from src.detection.hands2d import MediaPipeHandsWrapper
            hands_2d = MediaPipeHandsWrapper(
                use_gpu=config.use_gpu_mediapipe,
                max_num_hands=config.max_num_hands,
                min_detection_confidence=config.mediapipe_confidence,
                min_tracking_confidence=config.mediapipe_tracking_confidence
            )
            
            # 3D投影器を作成
            from src.detection.hands3d import Hand3DProjector
            hands_3d = Hand3DProjector(
                camera_intrinsics=intrinsics,
                depth_scale=1000.0,
                min_confidence_3d=0.3
            )
            
            # 統合コントローラー作成
            self.integrated_controller = IntegratedHandTrackingController(
                config=config,
                camera_intrinsics=intrinsics,
                hands_2d=hands_2d,
                hands_3d=hands_3d,
                collision_searcher=self.collision_searcher,
                audio_mapper=self.audio_mapper,
                audio_synthesizer=self.audio_synthesizer
            )
            
            # イベントリスナー登録
            self.integrated_controller.add_event_listener(self._handle_collision_event)
            self.integrated_controller.add_event_listener(self._handle_audio_event)
            
            self.logger.info("Integrated controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Integrated controller initialization failed: {e}")
            return False

    def _initialize_dual_viewer(self) -> bool:
        """表示システム初期化"""
        try:
            if self.args.headless:
                return True
            
            # DualViewer設定（cameraは引数で渡さない）
            self.dual_viewer = DualViewer(
                enable_filter=not self.args.no_filter,
                enable_hand_detection=not self.args.no_hand_detection,
                enable_tracking=not self.args.no_tracking,
                update_interval=3,
                point_size=2.0,
                rgb_window_size=(self.args.window_width, self.args.window_height),
                min_detection_confidence=0.7,
                use_gpu_mediapipe=self.args.gpu_mediapipe
            )
            
            # コンポーネントを外部から注入
            self.dual_viewer.camera = self.camera
            self.dual_viewer.pointcloud_converter = self.pointcloud_converter
            self.dual_viewer.depth_filter = self.depth_filter
            
            # DualViewer初期化
            if not self.dual_viewer.initialize():
                self.logger.error("DualViewer initialization failed")
                return False
            
            self.logger.info("Dual viewer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Dual viewer initialization failed: {e}")
            return False

    def _handle_collision_event(self, event: UnifiedHandEvent):
        """衝突イベントハンドラー"""
        if event.event_type == UnifiedHandEventType.COLLISION_DETECTED:
            self.performance_stats['collision_events_count'] += 1
            self.logger.debug(f"Collision detected: hand_id={event.hand_id}, position={event.hand_position}")

    def _handle_audio_event(self, event: UnifiedHandEvent):
        """音響イベントハンドラー"""
        if event.event_type == UnifiedHandEventType.AUDIO_TRIGGERED:
            self.performance_stats['audio_notes_played'] += 1
            self.logger.debug(f"Audio triggered: hand_id={event.hand_id}")

    def run(self):
        """メインループ実行"""
        self.logger.info("Starting 60fps tracking demo...")
        
        if self.args.headless:
            self._run_headless()
        else:
            self._run_with_gui()

    def _run_headless(self):
        """ヘッドレスモード実行"""
        self.logger.info(f"Running headless mode for {self.args.headless_duration} seconds")
        
        start_time = time.time()
        frame_count = 0
        fps_samples = []
        
        try:
            while time.time() - start_time < self.args.headless_duration:
                frame_start = time.time()
                
                # フレーム処理
                success = self._process_frame()
                
                if success:
                    frame_count += 1
                    frame_time = time.time() - frame_start
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_samples.append(current_fps)
                    
                    # 統計更新
                    self.performance_stats['fps'] = current_fps
                    self.performance_stats['frame_time'] = frame_time * 1000
                    self.performance_stats['total_frames'] = frame_count
                    
                    # 定期的な統計表示
                    if frame_count % 300 == 0:
                        avg_fps = sum(fps_samples[-100:]) / len(fps_samples[-100:]) if fps_samples else 0
                        elapsed = time.time() - start_time
                        self.logger.info(f"[{elapsed:.1f}s] Frame: {frame_count}, Avg FPS: {avg_fps:.1f}")
                        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in headless mode: {e}")
        
        # 結果表示
        self._display_final_stats(time.time() - start_time, frame_count, fps_samples)

    def _run_with_gui(self):
        """GUI付きモード実行"""
        self.logger.info("Running with GUI mode")
        
        if not self.dual_viewer:
            self.logger.error("Dual viewer not initialized")
            return
        
        # DualViewerのrunメソッドを使用
        self.dual_viewer.run()

    def _process_frame(self) -> bool:
        """1フレーム処理"""
        try:
            frame_start_time = time.perf_counter()
            
            # フレームデータ取得
            frame_data = self._get_frame_data()
            if frame_data is None:
                return True
            
            # 深度・カラー画像の抽出
            depth_image, color_image = self._extract_images_from_frame(frame_data)
            if depth_image is None:
                return True
            
            # 深度フィルタ適用
            if self.depth_filter:
                depth_image = self.depth_filter.apply_filter(depth_image)
            
            # 統合コントローラーでの処理
            if self.integrated_controller:
                # 60fps手追跡実行
                results = self.integrated_controller.process_frame(
                    depth_image=depth_image,
                    color_image=color_image,
                    timestamp=time.perf_counter()
                )
                
                # 結果の保存
                if results:
                    self.current_tracked_hands = results.get('tracked_hands', [])
                    self.performance_stats['hand_detection_time'] = results.get('hand_detection_time_ms', 0.0)
            
            # 点群生成
            points_3d = self._generate_point_cloud(depth_image)
            
            # メッシュ・衝突パイプライン処理
            if points_3d is not None:
                self._process_mesh_collision_pipeline(points_3d)
            
            # フレームカウンター更新
            self.frame_count += 1
            
            # パフォーマンス統計更新
            frame_time = (time.perf_counter() - frame_start_time) * 1000
            self.performance_stats['frame_time'] = frame_time
            self.performance_stats['fps'] = 1000.0 / frame_time if frame_time > 0 else 0.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return False

    def _get_frame_data(self) -> Optional[Any]:
        """フレームデータ取得"""
        if self.camera:
            return self.camera.get_frame(timeout_ms=100)
        return None

    def _extract_images_from_frame(self, frame_data: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """フレームデータから画像を抽出"""
        depth_image = None
        color_image = None
        
        try:
            if frame_data.depth_frame is not None:
                depth_data = np.frombuffer(frame_data.depth_frame.get_data(), dtype=np.uint16)
                if self.camera and self.camera.depth_intrinsics:
                    depth_image = depth_data.reshape((self.camera.depth_intrinsics.height, self.camera.depth_intrinsics.width))
            
            if frame_data.color_frame is not None:
                color_data = np.frombuffer(frame_data.color_frame.get_data(), dtype=np.uint8)
                color_format = frame_data.color_frame.get_format()
                color_image = self._convert_color_format(color_data, color_format)
        
        except Exception as e:
            self.logger.error(f"Image extraction error: {e}")
        
        return depth_image, color_image

    def _convert_color_format(self, color_data: np.ndarray, color_format: Any) -> Optional[np.ndarray]:
        """カラーフォーマット変換"""
        try:
            if color_format == OBFormat.RGB:
                rgb_image = color_data.reshape((720, 1280, 3))
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif color_format == OBFormat.BGR:
                return color_data.reshape((720, 1280, 3))
            elif color_format == OBFormat.MJPG:
                return cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.error(f"Color format conversion error: {e}")
        return None

    def _generate_point_cloud(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """点群生成"""
        if self.pointcloud_converter and depth_image is not None:
            try:
                points_3d, _ = self.pointcloud_converter.numpy_to_pointcloud(depth_image)
                return points_3d
            except Exception as e:
                self.logger.error(f"Point cloud generation error: {e}")
        return None

    def _process_mesh_collision_pipeline(self, points_3d: np.ndarray):
        """メッシュ・衝突パイプライン処理"""
        if self.args.no_mesh or points_3d is None:
            return
        
        try:
            pipeline_start = time.perf_counter()
            
            # 手の存在チェック
            hands_present = len(self.current_tracked_hands) > 0
            
            # メッシュ更新判定（demo_collision_detection.pyと同じロジック）
            frame_diff = self.frame_count - getattr(self, 'last_mesh_update', -999)
            mesh_condition = (
                self.enable_mesh_generation and (
                    getattr(self, 'force_mesh_update_requested', False) or
                    (not hands_present and frame_diff >= 20) or  # 手がない時は20フレームごと
                    (frame_diff >= 60) or  # 最大60フレームで強制更新
                    (self.current_mesh is None)
                ) and
                len(points_3d) >= 100
            )
            
            if mesh_condition:
                self.logger.info(f"[MESH] Frame {self.frame_count}: *** UPDATING TERRAIN MESH *** with {len(points_3d)} points")
                mesh_start = time.perf_counter()
                
                # MeshPipeline を使用してメッシュ生成
                mesh_res = self.mesh_pipeline.generate_mesh(
                    points_3d,
                    self.current_tracked_hands,
                    force_update=getattr(self, 'force_mesh_update_requested', False),
                )
                
                if mesh_res.mesh is not None:
                    self.current_mesh = mesh_res.mesh
                    
                    # 空間インデックスは「メッシュが変わったとき」のみ再構築
                    if mesh_res.changed or self.spatial_index is None:
                        self.spatial_index = SpatialIndex(self.current_mesh, index_type=IndexType.BVH)
                        self.collision_searcher = CollisionSearcher(self.spatial_index)
                        self.collision_tester = SphereTriangleCollision(self.current_mesh)
                        self.logger.info(f"[MESH] Collision system updated with {self.current_mesh.num_triangles} triangles")
                    
                    # メッシュ更新タイムスタンプ
                    self.last_mesh_update = self.frame_count
                    
                    # 強制更新フラグをクリア
                    if hasattr(self, 'force_mesh_update_requested'):
                        self.force_mesh_update_requested = False
                    
                    mesh_time = (time.perf_counter() - mesh_start) * 1000
                    self.logger.info(f"[MESH] Frame {self.frame_count}: Mesh update completed in {mesh_time:.1f}ms")
                    self.performance_stats['mesh_generation_time'] = mesh_time
            
            # 衝突検出実行（demo_collision_detection.pyと同じロジック）
            if not self.args.no_collision and self.current_mesh and self.current_tracked_hands:
                self.logger.info(f"[COLLISION] Frame {self.frame_count}: *** CHECKING COLLISIONS *** "
                           f"with {len(self.current_tracked_hands)} hands and mesh available")
                
                collision_start = time.perf_counter()
                collision_events = self._detect_collisions_full(self.current_tracked_hands)
                collision_time = (time.perf_counter() - collision_start) * 1000
                
                self.performance_stats['collision_detection_time'] = collision_time
                self.performance_stats['collision_events_count'] += len(collision_events)
                
                if len(collision_events) > 0:
                    self.logger.info(f"[COLLISION] Frame {self.frame_count}: *** COLLISION DETECTED! *** "
                               f"{len(collision_events)} events")
                    
                    # 音響生成
                    if self.audio_enabled:
                        audio_start = time.perf_counter()
                        audio_notes = self._generate_audio_from_collisions(collision_events)
                        audio_time = (time.perf_counter() - audio_start) * 1000
                        
                        self.performance_stats['audio_synthesis_time'] = audio_time
                        self.performance_stats['audio_notes_played'] += audio_notes
                        
                        self.logger.info(f"[AUDIO] Frame {self.frame_count}: Generated {audio_notes} audio notes")
                else:
                    self.logger.info(f"[COLLISION] Frame {self.frame_count}: No collisions detected")
            
            # パフォーマンス統計更新
            total_pipeline_time = (time.perf_counter() - pipeline_start) * 1000
            self.performance_stats['total_pipeline_time'] = total_pipeline_time
                
        except Exception as e:
            self.logger.error(f"Mesh/collision pipeline error: {e}")
            import traceback
            traceback.print_exc()

    def _detect_collisions_full(self, tracked_hands: List[Any]) -> List[Any]:
        """完全な衝突検出実装（demo_collision_detection.pyベース）"""
        if not self.collision_searcher:
            self.logger.debug("No collision searcher available")
            return []
        
        events = []
        current_collision_points = []
        self.logger.debug(f"Processing {len(tracked_hands)} hands")
        
        for i, hand in enumerate(tracked_hands):
            if hand.position is None:
                self.logger.debug(f"Hand {i} has no position")
                continue
            
            hand_pos_np = np.array(hand.position)
            self.logger.debug(f"Hand {i} position: ({hand_pos_np[0]:.3f}, {hand_pos_np[1]:.3f}, {hand_pos_np[2]:.3f})")
            
            try:
                # 衝突検索
                search_result = self.collision_searcher._search_point(
                    hand_pos_np, 
                    self.args.sphere_radius
                )
                
                if not search_result.triangle_indices:
                    continue
                
                # 衝突テスト
                if self.collision_tester is not None:
                    collision_info = self.collision_tester.test_sphere_collision(
                        hand_pos_np, 
                        self.args.sphere_radius, 
                        search_result
                    )
                    
                    if collision_info.has_collision:
                        # 衝突イベント生成
                        from src.collision.events import CollisionEventQueue
                        if not hasattr(self, 'event_queue'):
                            self.event_queue = CollisionEventQueue()
                        
                        velocity = np.array(hand.velocity) if hasattr(hand, 'velocity') and hand.velocity is not None else np.zeros(3)
                        event = self.event_queue.create_event(collision_info, hand.id, hand_pos_np, velocity)
                        
                        if event:
                            events.append(event)
                            
                            # 接触点を保存
                            for cp in collision_info.contact_points:
                                current_collision_points.append(cp)
                            
                            self.logger.debug(f"Collision detected for hand {hand.id}")
                
            except Exception as e:
                self.logger.error(f"Error processing hand {i}: {e}")
        
        # 衝突点を保存
        self.current_collision_points = current_collision_points
        
        self.logger.debug(f"Total collision events: {len(events)}")
        return events

    def _generate_audio_from_collisions(self, collision_events: List[Any]) -> int:
        """衝突イベントから音響を生成"""
        if not self.audio_enabled or not self.audio_synthesizer:
            return 0
        
        notes_played = 0
        current_time = time.perf_counter()
        
        for event in collision_events:
            try:
                # クールダウンチェック
                if not self._check_audio_cooldown(event.hand_id, current_time):
                    continue
                
                # 音響パラメータマッピング
                if not hasattr(self, 'audio_mapper') or self.audio_mapper is None:
                    from src.sound.mapping import AudioMapper, ScaleType, InstrumentType
                    self.audio_mapper = AudioMapper(
                        scale=ScaleType.PENTATONIC,
                        default_instrument=InstrumentType.MARIMBA,
                        pitch_range=(48, 84),
                        enable_adaptive_mapping=True
                    )
                
                audio_params = self.audio_mapper.map_collision_event(event)
                
                # 音響再生
                voice_id = self.audio_synthesizer.play_audio_parameters(audio_params)
                
                if voice_id:
                    notes_played += 1
                    self.last_audio_trigger_time[event.hand_id] = current_time
                    self.logger.debug(f"[AUDIO-TRIGGER] Hand {event.hand_id}: Note triggered - voice_id: {voice_id}")
                else:
                    self.logger.warning(f"[AUDIO-SKIP] Could not play audio for event {event.event_id}")
            
            except Exception as e:
                self.logger.error(f"Audio generation error (event: {event.event_id}): {e}")
        
        return notes_played

    def _check_audio_cooldown(self, hand_id: str, current_time: float) -> bool:
        """音響クールダウンをチェック"""
        last_trigger = self.last_audio_trigger_time.get(hand_id, 0)
        time_since_last = current_time - last_trigger
        
        if time_since_last < self.audio_cooldown_time:
            self.logger.debug(f"[AUDIO-COOLDOWN] Hand {hand_id}: {time_since_last*1000:.1f}ms since last trigger")
            return False
        
        return True

    def _generate_audio_for_collision(self, hand: Any, collision_info: Any):
        """衝突音響生成"""
        try:
            if not self.audio_mapper or not self.audio_synthesizer:
                return
            
            # 音響パラメータマッピング
            # 衝突イベントを作成してマッピング
            from src.collision.events import CollisionEvent, CollisionIntensity
            collision_event = CollisionEvent(
                event_id=f"collision_{hand.id}_{int(time.time()*1000)}",
                hand_id=hand.id,
                contact_position=collision_info.contact_points[0].position,
                velocity=np.linalg.norm(hand.velocity) if hasattr(hand, 'velocity') and hand.velocity is not None else 0.0,
                penetration_depth=collision_info.total_penetration_depth,
                surface_normal=collision_info.collision_normal,
                contact_area=0.01,
                intensity=CollisionIntensity.MEDIUM,
                timestamp=time.perf_counter()
            )
            audio_params = self.audio_mapper.map_collision_event(collision_event)
            
            # 音響再生
            voice_id = self.audio_synthesizer.play_audio_parameters(audio_params)
            
            if voice_id:
                self.last_audio_trigger_time[hand.id] = time.perf_counter()
                self.performance_stats['audio_notes_played'] += 1
                self.logger.debug(f"Audio played for hand {hand.id}: voice_id={voice_id}")
            
        except Exception as e:
            self.logger.error(f"Audio generation error: {e}")

    def _display_final_stats(self, execution_time: float, frame_count: int, fps_samples: List[float]):
        """最終統計表示"""
        avg_fps = frame_count / execution_time if execution_time > 0 else 0
        max_fps = max(fps_samples) if fps_samples else 0
        min_fps = min(fps_samples) if fps_samples else 0
        
        print("\n" + "=" * 60)
        print("60fps Tracking Demo - Final Results")
        print("=" * 60)
        print(f"Execution Time: {execution_time:.1f}s")
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Max FPS: {max_fps:.1f}")
        print(f"Min FPS: {min_fps:.1f}")
        print(f"Collision Events: {self.performance_stats['collision_events_count']}")
        print(f"Audio Notes: {self.performance_stats['audio_notes_played']}")
        
        # 統合コントローラー統計
        if self.integrated_controller:
            controller_stats = self.integrated_controller.get_performance_stats()
            print(f"\n60fps Tracking System:")
            print(f"  Interpolation Rate: {controller_stats.get('interpolation_rate', 0):.1f}%")
            print(f"  MediaPipe Executions: {controller_stats.get('mediapipe_executions', 0)}")
            print(f"  Tracking Accuracy: {controller_stats.get('tracking_accuracy', 0):.1f}%")
        
        print("=" * 60)

    def cleanup(self):
        """リソースクリーンアップ"""
        self.logger.info("Cleaning up resources...")
        
        try:
            # 音響システム停止
            if self.audio_synthesizer:
                self.audio_synthesizer.stop_engine()
            
            # 統合コントローラー停止
            if self.integrated_controller:
                self.integrated_controller.stop()
            
            # カメラ停止
            if self.camera and hasattr(self.camera, 'stop'):
                self.camera.stop()
            
            # 表示システム停止
            if self.dual_viewer:
                cv2.destroyAllWindows()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# =============================================================================
# コマンドライン引数解析
# =============================================================================

def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="60fps Hand Tracking Demo with Unified Events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python3 demo_60fps_tracking.py                    # デフォルト設定
    python3 demo_60fps_tracking.py --no-audio         # 音響無効
    python3 demo_60fps_tracking.py --headless         # ヘッドレスモード
    python3 demo_60fps_tracking.py --high-resolution  # 高解像度モード
    python3 demo_60fps_tracking.py --sphere-radius 0.08 # 球半径8cm

操作方法:
    Q/ESC: 終了
    H: 手検出 ON/OFF
    T: トラッキング ON/OFF
    A: 音響合成 ON/OFF
    P: パフォーマンス統計表示
        """
    )
    
    # 基本設定
    parser.add_argument('--no-filter', action='store_true', help='深度フィルタを無効にする')
    parser.add_argument('--no-hand-detection', action='store_true', help='手検出を無効にする')
    parser.add_argument('--no-tracking', action='store_true', help='トラッキングを無効にする')
    parser.add_argument('--gpu-mediapipe', action='store_true', help='MediaPipeでGPUを使用')
    
    # 解像度設定
    parser.add_argument('--low-resolution', action='store_true', default=True, help='低解像度モード (424x240)')
    parser.add_argument('--high-resolution', action='store_true', help='高解像度モード (848x480)')
    parser.add_argument('--depth-width', type=int, help='深度解像度幅を直接指定')
    parser.add_argument('--depth-height', type=int, help='深度解像度高さを直接指定')
    
    # メッシュ・衝突設定
    parser.add_argument('--no-mesh', action='store_true', help='メッシュ生成を無効にする')
    parser.add_argument('--no-collision', action='store_true', help='衝突検出を無効にする')
    parser.add_argument('--sphere-radius', type=float, default=DEFAULT_SPHERE_RADIUS, help='衝突検出球の半径（メートル）')
    
    # 音響設定
    parser.add_argument('--no-audio', action='store_true', help='音響合成を無効にする')
    parser.add_argument('--audio-scale', type=str, default='PENTATONIC', 
                       choices=['PENTATONIC', 'MAJOR', 'MINOR', 'DORIAN', 'MIXOLYDIAN', 'CHROMATIC', 'BLUES'],
                       help='音階の種類')
    parser.add_argument('--audio-instrument', type=str, default='MARIMBA',
                       choices=['MARIMBA', 'SYNTH_PAD', 'BELL', 'PLUCK', 'BASS', 'LEAD', 'PERCUSSION', 'AMBIENT'],
                       help='楽器の種類')
    parser.add_argument('--audio-polyphony', type=int, default=DEFAULT_AUDIO_POLYPHONY, help='最大同時発音数')
    parser.add_argument('--audio-volume', type=float, default=DEFAULT_MASTER_VOLUME, help='マスター音量 (0.0-1.0)')
    
    # 表示設定
    parser.add_argument('--window-width', type=int, default=640, help='RGBウィンドウの幅')
    parser.add_argument('--window-height', type=int, default=480, help='RGBウィンドウの高さ')
    
    # モード設定
    parser.add_argument('--headless', action='store_true', help='ヘッドレスモード（GUI無効）')
    parser.add_argument('--headless-duration', type=int, default=30, help='ヘッドレスモード実行時間（秒）')
    
    return parser.parse_args()


# =============================================================================
# メイン関数
# =============================================================================

def main():
    """メイン関数"""
    args = parse_arguments()
    
    # 引数検証
    if args.sphere_radius <= 0.0 or args.sphere_radius > 0.5:
        print("Error: --sphere-radius must be between 0.0 and 0.5")
        return 1
    
    if args.audio_volume < 0.0 or args.audio_volume > 1.0:
        print("Error: --audio-volume must be between 0.0 and 1.0")
        return 1
    
    # 解像度設定
    if args.depth_width and args.depth_height:
        pass  # 直接指定
    elif args.high_resolution:
        args.depth_width, args.depth_height = HIGH_RESOLUTION
    else:
        args.depth_width, args.depth_height = LOW_RESOLUTION
    
    # 音響パラメータ変換
    try:
        args.audio_scale = ScaleType[args.audio_scale]
        args.audio_instrument = InstrumentType[args.audio_instrument]
    except KeyError as e:
        print(f"Error: Invalid audio parameter: {e}")
        return 1
    
    # 設定表示
    print("=" * 60)
    print("60fps Hand Tracking Demo")
    print("=" * 60)
    print(f"Resolution: {args.depth_width}x{args.depth_height}")
    print(f"Hand Detection: {'Disabled' if args.no_hand_detection else 'Enabled'}")
    print(f"Mesh Generation: {'Disabled' if args.no_mesh else 'Enabled'}")
    print(f"Collision Detection: {'Disabled' if args.no_collision else 'Enabled'}")
    print(f"Audio Synthesis: {'Disabled' if args.no_audio else 'Enabled'}")
    if not args.no_audio:
        print(f"  - Scale: {args.audio_scale.value}")
        print(f"  - Instrument: {args.audio_instrument.value}")
        print(f"  - Volume: {args.audio_volume:.1f}")
    print(f"Mode: {'Headless' if args.headless else 'GUI'}")
    print("=" * 60)
    
    # デモ実行
    demo = None
    try:
        demo = Demo60FPSTracker(args)
        demo.run()
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 0
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if demo:
            demo.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 