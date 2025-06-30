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
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import signal
import numpy as np
import cv2

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OrbbecSDKの動的インポート
HAS_ORBBEC_SDK = False
try:
    from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBError, OBFormat
    HAS_ORBBEC_SDK = True
    print("OrbbecSDK is available")
except ImportError:
    print("Warning: OrbbecSDK is not available. Using mock implementations.")
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
    print("MediaPipe is available")
except ImportError:
    print("Warning: MediaPipe is not available. Hand detection will be disabled.")
    # Mock MediaPipe
    class MockMediaPipe:
        class solutions:
            class hands:
                Hands = lambda **kwargs: None
            class drawing_utils:
                @staticmethod
                def draw_landmarks(*args): pass
            class hands_connections:
                HAND_CONNECTIONS = []

# NumPy/SciPy/Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Warning: Open3D is not available. 3D visualization will be disabled.")
    HAS_OPEN3D = False

# 音響ライブラリ（オプション）
HAS_AUDIO = False
try:
    import pyo
    HAS_AUDIO = True
    print("Pyo audio engine is available")
except ImportError:
    print("Warning: Pyo audio engine is not available. Audio synthesis will be disabled.")

# 必要なクラスのimport
from typing import Optional, List
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

# フェーズ別コンポーネント
from src.debug.dual_viewer import DualViewer
from src.input.depth_filter import DepthFilter, FilterType
from src.input.pointcloud import PointCloudConverter
from src.config import get_config, InputConfig

# GPU加速コンポーネント（CuPy利用可能時のみ）
try:
    from src.collision.distance_gpu import GPUDistanceCalculator, create_gpu_distance_calculator
    from src.mesh.delaunay_gpu import GPUDelaunayTriangulator, create_gpu_triangulator
    HAS_GPU_ACCELERATION = True
    print("🚀 GPU acceleration modules loaded (CuPy available)")
except ImportError:
    HAS_GPU_ACCELERATION = False
    print("⚠️ GPU acceleration unavailable (CuPy not installed)")


class FullPipelineViewer(DualViewer):
    """全フェーズ統合拡張DualViewer（手検出+メッシュ生成+衝突検出+音響生成）"""
    
    def __init__(self, **kwargs):
        # 音響関連パラメータ
        self.enable_audio_synthesis = kwargs.pop('enable_audio_synthesis', True)
        self.audio_scale = kwargs.pop('audio_scale', ScaleType.PENTATONIC)
        self.audio_instrument = kwargs.pop('audio_instrument', InstrumentType.MARIMBA)
        self.audio_polyphony = kwargs.pop('audio_polyphony', 16)
        self.audio_master_volume = kwargs.pop('audio_master_volume', 0.7)
        
        # ヘッドレスモード設定
        self.headless_mode = kwargs.pop('headless_mode', False)
        self.headless_duration = kwargs.pop('headless_duration', 30)
        self.pure_headless_mode = kwargs.pop('pure_headless_mode', False)
        
        # 衝突検出パラメータ
        self.enable_mesh_generation = kwargs.pop('enable_mesh_generation', True)
        self.enable_collision_detection = kwargs.pop('enable_collision_detection', True)
        self.enable_collision_visualization = kwargs.pop('enable_collision_visualization', True)
        self.sphere_radius = kwargs.pop('sphere_radius', 0.05)  # 5cm
        
        # メッシュ更新間隔制御
        self.mesh_update_interval = kwargs.pop('mesh_update_interval', 10)  # 10フレームごと
        self.max_mesh_skip_frames = kwargs.pop('max_mesh_skip_frames', 60)  # 最大60フレームスキップ
        
        # ボクセルダウンサンプリングパラメータ（親クラスに渡さない）
        self.voxel_downsampling_enabled = kwargs.pop('enable_voxel_downsampling', True)
        self.voxel_size = kwargs.pop('voxel_size', 0.005)  # 5mm デフォルト
        
        # 親クラス初期化
        super().__init__(**kwargs)
        
        # 地形メッシュ生成コンポーネント
        self.projector = PointCloudProjector(
            resolution=0.01,  # 1cm解像度
            method=ProjectionMethod.MEDIAN_HEIGHT,
            fill_holes=True
        )
        
        self.triangulator = DelaunayTriangulator(
            adaptive_sampling=True,
            boundary_points=True,
            quality_threshold=0.3,
            use_gpu=True
        )
        
        self.simplifier = MeshSimplifier(
            target_reduction=0.7,  # 70%削減でリアルタイム用に軽量化
            preserve_boundary=True
        )
        
        # 衝突検出コンポーネント
        self.spatial_index: Optional[SpatialIndex] = None
        self.collision_searcher: Optional[CollisionSearcher] = None
        self.collision_tester: Optional[SphereTriangleCollision] = None
        self.event_queue = CollisionEventQueue()
        
        # 音響生成コンポーネント
        self.audio_mapper: Optional[AudioMapper] = None
        self.audio_synthesizer: Optional[AudioSynthesizer] = None
        self.voice_manager: Optional[VoiceManager] = None
        self.audio_enabled = False  # 音響エンジンの状態
        
        # 音響クールダウン管理（「叩いた瞬間」のみ音を鳴らす）
        self.audio_cooldown_time = 0.15  # 150ms間隔での音発生制限
        self.last_audio_trigger_time = {}  # hand_id別の最後のトリガー時間
        
        # 状態管理
        self.current_mesh = None
        self.current_collision_points = []
        self.current_tracked_hands = []  # 直近フレームのトラッキング結果
        self.frame_counter = 0
        self.last_mesh_update = -999  # 初回メッシュ生成を確実にするため負の値で初期化
        self.force_mesh_update_requested = False  # メッシュ強制更新フラグ
        
        # パフォーマンス統計
        self.perf_stats = {
            'frame_count': 0,
            'mesh_generation_time': 0.0,
            'collision_detection_time': 0.0,
            'audio_synthesis_time': 0.0,
            'collision_events_count': 0,
            'audio_notes_played': 0,
            'total_pipeline_time': 0.0
        }
        
        # メッシュとコリジョンの可視化オブジェクト
        self.mesh_geometries = []
        self.collision_geometries = []
        
        # 音響システム初期化
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
        
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
        
        # カメラ初期化は親クラスで行われるため削除
        self.enable_hand_detection = True
        self.enable_hand_tracking = True  # 手トラッキングを有効化
        self.enable_tracking = True
        self.min_detection_confidence = 0.2  # 検出感度を上げてテスト
        self.hands_2d = MediaPipeHandsWrapper(
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
        # projector_3dとtrackerの初期化は親クラスの初期化後に行う
        self.projector_3d = None
        self.tracker = None
        
        # 初期化完了フラグ
        self._components_initialized = False
    
    def _initialize_audio_system(self):
        """音響システムを初期化"""
        try:
            print("音響システムを初期化中...")
            
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
                print("音響システム初期化完了")
            else:
                print("音響エンジンの開始に失敗しました")
                self.audio_enabled = False
        
        except Exception as e:
            print(f"音響システム初期化エラー: {e}")
            self.audio_enabled = False
    
    def _shutdown_audio_system(self):
        """音響システムを停止（安全版）"""
        try:
            print("[AUDIO-SHUTDOWN] 音響システムを停止中...")
            self.audio_enabled = False  # 最初に無効化して新しい音生成を防ぐ
            
            # ボイス管理システムの停止
            if self.voice_manager:
                try:
                    self.voice_manager.stop_all_voices(fade_out_time=0.01)  # 短時間フェード
                    time.sleep(0.05)  # 少し待機してボイス停止を確実にする
                    self.voice_manager = None
                except Exception as e:
                    print(f"[AUDIO-SHUTDOWN] VoiceManager停止エラー: {e}")
            
            # シンセサイザーエンジンの停止
            if self.audio_synthesizer:
                try:
                    self.audio_synthesizer.stop_engine()
                    time.sleep(0.05)  # 少し待機してエンジン停止を確実にする
                    self.audio_synthesizer = None
                except Exception as e:
                    print(f"[AUDIO-SHUTDOWN] Synthesizer停止エラー: {e}")
            
            # 音響マッパーもクリア
            self.audio_mapper = None
            
            print("[AUDIO-SHUTDOWN] 音響システムを停止しました")
        
        except Exception as e:
            print(f"[AUDIO-SHUTDOWN] 音響システム停止エラー: {e}")
            # エラーでも状態を無効にする
            self.audio_enabled = False
    
    def run(self):
        """ビューワーを実行"""
        if self.headless_mode:
            self.run_headless()
        else:
            # 親クラスのrun()を呼び出し
            super().run()

    def _process_frame(self) -> bool:
        """
        1フレーム処理（衝突検出版オーバーライド）
        DualViewerのフレーム処理をオーバーライドして衝突検出を統合
        """
        frame_start_time = time.perf_counter()
        
        # 3D手検出コンポーネントの遅延初期化
        if not self._components_initialized and hasattr(self, 'camera') and self.camera is not None:
            try:
                print("Setting up 3D hand detection components...")
                # カメラの初期化確認
                if self.camera.depth_intrinsics is not None:
                    # 信頼度閾値を下げてテスト
                    self.projector_3d = Hand3DProjector(
                        self.camera.depth_intrinsics,
                        min_confidence_3d=0.1  # 10%に下げてテスト
                    )
                    self.tracker = Hand3DTracker()
                    self._components_initialized = True
                    print("3D hand detection components initialized with lowered confidence threshold")
                else:
                    print("Camera depth intrinsics not available")
            except Exception as e:
                print(f"3D component initialization error: {e}")
        
        # カメラがない場合は終了
        if self.camera is None:
            print("Camera not available")
            return False
        
        # フレーム取得
        frame_data = self.camera.get_frame(timeout_ms=100)
        if frame_data is None or frame_data.depth_frame is None:
            return True
        
        # 深度画像の抽出
        depth_data = np.frombuffer(frame_data.depth_frame.get_data(), dtype=np.uint16)
        # カメラの内部パラメータチェック
        if self.camera.depth_intrinsics is not None:
            depth_image = depth_data.reshape(
                (self.camera.depth_intrinsics.height, self.camera.depth_intrinsics.width)
            )
        else:
            print("Depth intrinsics not available")
            return True
        
        # フィルタ適用
        filter_start_time = time.perf_counter()
        if self.depth_filter is not None and self.enable_filter:
            depth_image = self.depth_filter.apply_filter(depth_image)
        self.performance_stats['filter_time'] = (time.perf_counter() - filter_start_time) * 1000
        
        # 点群生成（必要時）
        points_3d = None
        need_points_for_mesh = (self.enable_mesh_generation and 
                                self.frame_count - self.last_mesh_update >= self.mesh_update_interval)
        
        if self.pointcloud_converter and (self.frame_count % self.update_interval == 0 or need_points_for_mesh):
            pointcloud_start = time.perf_counter()
            # depth_imageは既にnumpy配列なので、numpy_to_pointcloudを使用
            points_3d, _ = self.pointcloud_converter.numpy_to_pointcloud(depth_image)
            self.performance_stats['pointcloud_time'] = (time.perf_counter() - pointcloud_start) * 1000
            if need_points_for_mesh:
                print(f"[MESH-PREP] Frame {self.frame_count}: Generated points for mesh update: {len(points_3d) if points_3d is not None else 'None'}")
        
        # 手検出処理（一元化：ここで1回のみ実行）
        hands_2d, hands_3d, tracked_hands = [], [], []
        
        # 実際に手検出を実行（DualViewerから継承したメソッドを使用）
        if self.enable_hand_detection and self.hands_2d is not None:
            hand_start_time = time.perf_counter()
            hands_2d, hands_3d, tracked_hands = self._process_hand_detection(depth_image)
            self.performance_stats['hand_detection_time'] = (time.perf_counter() - hand_start_time) * 1000
            print(f"[HAND-OPTIMIZED] Frame {self.frame_count}: Hand detection completed in {self.performance_stats['hand_detection_time']:.1f}ms - 2D:{len(hands_2d)}, 3D:{len(hands_3d)}, Tracked:{len(tracked_hands)}")
        else:
            self.performance_stats['hand_detection_time'] = 0.0
        
        # 手検出結果をクラス変数に保存（RGB表示で使い回すため）
        self.current_hands_2d = hands_2d
        self.current_hands_3d = hands_3d
        self.current_tracked_hands = tracked_hands
        
        # 手が検出された場合のみ詳細ログを出力
        if len(hands_2d) > 0 or len(hands_3d) > 0 or len(tracked_hands) > 0:
            print(f"[HANDS] Frame {self.frame_count}: *** HANDS DETECTED *** 2D:{len(hands_2d)} 3D:{len(hands_3d)} Tracked:{len(tracked_hands)}")
        
        # 衝突検出とメッシュ生成のパイプライン
        pipeline_start = time.perf_counter()
        self.frame_counter = self.frame_count  # フレームカウンターを同期
        
        # 手が存在するか判定
        hands_present = (len(hands_2d) > 0 or len(hands_3d) > 0 or len(tracked_hands) > 0)
        frame_diff = self.frame_count - self.last_mesh_update

        # 地形メッシュ生成（条件判定）
        mesh_condition_check = (
            self.enable_mesh_generation and (
                # 強制更新要求がある場合は即更新
                self.force_mesh_update_requested or
                # 手が写っていない & 通常間隔を超えた
                (not hands_present and frame_diff >= self.mesh_update_interval) or
                # 最大スキップ時間を超えた
                (frame_diff >= self.max_mesh_skip_frames) or
                # まだメッシュが無い
                (self.current_mesh is None)
            ) and
            points_3d is not None and len(points_3d) > 100
        )
        
        # メッシュ生成条件の診断ログ
        if self.frame_count % 10 == 0:  # 10フレーム毎に診断ログ
            print(f"[MESH-DIAG] Frame {self.frame_count}: enable_mesh={self.enable_mesh_generation}, "
                  f"frame_diff={frame_diff}, "
                  f"interval={self.mesh_update_interval}, "
                  f"max_skip={self.max_mesh_skip_frames}, "
                  f"points={len(points_3d) if points_3d is not None else 'None'}, "
                  f"condition={mesh_condition_check}")
        
        if mesh_condition_check:
            mesh_start = time.perf_counter()
            points_len = len(points_3d) if points_3d is not None else 0
            print(f"[MESH] Frame {self.frame_count}: *** UPDATING TERRAIN MESH *** with {points_len} points")
            self._update_terrain_mesh(points_3d)
            self.last_mesh_update = self.frame_count
            # 強制更新フラグをクリア
            self.force_mesh_update_requested = False
            self.perf_stats['mesh_generation_time'] = (time.perf_counter() - mesh_start) * 1000
            print(f"[MESH] Frame {self.frame_count}: Mesh update completed in {self.perf_stats['mesh_generation_time']:.1f}ms")
        
        # 衝突検出
        collision_events = []
        if (self.enable_collision_detection and self.current_mesh is not None and tracked_hands):
            print(f"[COLLISION] Frame {self.frame_count}: *** CHECKING COLLISIONS *** with {len(tracked_hands)} hands and mesh available")
            collision_start = time.perf_counter()
            collision_events = self._detect_collisions(tracked_hands)
            self.perf_stats['collision_detection_time'] = (time.perf_counter() - collision_start) * 1000
            self.perf_stats['collision_events_count'] += len(collision_events)
            if len(collision_events) > 0:
                print(f"[COLLISION] Frame {self.frame_count}: *** COLLISION DETECTED! *** {len(collision_events)} events")
            else:
                print(f"[COLLISION] Frame {self.frame_count}: No collisions detected")
        
        # 音響生成
        if (self.enable_audio_synthesis and self.audio_enabled and collision_events):
            audio_start = time.perf_counter()
            print(f"[AUDIO] Frame {self.frame_count}: *** GENERATING AUDIO *** for {len(collision_events)} collision events")
            audio_notes = self._generate_audio(collision_events)
            self.perf_stats['audio_notes_played'] += audio_notes
            self.perf_stats['audio_synthesis_time'] = (time.perf_counter() - audio_start) * 1000
            print(f"[AUDIO] Frame {self.frame_count}: Generated {audio_notes} audio notes in {self.perf_stats['audio_synthesis_time']:.1f}ms")
        
        self.perf_stats['total_pipeline_time'] = (time.perf_counter() - pipeline_start) * 1000
        
        # 衝突イベントをクラス変数に保存（RGB表示で使用）
        self.current_collision_events = collision_events
        
        # RGB表示処理（既存のDualViewerロジックを使用）
        if not self._process_rgb_display(frame_data):
            return False
        
        # 点群表示処理（間隔制御）
        if self.frame_count % self.update_interval == 0:
            if not self._process_pointcloud_display(frame_data):
                return False
        
        self.frame_count += 1
        self.performance_stats['frame_time'] = (time.perf_counter() - frame_start_time) * 1000
        
        return True

    def _update_terrain_mesh(self, points_3d):
        """地形メッシュを更新"""
        if points_3d is None or len(points_3d) < 100:
            return
        
        try:
            import time
            
            # 1. 点群投影
            projection_start = time.perf_counter()
            height_map = self.projector.project_points(points_3d)
            projection_time = (time.perf_counter() - projection_start) * 1000
            
            # 2. Delaunay三角分割
            triangulation_start = time.perf_counter()
            triangle_mesh = self.triangulator.triangulate_heightmap(height_map)
            triangulation_time = (time.perf_counter() - triangulation_start) * 1000
            
            if triangle_mesh is None or triangle_mesh.num_triangles == 0:
                return
            
            # 3. メッシュ簡略化
            simplification_start = time.perf_counter()
            simplified_mesh = self.simplifier.simplify_mesh(triangle_mesh)
            simplification_time = (time.perf_counter() - simplification_start) * 1000
            
            if simplified_mesh is None:
                simplified_mesh = triangle_mesh
            
            # 4. 空間インデックス構築
            self.spatial_index = SpatialIndex(simplified_mesh, index_type=IndexType.BVH)
            
            # 5. 衝突検出コンポーネント初期化
            self.collision_searcher = CollisionSearcher(self.spatial_index)
            self.collision_tester = SphereTriangleCollision(simplified_mesh)
            
            # メッシュ保存
            self.current_mesh = simplified_mesh
            
            # デバッグ用時間測定出力
            if hasattr(self, 'frame_counter') and self.frame_counter % 50 == 0:
                total_mesh_time = projection_time + triangulation_time + simplification_time
                print(f"[MESH] Projection: {projection_time:.1f}ms, Triangulation: {triangulation_time:.1f}ms, Simplification: {simplification_time:.1f}ms (Total: {total_mesh_time:.1f}ms)")
            
            print(f"メッシュ更新完了: {simplified_mesh.num_triangles}三角形")
            
        except Exception as e:
            print(f"メッシュ生成中にエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_collisions(self, tracked_hands: List[TrackedHand]) -> list:
        if not self.collision_searcher: 
            print(f"[DEBUG] _detect_collisions: No collision searcher available")
            return []
        events = []
        self.current_collision_points = []
        print(f"[DEBUG] _detect_collisions: Processing {len(tracked_hands)} hands")
        
        for i, hand in enumerate(tracked_hands):
            if hand.position is None: 
                print(f"[DEBUG] _detect_collisions: Hand {i} has no position")
                continue
            hand_pos_np = np.array(hand.position)
            print(f"[DEBUG] _detect_collisions: Hand {i} position: ({hand_pos_np[0]:.3f}, {hand_pos_np[1]:.3f}, {hand_pos_np[2]:.3f})")
            
            try:
                res = self.collision_searcher.search_near_hand(hand, override_radius=self.sphere_radius)
                print(f"[DEBUG] _detect_collisions: Hand {i} found {len(res.triangle_indices)} nearby triangles")
                
                if not res.triangle_indices: continue
                
                # 従来のCPU衝突検出
                if self.collision_tester is not None:
                    info = self.collision_tester.test_sphere_collision(hand_pos_np, self.sphere_radius, res)
                else:
                    continue
                
                print(f"[DEBUG] _detect_collisions: Hand {i} collision test result: {info.has_collision}")
                
                if info.has_collision:
                    velocity = np.array(hand.velocity) if hasattr(hand, 'velocity') and hand.velocity is not None else np.zeros(3)
                    event = self.event_queue.create_event(info, hand.id, hand_pos_np, velocity)
                    if event:
                        events.append(event)
                        for cp in info.contact_points:
                           self.current_collision_points.append(cp.position)
                        print(f"[DEBUG] _detect_collisions: Hand {i} generated collision event with {len(info.contact_points)} contact points")
            except Exception as e:
                logger.error(f"[DEBUG] _detect_collisions: Error processing hand {i}: {e}")
        
        print(f"[DEBUG] _detect_collisions: Total collision events: {len(events)}")
        return events
    
    def _generate_audio(self, collision_events):
        """衝突イベントから音響を生成（クールダウン機構付き）"""
        if not self.audio_enabled or not self.audio_mapper or not self.voice_manager:
            return 0
        
        notes_played = 0
        current_time = time.perf_counter()
        
        for event in collision_events:
            try:
                # クールダウンチェック（手ID別）
                hand_id = event.hand_id
                last_trigger = self.last_audio_trigger_time.get(hand_id, 0)
                time_since_last = current_time - last_trigger
                
                if time_since_last < self.audio_cooldown_time:
                    print(f"[AUDIO-COOLDOWN] Hand {hand_id}: {time_since_last*1000:.1f}ms since last trigger, skipping")
                    continue
                
                # 衝突イベントを音響パラメータにマッピング
                audio_params = self.audio_mapper.map_collision_event(event)
                
                # 空間位置設定（numpy.float64 → Python float変換）
                spatial_position = np.array([
                    float(event.contact_position[0]),
                    0.0,
                    float(event.contact_position[2])
                ], dtype=float)
                
                # 音響再生
                voice_id = allocate_and_play(
                    self.voice_manager,
                    audio_params,
                    priority=7,
                    spatial_position=spatial_position
                )
                
                if voice_id:
                    notes_played += 1
                    # クールダウンタイマー更新
                    self.last_audio_trigger_time[hand_id] = current_time
                    print(f"[AUDIO-TRIGGER] Hand {hand_id}: Note triggered (cooldown reset)")
            
            except Exception as e:
                print(f"音響生成エラー（イベント: {event.event_id}）: {e}")
        
        # 終了したボイスのクリーンアップ（フレーム間隔を空けて負荷軽減）
        if self.voice_manager and self.frame_count % 10 == 0:  # 10フレームに1回のみ
            try:
                self.voice_manager.cleanup_finished_voices()
            except Exception as e:
                print(f"[AUDIO-CLEANUP] Error during cleanup: {e}")
        
        return notes_played

    def run_headless(self):
        """ヘッドレスモード実行（GUI無効化でFPS測定特化）"""
        import time
        
        print("\\n🖥️  ヘッドレスモード開始 - GUI無効化によるFPS最適化")
        print(f"⏱️  実行時間: {self.headless_duration}秒")
        print("=" * 50)
        
        start_time = time.time()
        frame_count = 0
        total_pipeline_time = 0.0
        
        # FPSの統計
        fps_samples = []
        frame_times = []
        last_report_time = start_time
        
        try:
            while True:
                frame_start = time.time()
                
                # フレーム処理（GUI無し）
                success = self._process_frame_headless()
                
                frame_end = time.time()
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)
                
                if success:
                    frame_count += 1
                    total_pipeline_time += frame_time
                    
                    # FPS計算
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_samples.append(current_fps)
                    
                    # 5秒間隔で統計表示
                    elapsed = frame_end - start_time
                    if elapsed - (last_report_time - start_time) >= 5.0:
                        avg_fps = sum(fps_samples[-100:]) / len(fps_samples[-100:]) if fps_samples else 0
                        print(f"📊 [{elapsed:.1f}s] フレーム: {frame_count}, 平均FPS: {avg_fps:.1f}, 現在FPS: {current_fps:.1f}")
                        last_report_time = frame_end
                
                # 実行時間チェック
                if time.time() - start_time >= self.headless_duration:
                    break
                    
        except KeyboardInterrupt:
            print("\\n⏹️  ユーザーによる中断")
        except Exception as e:
            print(f"\\n❌ ヘッドレスモード実行エラー: {e}")
            import traceback
            traceback.print_exc()
        
        # 統計計算と表示
        execution_time = time.time() - start_time
        avg_fps = frame_count / execution_time if execution_time > 0 else 0
        avg_frame_time = total_pipeline_time / frame_count if frame_count > 0 else 0
        max_fps = max(fps_samples) if fps_samples else 0
        min_fps = min(fps_samples) if fps_samples else 0
        
        # 結果表示
        print("\\n" + "=" * 50)
        print("🏁 ヘッドレスモード 実行結果")
        print("=" * 50)
        print(f"⏱️  実行時間: {execution_time:.1f}秒")
        print(f"🎬 総フレーム数: {frame_count}")
        print(f"🚀 平均FPS: {avg_fps:.1f}")
        print(f"⚡ 平均フレーム時間: {avg_frame_time*1000:.1f}ms")
        print(f"📈 最大FPS: {max_fps:.1f}")
        print(f"📉 最小FPS: {min_fps:.1f}")
        print(f"⚙️  平均パイプライン時間: {total_pipeline_time/frame_count*1000:.1f}ms" if frame_count > 0 else "⚙️  パイプライン時間: N/A")
        print(f"🎵 衝突イベント総数: {self.perf_stats.get('collision_events_count', 0)}")
        print(f"🔊 音響ノート総数: {getattr(self, 'audio_notes_played', 0)}")
        print()

    def _process_frame_headless(self) -> bool:
        """ヘッドレス専用フレーム処理（GUI描画なし）"""
        # ヘッドレスモード用モックデータ生成
        if not self.camera:
            # モック深度・カラー画像生成
            import numpy as np
            depth_image = np.random.randint(500, 2000, (240, 424), dtype=np.uint16)
            color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_data = (depth_image, color_image)
            
        else:
            # 実カメラからフレーム取得
            try:
                frame_data = self.camera.get_frame()
                if frame_data is None:
                    return False
                
                # フレームデータからdepth_imageとcolor_imageを抽出
                if hasattr(frame_data, 'depth_frame') and hasattr(frame_data, 'color_frame'):
                    depth_frame = frame_data.depth_frame
                    color_frame = frame_data.color_frame
                    
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                    
                    if color_frame is not None:
                        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                        color_image = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
                    else:
                        color_image = None
                else:
                    # フォールバック: タプルとして扱う
                    depth_image, color_image = frame_data, None
                    
                if depth_image is None:
                    return False
                    
            except Exception as e:
                print(f"❌ カメラフレーム取得エラー: {e}")
                return False
        
        # フレームデータの取得
        if hasattr(frame_data, 'depth_frame') and hasattr(frame_data, 'color_frame'):
            depth_frame = frame_data.depth_frame
            color_frame = frame_data.color_frame
            
            if depth_frame is not None:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            else:
                depth_image = None
                
            if color_frame is not None:
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                color_image = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
            else:
                color_image = None
        else:
            # フォールバック: タプルとして扱う
            depth_image, color_image = frame_data
        
        if depth_image is None:
            return False
            
        self.frame_counter += 1
        collision_events = []
        
        try:
            # 手検出処理（ヘッドレスでは簡易版）
            if self.enable_hand_detection and hasattr(self, 'hands_2d') and self.hands_2d and not getattr(self, 'pure_headless_mode', False):
                try:
                    # MediaPipe 2D検出（正しいメソッド名を使用）
                    self.current_hands_2d = self.hands_2d.detect_hands(color_image) if color_image is not None else []
                except Exception as e:
                    # ヘッドレスでは手検出エラーは無視
                    self.current_hands_2d = []
                
                # 3D投影は無効化（ヘッドレス）
                self.current_tracked_hands = []
            else:
                # 純粋ヘッドレスモードまたは手検出無効
                self.current_hands_2d = []
                self.current_tracked_hands = []
            
            # 点群生成（モックデータ）
            points_3d = None
            if self.enable_mesh_generation:
                # モック点群データ生成
                import numpy as np
                mock_points = np.random.rand(5000, 3).astype(np.float32)  # 5000点のモック点群
                mock_points[:, 2] += 0.5  # Z座標をカメラから離す
                points_3d = mock_points
                
            # メッシュ更新判定と生成
            if self.enable_mesh_generation and points_3d is not None:
                should_update = self._should_update_mesh()
                if should_update:
                    import time
                    mesh_start_time = time.time()
                    self._update_terrain_mesh(points_3d)
                    mesh_time = time.time() - mesh_start_time
                    self.perf_stats['mesh_generation_time'] += mesh_time
                    self.last_mesh_update = self.frame_counter
            
            # 衝突検出（ヘッドレスでは簡易版）
            if self.enable_collision_detection and self.current_tracked_hands and hasattr(self, 'current_mesh') and self.current_mesh:
                try:
                    import time
                    collision_start_time = time.time()
                    collision_events = self._detect_collisions(self.current_tracked_hands)
                    collision_time = time.time() - collision_start_time
                    self.perf_stats['collision_detection_time'] += collision_time
                    self.perf_stats['collision_events_count'] += len(collision_events)
                except Exception:
                    # ヘッドレスでは衝突検出エラーは無視
                    collision_events = []
            
            # 音響生成（音は出力される）
            if self.enable_audio_synthesis and collision_events:
                try:
                    import time
                    audio_start_time = time.time()
                    self._generate_audio(collision_events)
                    audio_time = time.time() - audio_start_time
                    self.perf_stats['audio_synthesis_time'] += audio_time
                except Exception:
                    # ヘッドレスでは音響エラーは無視
                    pass
            
            self.perf_stats['frame_count'] += 1
            
            # 処理時間シミュレーション（モックの場合）
            if not self.camera:
                # 実際の処理時間をシミュレーション
                import time as time_module
                processing_delay = 0.015  # 15ms 処理時間シミュレーション
                time_module.sleep(processing_delay)
            
            return True
            
        except Exception as e:
            # ヘッドレスではエラーでも継続
            if self.frame_counter <= 3:
                print(f"⚠️  フレーム処理警告: {e}")
            return True  # エラーでも継続
    
    def _should_update_mesh(self) -> bool:
        """メッシュ更新判定"""
        frames_since_update = self.frame_counter - self.last_mesh_update
        
        # 強制更新要求
        if hasattr(self, 'force_mesh_update_requested') and self.force_mesh_update_requested:
            self.force_mesh_update_requested = False
            return True
            
        # 手が検出されていない場合は通常間隔で更新
        if not self.current_tracked_hands:
            return frames_since_update >= self.mesh_update_interval
        
        # 手が検出されている場合は更新間隔を長くする
        # ただし、最大スキップフレーム数を超えたら強制更新
        return frames_since_update >= getattr(self, 'max_mesh_skip_frames', 60)

    def __del__(self):
        """デストラクタ - 音響システムを適切に停止"""
        try:
            if hasattr(self, 'audio_enabled') and self.audio_enabled:
                print("[DESTRUCTOR] 音響システムをクリーンアップ中...")
                self._shutdown_audio_system()
        except Exception as e:
            print(f"[DESTRUCTOR] デストラクタでエラー: {e}")
            
    def cleanup(self):
        """明示的なクリーンアップメソッド"""
        try:
            if self.audio_enabled:
                self._shutdown_audio_system()
        except Exception as e:
            print(f"[CLEANUP] クリーンアップエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    )
    
    # 基本設定
    parser.add_argument('--no-filter', action='store_true', help='深度フィルタを無効にする')
    parser.add_argument('--no-hand-detection', action='store_true', help='手検出を無効にする')
    parser.add_argument('--no-tracking', action='store_true', help='トラッキングを無効にする')
    parser.add_argument('--gpu-mediapipe', action='store_true', help='MediaPipeでGPUを使用')
    
    # 衝突検出設定
    parser.add_argument('--no-mesh', action='store_true', help='メッシュ生成を無効にする')
    parser.add_argument('--no-collision', action='store_true', help='衝突検出を無効にする')
    parser.add_argument('--no-collision-viz', action='store_true', help='衝突可視化を無効にする')
    parser.add_argument('--mesh-interval', type=int, default=15, help='メッシュ更新間隔（フレーム数） ※低解像度時は15frame推奨')
    parser.add_argument('--sphere-radius', type=float, default=0.05, help='衝突検出球の半径（メートル）')
    parser.add_argument('--max-mesh-skip', type=int, default=60, help='手が写っている場合でもこのフレーム数経過で強制更新')
    
    # 音響生成設定
    parser.add_argument('--no-audio', action='store_true', help='音響合成を無効にする')
    parser.add_argument('--audio-scale', type=str, default='PENTATONIC', 
                       choices=['PENTATONIC', 'MAJOR', 'MINOR', 'DORIAN', 'MIXOLYDIAN', 'CHROMATIC', 'BLUES'],
                       help='音階の種類')
    parser.add_argument('--audio-instrument', type=str, default='MARIMBA',
                       choices=['MARIMBA', 'SYNTH_PAD', 'BELL', 'PLUCK', 'BASS', 'LEAD', 'PERCUSSION', 'AMBIENT'],
                       help='楽器の種類')
    parser.add_argument('--audio-polyphony', type=int, default=16, help='最大同時発音数')
    parser.add_argument('--audio-volume', type=float, default=0.7, help='マスター音量 (0.0-1.0)')
    
    # 手検出設定
    parser.add_argument('--min-confidence', type=float, default=0.7, help='最小検出信頼度 (0.0-1.0)')
    
    # 表示設定
    parser.add_argument('--update-interval', type=int, default=3, help='点群更新間隔（フレーム数）')
    parser.add_argument('--point-size', type=float, default=2.0, help='点群の点サイズ')
    parser.add_argument('--high-resolution', action='store_true', help='高解像度表示 (1280x720)')
    
    # ヘッドレスモード（FPS向上のためのGUI無効化）
    parser.add_argument('--headless', action='store_true', help='ヘッドレスモード（GUI無効）※FPS大幅向上')
    parser.add_argument('--headless-duration', type=int, default=30, help='ヘッドレスモード実行時間（秒）')
    parser.add_argument('--headless-pure', action='store_true', help='純粋ヘッドレス（手検出無効、最大FPS測定）')
    
    # ウィンドウサイズ
    parser.add_argument('--window-width', type=int, default=640, help='RGBウィンドウの幅')
    parser.add_argument('--window-height', type=int, default=480, help='RGBウィンドウの高さ')
    
    args = parser.parse_args()
    
    # 設定値検証
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("Error: --min-confidence must be between 0.0 and 1.0")
        return 1
    
    if args.sphere_radius <= 0.0 or args.sphere_radius > 0.5:
        print("Error: --sphere-radius must be between 0.0 and 0.5")
        return 1
    
    if args.audio_polyphony < 1 or args.audio_polyphony > 64:
        print("Error: --audio-polyphony must be between 1 and 64")
        return 1
    
    if args.audio_volume < 0.0 or args.audio_volume > 1.0:
        print("Error: --audio-volume must be between 0.0 and 1.0")
        return 1
    
    # 音階と楽器の列挙値変換
    try:
        audio_scale = ScaleType[args.audio_scale]
        audio_instrument = InstrumentType[args.audio_instrument]
    except KeyError as e:
        print(f"Error: Invalid audio parameter: {e}")
        return 1
    
    # 情報表示
    print("=" * 70)
    print("Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）")
    print("=" * 70)
    
    print(f"深度フィルタ: {'無効' if args.no_filter else '有効'}")
    print(f"手検出: {'無効' if args.no_hand_detection else '有効'}")
    print(f"メッシュ生成: {'無効' if args.no_mesh else '有効'}")
    print(f"衝突検出: {'無効' if args.no_collision else '有効'}")
    if not args.no_collision:
        print(f"  - 球半径: {args.sphere_radius*100:.1f}cm")
        print(f"  - 可視化: {'無効' if args.no_collision_viz else '無効'}")
    print(f"音響合成: {'無効' if args.no_audio else '有効'}")
    if not args.no_audio:
        print(f"  - 音階: {audio_scale.value}")
        print(f"  - 楽器: {audio_instrument.value}")
        print(f"  - ポリフォニー: {args.audio_polyphony}")
        print(f"  - 音量: {args.audio_volume:.1f}")
    
    # ヘッドレスモード情報表示
    if args.headless:
        print(f"🖥️  ヘッドレスモード: 有効（GUI無効化でFPS向上）")
        print(f"⏱️  実行時間: {args.headless_duration}秒")
        print(f"🚀 予想FPS向上: +5-15 FPS (GUI負荷削除)")
    else:
        print(f"🖥️  表示モード: GUI有効")
    
    print("=" * 70)
    
    # FullPipelineViewer実行
    try:
        viewer = FullPipelineViewer(
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
        
        print("\n全フェーズ統合ビューワーを開始します...")
        print("=" * 70)
        
        # ヘッドレスモード時は直接実行
        if args.headless:
            print("🖥️  ヘッドレスモード: カメラ初期化をスキップ")
            print("🎯 モックデータによるFPS測定を開始します...")
            viewer.run()
            print("\nビューワーが正常に終了しました")
            return 0
        
        print("カメラを初期化中...")
        viewer.camera = OrbbecCamera(enable_color=True)
        
        # DualViewerの初期化を実行
        if not viewer.initialize():
            print("Failed to initialize dual viewer")
            return 1
        
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


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)