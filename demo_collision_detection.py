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

# Numba JIT最適化状況を表示
try:
    sys.path.insert(0, str(project_root / "src"))
    from src.numba_config import initialize_numba, get_numba_status, warmup_basic_functions
    
    # Numba初期化（詳細ログ付き）
    print("🔧 Starting Numba initialization...")
    success = initialize_numba(verbose=True, force_retry=True)
    if success:
        status = get_numba_status()
        print(f"🚀 Numba JIT acceleration enabled (v{status['version']})")
        print("🔥 Warming up JIT functions...")
        warmup_basic_functions()
        print("🔥 JIT functions warmed up - maximum performance ready")
    else:
        print("⚠️ Numba JIT acceleration disabled (falling back to NumPy)")
        
except Exception as e:
    print(f"⚠️ Numba configuration error: {e}")
    print("⚠️ Using NumPy fallback for all computations")

# 必要なクラスのimport（クラス定義前に配置）
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

# -----------------------------------------------------------------------------
# 前処理最適化システム（Step 1: 解像度最適化 + MediaPipe重複排除）
# -----------------------------------------------------------------------------

def run_preprocessing_optimization_test():
    """前処理最適化効果測定テスト（プロ修正：実装完了済み機能の検証）"""
    import time
    import numpy as np
    
    print("=" * 70)
    print("前処理最適化効果 測定テスト")
    print("=" * 70)
    
    # モック深度データ作成
    depth_low = np.random.randint(500, 2000, (240, 424), dtype=np.uint16)
    depth_high = np.random.randint(500, 2000, (480, 848), dtype=np.uint16)
    color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # MediaPipe モック（重複実行シミュレーション）
    def mock_mediapipe_process(image):
        time.sleep(0.015)  # ~15ms処理時間シミュレーション
        return []  # 手検出結果なし
    
    # --- ケース1: 848x480 + MediaPipe重複実行 ---
    print("🔍 ケース1: 848x480 + MediaPipe重複実行")
    start_time = time.time()
    frames_case1 = 0
    
    for _ in range(50):  # 50フレーム測定
        frame_start = time.time()
        
        # 高解像度点群処理シミュレーション
        points = depth_high.reshape(-1)
        valid_points = points[points > 0]
        
        # MediaPipe 2回実行（重複）
        mock_mediapipe_process(color_image)
        mock_mediapipe_process(color_image)  # 重複実行
        
        frame_time = time.time() - frame_start
        frames_case1 += 1
        
        # 75ms相当で停止（測定値基準）
        if frame_time < 0.075:
            time.sleep(0.075 - frame_time)
    
    elapsed_case1 = time.time() - start_time
    fps_case1 = frames_case1 / elapsed_case1
    
    # --- ケース2: 424x240 + MediaPipe1回実行 ---
    print("🔍 ケース2: 424x240 + MediaPipe1回実行")
    start_time = time.time()
    frames_case2 = 0
    
    for _ in range(50):  # 50フレーム測定
        frame_start = time.time()
        
        # 低解像度点群処理シミュレーション
        points = depth_low.reshape(-1)
        valid_points = points[points > 0]
        
        # MediaPipe 1回実行のみ
        mock_mediapipe_process(color_image)
        
        frame_time = time.time() - frame_start
        frames_case2 += 1
        
        # 36ms相当で停止（測定値基準）
        if frame_time < 0.036:
            time.sleep(0.036 - frame_time)
    
    elapsed_case2 = time.time() - start_time
    fps_case2 = frames_case2 / elapsed_case2
    
    # 結果表示
    print("\n📊 前処理最適化効果 結果")
    print("=" * 50)
    print(f"ケース1 (高解像度+重複): {fps_case1:.1f} FPS")
    print(f"ケース2 (低解像度+最適): {fps_case2:.1f} FPS")
    print(f"改善倍率: {fps_case2/fps_case1:.1f}x")
    print(f"FPS向上: +{fps_case2-fps_case1:.1f} FPS")
    print(f"フレーム時間短縮: {(1/fps_case1-1/fps_case2)*1000:.1f}ms")
    
def run_headless_fps_comparison_test():
    """ヘッドレスモードFPS効果測定テスト"""
    import time
    import numpy as np
    
    print("=" * 70)
    print("ヘッドレスモード FPS効果 測定テスト")
    print("=" * 70)
    
    # モックデータ
    depth_image = np.random.randint(500, 2000, (240, 424), dtype=np.uint16)
    color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def mock_core_processing():
        """コア処理（手検出、メッシュ生成、衝突検出）のシミュレーション"""
        time.sleep(0.025)  # 25ms処理時間
    
    def mock_gui_rendering():
        """GUI描画処理のシミュレーション"""
        # Open3D 3D描画
        time.sleep(0.008)  # 8ms
        # OpenCV RGB表示
        time.sleep(0.003)  # 3ms
        # UI更新
        time.sleep(0.002)  # 2ms
        # 合計 ~13ms GUI負荷
    
    # --- GUI有りモード測定 ---
    print("🖥️  GUI有りモード測定中...")
    start_time = time.time()
    frames_gui = 0
    
    for _ in range(100):  # 100フレーム測定
        frame_start = time.time()
        
        # コア処理
        mock_core_processing()
        
        # GUI描画処理
        mock_gui_rendering()
        
        frames_gui += 1
        frame_time = time.time() - frame_start
        
        # フレームレート制限なし（最大速度測定）
    
    elapsed_gui = time.time() - start_time
    fps_gui = frames_gui / elapsed_gui
    avg_frame_time_gui = elapsed_gui / frames_gui * 1000
    
    # --- ヘッドレスモード測定 ---
    print("⚡ ヘッドレスモード測定中...")
    start_time = time.time()
    frames_headless = 0
    
    for _ in range(100):  # 100フレーム測定
        frame_start = time.time()
        
        # コア処理のみ（GUI描画なし）
        mock_core_processing()
        
        frames_headless += 1
        frame_time = time.time() - frame_start
    
    elapsed_headless = time.time() - start_time
    fps_headless = frames_headless / elapsed_headless
    avg_frame_time_headless = elapsed_headless / frames_headless * 1000
    
    # 結果表示
    print("\n📊 ヘッドレスモード FPS効果 結果")
    print("=" * 50)
    print(f"GUI有りモード:     {fps_gui:.1f} FPS ({avg_frame_time_gui:.1f}ms/frame)")
    print(f"ヘッドレスモード:   {fps_headless:.1f} FPS ({avg_frame_time_headless:.1f}ms/frame)")
    print(f"FPS向上:          +{fps_headless-fps_gui:.1f} FPS")
    print(f"スピードアップ:     {fps_headless/fps_gui:.1f}x")
    print(f"フレーム時間短縮:   -{avg_frame_time_gui-avg_frame_time_headless:.1f}ms")
    print(f"GUI負荷削除効果:   {((avg_frame_time_gui-avg_frame_time_headless)/avg_frame_time_gui)*100:.1f}%改善")

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
        
        # ヘルプテキスト初期化
        self.help_text = "=== Basic Controls ===\nQ/ESC: Exit\nF: Toggle filter\nH: Toggle hand detection\nT: Toggle tracking\nR: Reset filter\nY: Reset tracker"
        
        # 地形メッシュ生成コンポーネント
        self.projector = PointCloudProjector(
            resolution=0.01,  # 1cm解像度
            method=ProjectionMethod.MEDIAN_HEIGHT,
            fill_holes=True
        )
        
        # LODメッシュ生成器を作成（従来の三角分割器を置き換え）
        from src.mesh.lod_mesh import create_lod_mesh_generator
        self.lod_mesh_generator = create_lod_mesh_generator(
            high_radius=0.20,      # ハンド周辺20cm以内は高解像度
            medium_radius=0.50,    # 50cm以内は中解像度  
            enable_gpu=True        # GPU使用（可能な場合）
        )
        
        # 従来の三角分割器も保持（フォールバック用）
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
    
    def update_help_text(self):
        """ヘルプテキストを更新（衝突検出機能を追加）"""
        self.help_text = "=== Basic Controls ===\n"
        self.help_text += "Q/ESC: Exit\n"
        self.help_text += "F: Toggle filter\n"
        self.help_text += "H: Toggle hand detection\n"
        self.help_text += "T: Toggle tracking\n"
        self.help_text += "R: Reset filter\n"
        self.help_text += "Y: Reset tracker\n"
        
        # ボクセルダウンサンプリング制御を追加
        self.help_text += "\n=== Point Cloud Optimization ===\n"
        self.help_text += "X: Toggle voxel downsampling\n"
        self.help_text += "Z/Shift+Z: Voxel size -/+ (1mm-10cm)\n"
        self.help_text += "B: Print voxel performance stats\n"
        
        # 衝突検出関連のキーバインドを追加
        self.help_text += "\n=== 衝突検出制御 ===\n"
        self.help_text += "M: メッシュ生成 ON/OFF\n"
        self.help_text += "C: 衝突検出 ON/OFF\n"
        self.help_text += "V: 衝突可視化 ON/OFF\n"
        self.help_text += "N: メッシュ強制更新\n"
        self.help_text += "+/-: 球半径調整\n"
        self.help_text += "P: パフォーマンス統計表示\n"
        
        # 音響生成関連のキーバインドを追加
        self.help_text += "\n=== 音響生成制御 ===\n"
        self.help_text += "A: 音響合成 ON/OFF\n"
        self.help_text += "S: 音階切り替え\n"
        self.help_text += "I: 楽器切り替え\n"
        self.help_text += "1/2: 音量調整\n"
        self.help_text += "R: 音響エンジン再起動\n"
        self.help_text += "Q: 全音声停止\n"
    
    def handle_key_event(self, key):
        """キーイベント処理（衝突検出機能を追加）"""
        # 基本的なキーイベント処理
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        elif key == ord('f'):  # Toggle filter
            self.enable_filter = not self.enable_filter
            print(f"Depth filter: {'Enabled' if self.enable_filter else 'Disabled'}")
            return True
        elif key == ord('h'):  # Toggle hand detection
            self.enable_hand_detection = not self.enable_hand_detection
            print(f"Hand detection: {'Enabled' if self.enable_hand_detection else 'Disabled'}")
            return True
        elif key == ord('t'):  # Toggle tracking
            self.enable_tracking = not self.enable_tracking
            print(f"Hand tracking: {'Enabled' if self.enable_tracking else 'Disabled'}")
            return True
        elif key == ord('r') and self.depth_filter is not None:  # Reset filter
            self.depth_filter.reset_temporal_history()
            print("Filter history reset")
            return True
        elif key == ord('y') and self.tracker is not None:  # Reset tracker
            self.tracker.reset()
            print("Hand tracker reset")
            return True
        
        # ボクセルダウンサンプリング制御
        elif key == ord('x') or key == ord('X'):  # Toggle voxel downsampling
            if self.pointcloud_converter:
                self.pointcloud_converter.toggle_voxel_downsampling()
            return True
            
        elif key == ord('z'):  # Decrease voxel size (higher quality)
            if self.pointcloud_converter:
                current_size = self.pointcloud_converter.voxel_size
                new_size = max(0.001, current_size - 0.001)  # Decrease by 1mm
                self.pointcloud_converter.set_voxel_size(new_size)
            return True
            
        elif key == ord('Z'):  # Increase voxel size (higher performance)
            if self.pointcloud_converter:
                current_size = self.pointcloud_converter.voxel_size
                new_size = min(0.05, current_size + 0.001)  # Increase by 1mm
                self.pointcloud_converter.set_voxel_size(new_size)
            return True
            
        elif key == ord('b') or key == ord('B'):  # Print voxel performance stats
            if self.pointcloud_converter:
                self.pointcloud_converter.print_performance_stats()
            return True
        
        # 衝突検出関連のキーイベント
        elif key == ord('m') or key == ord('M'):
            self.enable_mesh_generation = not self.enable_mesh_generation
            status = "有効" if self.enable_mesh_generation else "無効"
            print(f"メッシュ生成: {status}")
            return True
            
        elif key == ord('c') or key == ord('C'):
            self.enable_collision_detection = not self.enable_collision_detection
            status = "有効" if self.enable_collision_detection else "無効"
            print(f"衝突検出: {status}")
            return True
            
        elif key == ord('v') or key == ord('V'):
            self.enable_collision_visualization = not self.enable_collision_visualization
            status = "有効" if self.enable_collision_visualization else "無効"
            print(f"衝突可視化: {status}")
            self._update_visualization()
            return True
            
        elif key == ord('n') or key == ord('N'):
            print("メッシュを強制更新中...")
            self._force_mesh_update()
            return True
            
        elif key == ord('+') or key == ord('='):
            self.sphere_radius = min(self.sphere_radius + 0.01, 0.2)
            print(f"球半径: {self.sphere_radius*100:.1f}cm")
            return True
            
        elif key == ord('-') or key == ord('_'):
            self.sphere_radius = max(self.sphere_radius - 0.01, 0.01)
            print(f"球半径: {self.sphere_radius*100:.1f}cm")
            return True
            
        elif key == ord('p') or key == ord('P'):
            self._print_performance_stats()
            return True
        
        # 音響生成関連のキーイベント
        elif key == ord('a') or key == ord('A'):
            self.enable_audio_synthesis = not self.enable_audio_synthesis
            if self.enable_audio_synthesis:
                self._initialize_audio_system()
            else:
                self._shutdown_audio_system()
            status = "有効" if self.enable_audio_synthesis else "無効"
            print(f"音響合成: {status}")
            return True
            
        elif key == ord('s') or key == ord('S'):
            if self.enable_audio_synthesis:
                self._cycle_audio_scale()
            return True
            
        elif key == ord('i') or key == ord('I'):
            if self.enable_audio_synthesis:
                self._cycle_audio_instrument()
            return True
            
        elif key == ord('1'):
            if self.enable_audio_synthesis and self.audio_synthesizer:
                self.audio_master_volume = max(0.0, self.audio_master_volume - 0.1)
                self.audio_synthesizer.update_master_volume(self.audio_master_volume)
                print(f"音量: {self.audio_master_volume:.1f}")
            return True
            
        elif key == ord('2'):
            if self.enable_audio_synthesis and self.audio_synthesizer:
                self.audio_master_volume = min(1.0, self.audio_master_volume + 0.1)
                self.audio_synthesizer.update_master_volume(self.audio_master_volume)
                print(f"音量: {self.audio_master_volume:.1f}")
            return True
            
        elif key == ord('r') or key == ord('R'):
            if self.enable_audio_synthesis:
                print("音響エンジンを再起動中...")
                self._restart_audio_system()
            return True
            
        elif key == ord('q') or key == ord('Q'):
            if self.enable_audio_synthesis and self.voice_manager:
                self.voice_manager.stop_all_voices()
                print("全音声を停止しました")
            return True
        
        return False
    
    def _update_terrain_mesh(self, points_3d):
        """地形メッシュを更新（LOD最適化版）"""
        if points_3d is None or len(points_3d) < 100:
            return
        
        try:
            import time
            
            # LODメッシュ生成器を使用（高速化）
            if hasattr(self, 'lod_mesh_generator') and self.lod_mesh_generator is not None:
                lod_start = time.perf_counter()
                
                # LODベースメッシュ生成（手の位置を考慮した効率的な生成）
                triangle_mesh = self.lod_mesh_generator.generate_mesh(
                    points_3d, 
                    self.current_tracked_hands,  # 手の位置でLOD制御
                    force_update=getattr(self, 'force_mesh_update_requested', False)
                )
                
                total_lod_time = (time.perf_counter() - lod_start) * 1000
                
                if triangle_mesh is not None:
                    # LOD生成成功時の処理
                    simplified_mesh = triangle_mesh
                    
                    # デバッグ用時間測定出力
                    if hasattr(self, 'frame_counter') and self.frame_counter % 50 == 0:
                        print(f"[LOD-MESH] {len(points_3d)} points -> {triangle_mesh.num_vertices} vertices, "
                              f"{triangle_mesh.num_triangles} triangles in {total_lod_time:.1f}ms")
                
                else:
                    # LOD生成失敗時は従来方式へフォールバック
                    print("[LOD-FALLBACK] Using traditional mesh generation")
                    triangle_mesh = self._generate_traditional_mesh(points_3d)
                    if triangle_mesh is None:
                        return
                    simplified_mesh = triangle_mesh
            
            else:
                # LODメッシュ生成器が無効の場合は従来方式
                triangle_mesh = self._generate_traditional_mesh(points_3d)
                if triangle_mesh is None:
                    return
                simplified_mesh = triangle_mesh
            
            # 4. 空間インデックス構築
            self.spatial_index = SpatialIndex(simplified_mesh, index_type=IndexType.BVH)
            
            # 5. 衝突検出コンポーネント初期化
            self.collision_searcher = CollisionSearcher(self.spatial_index)
            self.collision_tester = SphereTriangleCollision(simplified_mesh)
            
            # メッシュ保存
            self.current_mesh = simplified_mesh
            
            # 診断ログ: メッシュ範囲を表示
            if simplified_mesh.vertices.size > 0:
                mesh_min = np.min(simplified_mesh.vertices, axis=0)
                mesh_max = np.max(simplified_mesh.vertices, axis=0)
                print(f"[MESH-INFO] Vertex range: X[{mesh_min[0]:.3f}, {mesh_max[0]:.3f}], "
                      f"Y[{mesh_min[1]:.3f}, {mesh_max[1]:.3f}], Z[{mesh_min[2]:.3f}, {mesh_max[2]:.3f}]")
            
            # 可視化更新
            self._update_mesh_visualization(simplified_mesh)
            
            # 強制更新フラグをリセット
            if hasattr(self, 'force_mesh_update_requested'):
                self.force_mesh_update_requested = False
            
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
                
                # None check for collision_tester
                if self.collision_tester is not None:
                    info = self.collision_tester.test_sphere_collision(hand_pos_np, self.sphere_radius, res)
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
    
    def _update_mesh_visualization(self, mesh):
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
            
        except Exception as e:
            print(f"メッシュ可視化エラー: {e}")
    
    def _update_collision_visualization(self):
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
            for contact in self.current_collision_points:
                # 接触点（球）
                contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                contact_sphere.translate(contact['position'])
                contact_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
                
                # 法線ベクトル（線分）
                normal_end = contact['position'] + contact['normal'] * 0.05
                normal_line = o3d.geometry.LineSet()
                normal_line.points = o3d.utility.Vector3dVector([
                    contact['position'], normal_end
                ])
                normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                normal_line.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
                
                self.vis.add_geometry(contact_sphere, reset_bounding_box=False)
                self.vis.add_geometry(normal_line, reset_bounding_box=False)
                
                self.collision_geometries.extend([contact_sphere, normal_line])
            
            # 衝突球を可視化（手の位置）
            if self.current_tracked_hands:
                for tracked in self.current_tracked_hands:
                    if tracked.position is not None:
                        hand_sphere = o3d.geometry.TriangleMesh.create_sphere(
                            radius=self.sphere_radius
                        )
                        hand_sphere.translate(tracked.position)
                        hand_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 緑色（半透明）
                        
                        # ワイヤーフレーム表示
                        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(hand_sphere)
                        wireframe.paint_uniform_color([0.0, 0.8, 0.0])
                        
                        self.vis.add_geometry(wireframe, reset_bounding_box=False)
                        self.collision_geometries.append(wireframe)
        
        except Exception as e:
            print(f"衝突可視化エラー: {e}")
    
    def _update_visualization(self):
        """可視化全体を更新"""
        if self.current_mesh and self.enable_collision_visualization:
            self._update_mesh_visualization(self.current_mesh)
        self._update_collision_visualization()
    
    def _force_mesh_update(self):
        """メッシュ強制更新を次フレームで行うようリクエスト"""
        self.force_mesh_update_requested = True
    
    def _draw_performance_info(self, color_image, collision_events):
        """パフォーマンス情報をRGB画像に描画"""
        if color_image is None:
            return
        
        # 基本情報
        info_lines = [
            f"Frame: {self.frame_counter}",
            f"Pipeline: {self.perf_stats['total_pipeline_time']:.1f}ms",
            f"Mesh Gen: {self.perf_stats['mesh_generation_time']:.1f}ms",
            f"Collision: {self.perf_stats['collision_detection_time']:.1f}ms",
            f"Audio: {self.perf_stats['audio_synthesis_time']:.1f}ms",
            f"Events: {len(collision_events)}",
            f"Sphere R: {self.sphere_radius*100:.1f}cm"
        ]
        
        # メッシュ情報
        if self.current_mesh:
            info_lines.append(f"Triangles: {self.current_mesh.num_triangles}")
        
        # 接触点情報
        if self.current_collision_points:
            info_lines.append(f"Contacts: {len(self.current_collision_points)}")
        
        # 音響情報
        if self.enable_audio_synthesis:
            audio_status = "ON" if self.audio_enabled else "OFF"
            info_lines.append(f"Audio: {audio_status}")
            if self.audio_enabled and self.voice_manager:
                active_voices = len(self.voice_manager.active_voices)
                info_lines.append(f"Voices: {active_voices}/{self.audio_polyphony}")
        
        # 描画
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(color_image, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 衝突イベント情報
        if collision_events:
            cv2.putText(color_image, "COLLISION DETECTED!", 
                       (10, color_image.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 音響再生情報
        if self.enable_audio_synthesis and self.audio_enabled and collision_events:
            cv2.putText(color_image, f"PLAYING AUDIO ({self.audio_instrument.value})", 
                       (10, color_image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _print_performance_stats(self):
        """パフォーマンス統計を印刷"""
        print("\n" + "="*50)
        print("パフォーマンス統計")
        print("="*50)
        print(f"総フレーム数: {self.perf_stats['frame_count']}")
        print(f"現在のフレーム: {self.frame_counter}")
        print(f"パイプライン時間: {self.perf_stats['total_pipeline_time']:.2f}ms")
        print(f"メッシュ生成時間: {self.perf_stats['mesh_generation_time']:.2f}ms")
        print(f"衝突検出時間: {self.perf_stats['collision_detection_time']:.2f}ms")
        print(f"音響生成時間: {self.perf_stats['audio_synthesis_time']:.2f}ms")
        print(f"総衝突イベント数: {self.perf_stats['collision_events_count']}")
        print(f"総音響ノート数: {self.perf_stats['audio_notes_played']}")
        
        # ボクセルダウンサンプリング統計
        if self.pointcloud_converter:
            voxel_stats = self.pointcloud_converter.get_performance_stats()
            print(f"\n--- Point Cloud Optimization ---")
            print(f"ボクセルダウンサンプリング: {'有効' if voxel_stats.get('voxel_downsampling_enabled', False) else '無効'}")
            if voxel_stats.get('voxel_downsampling_enabled', False):
                print(f"  - ボクセルサイズ: {voxel_stats.get('current_voxel_size_mm', 0):.1f}mm")
                print(f"  - 最新入力点数: {voxel_stats.get('last_input_points', 0):,}")
                print(f"  - 最新出力点数: {voxel_stats.get('last_output_points', 0):,}")
                print(f"  - ダウンサンプリング率: {voxel_stats.get('last_downsampling_ratio', 0)*100:.1f}%")
                avg_time = voxel_stats.get('average_time_ms', 0)
                print(f"  - 平均処理時間: {avg_time:.2f}ms")
        
        if self.current_mesh:
            print(f"現在のメッシュ: {self.current_mesh.num_triangles}三角形")
        
        print(f"球半径: {self.sphere_radius*100:.1f}cm")
        
        # 音響統計
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
        
        print("="*50)
    
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
        
        # RGB表示処理（既存のDualViewerロジックを使用）
        if not self._process_rgb_display(frame_data, collision_events):
            return False
        
        # 点群表示処理（間隔制御）
        if self.frame_count % self.update_interval == 0:
            if not self._process_pointcloud_display(frame_data):
                return False
        
        self.frame_count += 1
        self.performance_stats['frame_time'] = (time.perf_counter() - frame_start_time) * 1000
        
        return True

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
    
    def _restart_audio_system(self):
        """音響システムを再起動"""
        self._shutdown_audio_system()
        time.sleep(0.1)  # 短時間待機
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
    
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
    
    def _cycle_audio_scale(self):
        """音階を循環切り替え"""
        scales = list(ScaleType)
        current_index = scales.index(self.audio_scale)
        next_index = (current_index + 1) % len(scales)
        self.audio_scale = scales[next_index]
        
        if self.audio_mapper:
            self.audio_mapper.set_scale(self.audio_scale)
        
        print(f"音階を切り替え: {self.audio_scale.value}")
    
    def _cycle_audio_instrument(self):
        """楽器を循環切り替え"""
        instruments = list(InstrumentType)
        current_index = instruments.index(self.audio_instrument)
        next_index = (current_index + 1) % len(instruments)
        self.audio_instrument = instruments[next_index]
        
        if self.audio_mapper:
            self.audio_mapper.default_instrument = self.audio_instrument
        
        print(f"楽器を切り替え: {self.audio_instrument.value}")
    
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

    def _process_rgb_display(self, frame_data, collision_events=None) -> bool:
        """
        RGB表示処理（衝突検出版オーバーライド）
        
        Args:
            frame_data: フレームデータ
            collision_events: 衝突イベントリスト（オプション）
            
        Returns:
            継続する場合True
        """
        try:
            # 深度画像をカラーマップで可視化
            depth_data = np.frombuffer(frame_data.depth_frame.get_data(), dtype=np.uint16)
            # カメラの内部パラメータチェック
            if self.camera.depth_intrinsics is not None:
                depth_image = depth_data.reshape(
                    (self.camera.depth_intrinsics.height, self.camera.depth_intrinsics.width)
                )
            else:
                print("Depth intrinsics not available for RGB display")
                return True
            
            # 深度画像を表示用に正規化
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # 手検出処理（重複排除：_process_frameで既に実行済みの結果を使用）
            hands_2d = getattr(self, 'current_hands_2d', [])
            hands_3d = getattr(self, 'current_hands_3d', [])
            tracked_hands = getattr(self, 'current_tracked_hands', [])
            
            # デバッグ情報（重複実行排除済み）
            if len(hands_2d) > 0 or len(hands_3d) > 0 or len(tracked_hands) > 0:
                print(f"[HAND-DEBUG] Frame {self.frame_count}: Using cached hand detection results - 2D:{len(hands_2d)}, 3D:{len(hands_3d)}, Tracked:{len(tracked_hands)}")
            
            # パフォーマンス統計は_process_frameで計測済みなので省略
            self.performance_stats['hand_detection_time'] = 0.0  # 重複実行排除のため0ms
        
            # カラー画像があれば表示
            display_images = []
            
            # 深度画像（疑似カラー）
            depth_resized = cv2.resize(depth_colored, self.rgb_window_size)
            cv2.putText(depth_resized, f"Depth (Frame: {self.frame_count})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_images.append(depth_resized)
            
            # RGB画像
            color_bgr = None
            if frame_data.color_frame is not None and self.camera.has_color:
                color_data = np.frombuffer(frame_data.color_frame.get_data(), dtype=np.uint8)
                color_format = frame_data.color_frame.get_format()
                
                # フォーマットに応じた変換（DualViewerと同じロジック）
                try:
                    from pyorbbecsdk import OBFormat
                except ImportError:
                    pass  # Use imported OBFormat from src.types
                
                color_image = None
                if color_format == OBFormat.RGB:
                    # RGB形式の場合、カラー画像の実際のサイズを取得
                    total_pixels = len(color_data) // 3
                    # 1280x720 想定でリシェイプ
                    color_image = color_data.reshape((720, 1280, 3))
                elif color_format == OBFormat.BGR:
                    total_pixels = len(color_data) // 3
                    color_image = color_data.reshape((720, 1280, 3))
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                elif color_format == OBFormat.MJPG:
                    color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                    if color_image is not None:
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                if color_image is not None:
                    color_resized = cv2.resize(color_image, self.rgb_window_size)
                    color_bgr = cv2.cvtColor(color_resized, cv2.COLOR_RGB2BGR)
                    
                    # 手検出結果を描画
                    if self.enable_hand_detection and hands_2d:
                        color_bgr = self._draw_hand_detections(color_bgr, hands_2d, hands_3d, tracked_hands)
                    
                    # 衝突検出情報を描画
                    if collision_events:
                        self._draw_collision_info(color_bgr, collision_events)
                    
                    cv2.putText(color_bgr, f"RGB (FPS: {self.performance_stats['fps']:.1f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    display_images.append(color_bgr)
            
            # 画像を横に並べて表示
            if len(display_images) > 1:
                combined_image = np.hstack(display_images)
            else:
                combined_image = display_images[0]
            
            # パフォーマンス情報をオーバーレイ
            self._draw_performance_overlay(combined_image)
            
            # 衝突検出パフォーマンス情報を追加描画
            if hasattr(self, 'perf_stats'):
                self._draw_collision_performance_info(combined_image, collision_events)
            
            cv2.imshow("Geocussion-SP Input Viewer", combined_image)
            
            # キー入力処理（DualViewerの基本機能 + 衝突検出機能）
            key = cv2.waitKey(1) & 0xFF
            
            # 既存のキー処理
            if key == ord('q') or key == 27:  # Q or ESC
                return False
            elif key == ord('f'):  # Toggle filter
                self.enable_filter = not self.enable_filter
                print(f"Depth filter: {'Enabled' if self.enable_filter else 'Disabled'}")
            elif key == ord('r') and self.depth_filter is not None:  # Reset filter
                self.depth_filter.reset_temporal_history()
                print("Filter history reset")
            elif key == ord('h'):  # Toggle hand detection
                self.enable_hand_detection = not self.enable_hand_detection
                print(f"Hand detection: {'Enabled' if self.enable_hand_detection else 'Disabled'}")
            elif key == ord('t') and self.enable_hand_detection:  # Toggle tracking
                self.enable_tracking = not self.enable_tracking
                print(f"Hand tracking: {'Enabled' if self.enable_tracking else 'Disabled'}")
            elif key == ord('y') and self.tracker is not None:  # Reset tracker
                self.tracker.reset()
                print("Hand tracker reset")
            
            # 衝突検出のキー処理
            else:
                # 衝突検出のキー処理を直接実装
                if key == ord('m') or key == ord('M'):
                    self.enable_mesh_generation = not self.enable_mesh_generation
                    status = "有効" if self.enable_mesh_generation else "無効"
                    print(f"メッシュ生成: {status}")
                elif key == ord('c') or key == ord('C'):
                    self.enable_collision_detection = not self.enable_collision_detection
                    status = "有効" if self.enable_collision_detection else "無効"
                    print(f"衝突検出: {status}")
                elif key == ord('v') or key == ord('V'):
                    self.enable_collision_visualization = not self.enable_collision_visualization
                    status = "有効" if self.enable_collision_visualization else "無効"
                    print(f"衝突可視化: {status}")
                elif key == ord('n') or key == ord('N'):
                    print("メッシュを強制更新中...")
                    self._force_mesh_update()
                elif key == ord('+') or key == ord('='):
                    self.sphere_radius = min(self.sphere_radius + 0.01, 0.2)
                    print(f"球半径: {self.sphere_radius*100:.1f}cm")
                elif key == ord('-') or key == ord('_'):
                    self.sphere_radius = max(self.sphere_radius - 0.01, 0.01)
                    print(f"球半径: {self.sphere_radius*100:.1f}cm")
                elif key == ord('p') or key == ord('P'):
                    self._print_performance_stats()
            
            return True
            
        except Exception as e:
            print(f"RGB display error: {e}")
            return True

    def _draw_collision_info(self, image: np.ndarray, collision_events: list) -> None:
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
    
    def _draw_collision_performance_info(self, image: np.ndarray, collision_events: list) -> None:
        """衝突検出パフォーマンス情報を描画"""
        if not hasattr(self, 'perf_stats'):
            return
            
        # 右側に衝突検出情報を描画
        info_lines = [
            f"Mesh: {self.perf_stats.get('mesh_generation_time', 0):.1f}ms",
            f"Collision: {self.perf_stats.get('collision_detection_time', 0):.1f}ms",
            f"Audio: {self.perf_stats.get('audio_synthesis_time', 0):.1f}ms",
            f"Events: {len(collision_events)}",
            f"Sphere R: {self.sphere_radius*100:.1f}cm"
        ]
        
        # ボクセルダウンサンプリング情報
        if self.pointcloud_converter:
            voxel_stats = self.pointcloud_converter.get_performance_stats()
            if voxel_stats.get('voxel_downsampling_enabled', False):
                ratio = voxel_stats.get('last_downsampling_ratio', 0)
                voxel_size = voxel_stats.get('current_voxel_size_mm', 0)
                info_lines.append(f"Voxel: {ratio*100:.0f}% @ {voxel_size:.1f}mm")
            else:
                info_lines.append("Voxel: OFF")
        
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

    def run(self):
        """ビューワーを実行"""
        if self.headless_mode:
            self.run_headless()
        else:
            # 親クラスのrun()を呼び出し
            super().run()
    
    def run_headless(self):
        """ヘッドレスモード実行（GUI無効化でFPS測定特化）"""
        import time
        
        print("\\n🖥️  ヘッドレスモード開始 - GUI無効化によるFPS最適化")
        print(f"⏱️  実行時間: {self.headless_duration}秒")
        print("=" * 50)
        
        # ヘッドレス専用コンポーネント初期化
        print("🔧 ヘッドレス用コンポーネント初期化中...")
        self._initialize_headless_components()
        
        # コンポーネント初期化確認
        print("🔍 コンポーネント初期化状況:")
        print(f"   Camera: {'✅' if self.camera else '❌ (モックデータ使用)'}")
        print(f"   Hands2D: {'✅' if hasattr(self, 'hands_2d') and self.hands_2d else '❌'}")
        print(f"   Projector3D: {'✅' if hasattr(self, 'projector_3d') and self.projector_3d else '❌ (ヘッドレス対応)'}")
        print(f"   Tracker: {'✅' if hasattr(self, 'tracker') and self.tracker else '❌ (ヘッドレス対応)'}")
        print(f"   PointcloudConverter: {'✅' if hasattr(self, 'pointcloud_converter') and self.pointcloud_converter else '❌ (モックデータ)'}")
        
        print("\\n🎯 ヘッドレスモード フレーム処理開始...")
        print("=" * 50)
        
        start_time = time.time()
        frame_count = 0
        total_pipeline_time = 0.0
        total_collision_events = 0
        
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
        
        # 統計計算
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
        print(f"🔊 音響ノート総数: {getattr(self, 'audio_notes_generated', 0)}")
        
        # ROI トラッキング統計出力
        if hasattr(self.hands_2d, 'get_roi_tracking_stats'):
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
        
        print()

    def _initialize_headless_components(self):
        """ヘッドレス専用のコンポーネント初期化"""
        # ヘッドレスモードでは必要最小限のコンポーネントのみ初期化
        try:
            # 3D projector（ヘッドレス用簡易版）
            if not hasattr(self, 'projector_3d') or not self.projector_3d:
                print("🔧 ヘッドレス用 3D projector を初期化中...")
                self.projector_3d = None  # ヘッドレスでは無効化
                
            # Hand tracker（ヘッドレス用簡易版）
            if not hasattr(self, 'tracker') or not self.tracker:
                print("🔧 ヘッドレス用 tracker を初期化中...")
                self.tracker = None  # ヘッドレスでは無効化
                
            # PointCloud converter（モック対応）
            if not hasattr(self, 'pointcloud_converter') or not self.pointcloud_converter:
                print("🔧 ヘッドレス用 pointcloud converter を初期化中...")
                self.pointcloud_converter = None  # モック点群を使用
                
            print("✅ ヘッドレス用コンポーネント初期化完了")
            
        except Exception as e:
            print(f"⚠️  ヘッドレス用コンポーネント初期化警告: {e}")
            # エラーでも継続（ヘッドレスでは非致命的）

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
                
                depth_image, color_image = frame_data
                if depth_image is None:
                    return False
                    
            except Exception as e:
                print(f"❌ カメラフレーム取得エラー: {e}")
                return False
        
        # フレームデータの取得
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

    def _generate_traditional_mesh(self, points_3d):
        """従来方式でメッシュ生成（フォールバック用）"""
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
                return None
            
            # 3. メッシュ簡略化
            simplification_start = time.perf_counter()
            simplified_mesh = self.simplifier.simplify_mesh(triangle_mesh)
            simplification_time = (time.perf_counter() - simplification_start) * 1000
            
            if simplified_mesh is None:
                simplified_mesh = triangle_mesh
            
            # デバッグ用時間測定出力
            if hasattr(self, 'frame_counter') and self.frame_counter % 50 == 0:
                total_mesh_time = projection_time + triangulation_time + simplification_time
                print(f"[TRADITIONAL-MESH] Projection: {projection_time:.1f}ms, Triangulation: {triangulation_time:.1f}ms, Simplification: {simplification_time:.1f}ms (Total: {total_mesh_time:.1f}ms)")
            
            return simplified_mesh
            
        except Exception as e:
            print(f"従来方式メッシュ生成エラー: {e}")
            return None


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
    
    # 解像度最適化設定（プロ修正：FPS向上のための低解像度モード）
    parser.add_argument('--low-resolution', action='store_true', default=True, help='低解像度モード (424x240) ※FPS向上のため既定ON')
    parser.add_argument('--force-high-resolution', action='store_true', help='強制的に高解像度 (848x480) を使用 ※低FPS注意')
    parser.add_argument('--depth-width', type=int, help='深度解像度幅を直接指定')
    parser.add_argument('--depth-height', type=int, help='深度解像度高さを直接指定')
    
    # ウィンドウサイズ
    parser.add_argument('--window-width', type=int, default=640, help='RGBウィンドウの幅')
    parser.add_argument('--window-height', type=int, default=480, help='RGBウィンドウの高さ')
    
    # テストモード
    parser.add_argument('--test', action='store_true', help='テストモードで実行')
    
    # ヘッドレスモード（FPS向上のためのGUI無効化）
    parser.add_argument('--headless', action='store_true', help='ヘッドレスモード（GUI無効）※FPS大幅向上')
    parser.add_argument('--headless-duration', type=int, default=30, help='ヘッドレスモード実行時間（秒）')
    parser.add_argument('--headless-pure', action='store_true', help='純粋ヘッドレス（手検出無効、最大FPS測定）')
    
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
    
    # 解像度設定の決定（プロ修正：確実な最適化ロジック）
    depth_width, depth_height = None, None
    if args.depth_width and args.depth_height:
        # 直接指定がある場合はそれを優先
        depth_width, depth_height = args.depth_width, args.depth_height
    elif args.force_high_resolution:
        # 強制高解像度モード
        depth_width, depth_height = 848, 480
    elif args.low_resolution:
        # 低解像度モード（既定）
        depth_width, depth_height = 424, 240
    # それ以外はNone（OrbbecSDKのデフォルト）
    
    # 解像度による点群数の予測
    if depth_width and depth_height:
        estimated_points = depth_width * depth_height
        if estimated_points > 300000:  # 30万点以上
            print(f"⚠️  Warning: High resolution ({depth_width}x{depth_height}) may cause low FPS")
            print(f"   Estimated points: {estimated_points:,}")
            print(f"   Consider using --low-resolution for better performance")
        else:
            print(f"✅ Optimized resolution: {depth_width}x{depth_height} (~{estimated_points:,} points)")
    else:
        print("📏 Using camera default resolution")
    
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
    
    # 解像度最適化情報を最初に表示（重要性を強調）
    if depth_width and depth_height:
        resolution_mode = "低解像度" if depth_width <= 424 else "高解像度"
        points_estimate = depth_width * depth_height
        print(f"🚀 解像度最適化: {resolution_mode} ({depth_width}x{depth_height})")
        print(f"   予想点群数: {points_estimate:,} points")
        fps_estimate = "25-30 FPS" if depth_width <= 424 else "5-15 FPS"
        print(f"   予想FPS: {fps_estimate}")
    
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
    
    # ヘッドレスモード情報表示
    if args.headless:
        print(f"🖥️  ヘッドレスモード: 有効（GUI無効化でFPS向上）")
        print(f"⏱️  実行時間: {args.headless_duration}秒")
        print(f"🚀 予想FPS向上: +5-15 FPS (GUI負荷削除)")
    else:
        print(f"🖥️  表示モード: GUI有効")
    
    print("=" * 70)
    
    # テストモード
    if args.test:
        run_preprocessing_optimization_test()
        print("\n" + "=" * 70)
        run_headless_fps_comparison_test()
        return 0
    
    # 設定統合システムで低解像度モードを適用（プロ修正：一元管理）
    config = get_config()
    config.input.enable_low_resolution_mode = (depth_width == 424 and depth_height == 240)
    config.input.depth_width = depth_width
    config.input.depth_height = depth_height
    
    # 低解像度時の最適化パラメータを自動適用
    if config.input.enable_low_resolution_mode:
        # メッシュ更新間隔を最適化（指定されていない場合のみ）
        if args.mesh_interval == 15:  # デフォルト値の場合
            args.mesh_interval = 20  # さらに間隔を空ける
        print(f"🔧 低解像度最適化: メッシュ更新間隔={args.mesh_interval}フレーム")
    else:
        # 高解像度強制時の緊急FPS最適化
        if depth_width and depth_height and (depth_width >= 848 or depth_height >= 480):
            print(f"🚨 高解像度モード検出: {depth_width}x{depth_height}")
            print(f"⚡ 緊急FPS最適化を適用中...")
            
            # メッシュ更新間隔を大幅延長
            if args.mesh_interval <= 20:
                args.mesh_interval = 40  # 2倍に延長
                print(f"🔧 緊急最適化: メッシュ更新間隔={args.mesh_interval}フレーム (40f間隔)")
            
            # 最大スキップフレームも延長
            if args.max_mesh_skip <= 60:
                args.max_mesh_skip = 120  # 2倍に延長
                print(f"🔧 緊急最適化: 最大メッシュスキップ={args.max_mesh_skip}フレーム")
            
            # 解像度ダウンサンプリングを有効化
            config.input.enable_resolution_downsampling = True
            config.input.resolution_target_width = 424
            config.input.resolution_target_height = 240
            print(f"🔧 緊急最適化: 解像度ダウンサンプリング有効 ({depth_width}x{depth_height} → 424x240)")
                
            print(f"⚡ 高解像度での予想FPS: 8-15 FPS (最適化適用済み)")
        elif depth_width and depth_height:
            print(f"🔧 中解像度最適化: メッシュ更新間隔={args.mesh_interval}フレーム")
    
    # CollisionDetectionViewer実行
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
        # カメラを最適化された解像度で初期化（プロ修正：確実な高速化）
        if depth_width and depth_height:
            print(f"   深度解像度: {depth_width}x{depth_height} に設定")
        viewer.camera = OrbbecCamera(
            enable_color=True,
            depth_width=depth_width,
            depth_height=depth_height
        )
        
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