#!/usr/bin/env python3
"""
統合パイプライン処理（Clean Architecture適用）
責務: 入力→検出→メッシュ→衝突→音響の全フェーズ統合処理
"""

import time
import psutil
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

# 基本型・設定
from ..types import FrameData
from ..sound.mapping import ScaleType, InstrumentType

# 入力フェーズ
from ..input.stream import OrbbecCamera
from ..input.depth_filter import DepthFilter, FilterType
from ..input.pointcloud import PointCloudConverter

# 検出フェーズ
from ..detection.hands2d import MediaPipeHandsWrapper
from ..detection.hands3d import Hand3DProjector, DepthInterpolationMethod
from ..detection.tracker import Hand3DTracker, TrackedHand

# メッシュ生成フェーズ
from ..mesh.projection import PointCloudProjector, ProjectionMethod
from ..mesh.delaunay import DelaunayTriangulator
from ..mesh.simplify import MeshSimplifier

# LODメッシュ（利用可能な場合）
try:
    from ..mesh.lod_mesh import create_lod_mesh_generator
    HAS_LOD_MESH = True
except ImportError:
    HAS_LOD_MESH = False

# 衝突検出フェーズ
from ..collision.search import CollisionSearcher
from ..collision.sphere_tri import SphereTriangleCollision
from ..collision.events import CollisionEventQueue
from ..mesh.index import SpatialIndex, IndexType

# 音響生成フェーズ
from ..sound.mapping import AudioMapper
from ..sound.synth import AudioSynthesizer, create_audio_synthesizer
from ..sound.voice_mgr import VoiceManager, create_voice_manager, allocate_and_play, StealStrategy

# GPU加速（利用可能な場合）
try:
    from ..collision.distance_gpu import create_gpu_distance_calculator
    from ..mesh.delaunay_gpu import create_gpu_triangulator
    HAS_GPU_ACCELERATION = True
except ImportError:
    HAS_GPU_ACCELERATION = False


@dataclass
class PipelineResults:
    """パイプライン処理結果"""
    frame_data: Optional[FrameData] = None
    hands_2d: List = field(default_factory=list)
    hands_3d: List = field(default_factory=list)
    tracked_hands: List[TrackedHand] = field(default_factory=list)
    mesh: Optional[Any] = None
    collision_events: List = field(default_factory=list)
    audio_events: List = field(default_factory=list)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で結果を返す"""
        return {
            'frame_data': self.frame_data,
            'hands_2d': self.hands_2d,
            'hands_3d': self.hands_3d,
            'tracked_hands': self.tracked_hands,
            'mesh': self.mesh,
            'collision_events': self.collision_events,
            'audio_events': self.audio_events,
            'performance_stats': self.performance_stats
        }


@dataclass
class HandledPipelineConfig:
    """統合パイプライン設定"""
    # 入力設定
    enable_filter: bool = True
    enable_hand_detection: bool = True
    enable_tracking: bool = True
    min_detection_confidence: float = 0.7
    use_gpu_mediapipe: bool = False
    
    # メッシュ生成設定
    enable_mesh_generation: bool = True
    mesh_update_interval: int = 10
    max_mesh_skip_frames: int = 60
    mesh_resolution: float = 0.01
    mesh_quality_threshold: float = 0.3
    mesh_reduction: float = 0.7
    
    # 衝突検出設定
    enable_collision_detection: bool = True
    enable_collision_visualization: bool = True
    sphere_radius: float = 0.05
    
    # 音響合成設定
    enable_audio_synthesis: bool = True
    audio_scale: ScaleType = ScaleType.PENTATONIC
    audio_instrument: InstrumentType = InstrumentType.MARIMBA
    audio_polyphony: int = 16
    audio_master_volume: float = 0.7
    
    # ボクセルダウンサンプリング設定
    enable_voxel_downsampling: bool = True
    voxel_size: float = 0.005
    
    # パフォーマンス設定
    enable_gpu_acceleration: bool = True
    
    # ヘッドレスモード設定（新規追加）
    headless_mode: bool = False


class HandledPipeline:
    """
    統合パイプライン処理クラス
    責務: 入力→検出→メッシュ→衝突→音響の全フェーズ統合処理
    """
    
    def __init__(self, config: HandledPipelineConfig):
        """統合パイプライン初期化"""
        print("統合パイプライン初期化中...")
        
        # 設定保存
        self.config = config
        
        # エラーカウンター（無限ループ防止）
        self.color_extraction_errors = 0
        self.max_color_extraction_errors = 10
        self.color_extraction_disabled = False
        
        self.camera: Optional[OrbbecCamera] = None
        
        # 基本コンポーネント
        self.depth_filter: Optional[DepthFilter] = None
        self.pointcloud_converter: Optional[PointCloudConverter] = None
        self.hands_2d: Optional[MediaPipeHandsWrapper] = None
        self.projector_3d: Optional[Hand3DProjector] = None
        self.tracker: Optional[Hand3DTracker] = None
        
        # メッシュ生成コンポーネント
        self.projector: Optional[PointCloudProjector] = None
        self.triangulator: Optional[DelaunayTriangulator] = None
        self.lod_mesh_generator: Optional[Any] = None
        self.simplifier: Optional[MeshSimplifier] = None
        
        # 衝突検出コンポーネント
        self.spatial_index: Optional[SpatialIndex] = None
        self.collision_searcher: Optional[CollisionSearcher] = None
        self.collision_tester: Optional[SphereTriangleCollision] = None
        self.event_queue = CollisionEventQueue()
        
        # 音響生成コンポーネント
        self.audio_mapper: Optional[AudioMapper] = None
        self.audio_synthesizer: Optional[AudioSynthesizer] = None
        self.voice_manager: Optional[VoiceManager] = None
        self.audio_enabled = False
        
        # GPU加速コンポーネント
        self.gpu_distance_calc: Optional[Any] = None
        self.gpu_triangulator: Optional[Any] = None
        self.gpu_acceleration_enabled = False
        
        # 状態管理
        self.current_mesh = None
        self.current_collision_points = []
        self.current_tracked_hands = []
        self.frame_counter = 0
        self.last_mesh_update = -999
        self.force_mesh_update_requested = False
        
        # 音響クールダウン管理
        self.audio_cooldown_time = 0.15
        self.last_audio_trigger_time = {}
        
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
        
        # GPU統計
        self.gpu_stats = {
            'distance_calculations': 0,
            'triangulations': 0,
            'gpu_time_total_ms': 0.0,
            'cpu_fallbacks': 0
        }
        
        # 初期化完了フラグ
        self._initialized = False
    
    def initialize(self, camera: Optional[OrbbecCamera] = None) -> bool:
        """
        パイプライン初期化
        
        Args:
            camera: カメラオブジェクト（Noneの場合は内部で作成）
        
        Returns:
            初期化成功時True
        """
        try:
            print("統合パイプライン初期化中...")
            
            # カメラ初期化
            if camera:
                self.camera = camera
            else:
                self.camera = OrbbecCamera()
                if not self.camera.initialize():
                    print("Failed to initialize camera")
                    return False
            
            # カメラパラメータ取得後にPointCloudConverter初期化
            depth_intrinsics = self.camera.depth_intrinsics
            if depth_intrinsics:
                self.pointcloud_converter = PointCloudConverter(
                    depth_intrinsics=depth_intrinsics,
                    enable_voxel_downsampling=self.config.enable_voxel_downsampling,
                    voxel_size=self.config.voxel_size
                )
                
                # 3D投影初期化
                if self.config.enable_hand_detection:
                    try:
                        self.projector_3d = Hand3DProjector(
                            camera_intrinsics=depth_intrinsics,
                            interpolation_method=DepthInterpolationMethod.LINEAR,
                            depth_scale=1000.0,
                            min_confidence_3d=0.3
                        )
                    except Exception as e:
                        print(f"3D projector initialization failed: {e}")
                        # フォールバック: 最小パラメータで初期化
                        try:
                            self.projector_3d = Hand3DProjector(camera_intrinsics=depth_intrinsics)
                        except Exception as e2:
                            print(f"3D projector fallback also failed: {e2}")
                            self.projector_3d = None
            
            # 基本コンポーネント初期化
            if not self._initialize_basic_components():
                return False
            
            # メッシュ生成コンポーネント初期化
            if self.config.enable_mesh_generation:
                if not self._initialize_mesh_components():
                    return False
            
            # 衝突検出コンポーネント初期化
            if self.config.enable_collision_detection:
                if not self._initialize_collision_components():
                    return False
            
            # 音響生成コンポーネント初期化
            if self.config.enable_audio_synthesis:
                if not self._initialize_audio_components():
                    return False
            
            # GPU加速初期化
            if self.config.enable_gpu_acceleration and HAS_GPU_ACCELERATION:
                self._initialize_gpu_acceleration()
            
            self._initialized = True
            print("統合パイプライン初期化完了")
            return True
            
        except Exception as e:
            print(f"Pipeline initialization error: {e}")
            return False
    
    def _initialize_basic_components(self) -> bool:
        """基本コンポーネント初期化"""
        try:
            # 深度フィルタ
            if self.config.enable_filter:
                self.depth_filter = DepthFilter(
                    filter_types=[FilterType.TEMPORAL],
                    temporal_history_size=5,
                    temporal_alpha=0.3
                )
            
            # 点群コンバータ（カメラ初期化後に設定）
            self.pointcloud_converter = None
            
            # 手検出（2D）：ヘッドレスモード時は無効化
            if self.config.enable_hand_detection and not self.config.headless_mode:
                try:
                    self.hands_2d = MediaPipeHandsWrapper(
                        min_detection_confidence=self.config.min_detection_confidence,
                        min_tracking_confidence=0.5,
                        max_num_hands=2,
                        enable_roi_tracking=True,
                        tracker_type="KCF",
                        skip_interval=4,
                        roi_confidence_threshold=0.6,
                        max_tracking_age=15
                    )
                    print("✅ MediaPipe手検出が有効化されました（通常モード）")
                except Exception as e:
                    print(f"⚠️  MediaPipe手検出の初期化に失敗: {e}")
                    self.hands_2d = None
                    
            elif self.config.headless_mode:
                # ヘッドレスモード時はMediaPipeを無効化してパフォーマンスを向上
                self.hands_2d = None
                print("🖥️  ヘッドレスモード: MediaPipe手検出を無効化（FPS最適化）")
            else:
                self.hands_2d = None
                print("🔧 手検出が設定で無効化されています")
            
            # 3D投影（カメラ初期化が必要）
            self.projector_3d = None
            
            # 3Dトラッキング
            if self.config.enable_tracking:
                try:
                    self.tracker = Hand3DTracker(
                        max_assignment_distance=0.15,
                        max_lost_frames=20,
                        min_track_length=3,
                        dt=1.0/30.0
                    )
                except Exception as e:
                    print(f"3D tracker initialization failed: {e}")
                    # フォールバック: デフォルトパラメータで初期化
                    try:
                        self.tracker = Hand3DTracker()
                    except Exception as e2:
                        print(f"3D tracker fallback also failed: {e2}")
                        self.tracker = None
            
            return True
            
        except Exception as e:
            print(f"Basic components initialization error: {e}")
            return False
    
    def _initialize_mesh_components(self) -> bool:
        """メッシュ生成コンポーネント初期化（メモリ制限付き）"""
        try:
            # 地形メッシュ生成
            self.projector = PointCloudProjector(
                resolution=self.config.mesh_resolution,
                method=ProjectionMethod.MEDIAN_HEIGHT,
                fill_holes=True
            )
            
            # LODメッシュ生成器（メモリ制限付き）
            if HAS_LOD_MESH:
                try:
                    self.lod_mesh_generator = create_lod_mesh_generator(
                        high_radius=0.15,  # 0.20 -> 0.15 (メモリ使用量削減)
                        medium_radius=0.40,  # 0.50 -> 0.40
                    enable_gpu=self.config.enable_gpu_acceleration
                )
                except Exception as e:
                    print(f"LOD mesh generator initialization failed: {e}")
                    self.lod_mesh_generator = None
            
            # Delaunay三角分割（基本パラメータのみ）
            self.triangulator = DelaunayTriangulator()
            
            # メッシュ簡略化（基本パラメータのみ）
            self.simplifier = MeshSimplifier()
            
            return True
            
        except Exception as e:
            print(f"Mesh components initialization error: {e}")
            return False
    
    def _initialize_collision_components(self) -> bool:
        """衝突検出コンポーネント初期化"""
        try:
            # 空間インデックス（メッシュが必要な場合は後で初期化）
            self.spatial_index = None
            
            # 衝突検索（後で初期化）
            self.collision_searcher = None
            
            # 球-三角形衝突テスト（後で初期化）
            self.collision_tester = None
            
            return True
            
        except Exception as e:
            print(f"Collision components initialization error: {e}")
            return False
    
    def _initialize_audio_components(self) -> bool:
        """音響生成コンポーネント初期化"""
        try:
            # 音響マッピング
            self.audio_mapper = AudioMapper(
                scale=self.config.audio_scale,
                default_instrument=self.config.audio_instrument
            )
            
            # 音響シンセサイザー
            try:
                self.audio_synthesizer = create_audio_synthesizer()
                if not self.audio_synthesizer.initialize():
                    print("Warning: Audio synthesizer initialization failed")
                    return True  # 音響は必須ではない
            except (ImportError, AttributeError):
                # フォールバック: 直接AudioSynthesizerを使用
                self.audio_synthesizer = AudioSynthesizer()
                if not self.audio_synthesizer.initialize():
                    print("Warning: Audio synthesizer initialization failed")
                    return True  # 音響は必須ではない
            
            # ボイスマネージャー
            try:
                self.voice_manager = create_voice_manager(
                    synthesizer=self.audio_synthesizer,
                    max_polyphony=self.config.audio_polyphony,
                    steal_strategy=StealStrategy.OLDEST
                )
            except (ImportError, AttributeError):
                # フォールバック: 直接VoiceManagerを使用
                self.voice_manager = VoiceManager(
                    synthesizer=self.audio_synthesizer,
                    max_polyphony=self.config.audio_polyphony
                )
            
            self.audio_enabled = True
            return True
            
        except Exception as e:
            print(f"Audio components initialization error: {e}")
            return True  # 音響は必須ではない
    
    def _initialize_gpu_acceleration(self) -> None:
        """GPU加速初期化"""
        try:
            self.gpu_distance_calc = create_gpu_distance_calculator(
                use_gpu=True,
                batch_size=10000,
                memory_limit_ratio=0.8
            )
            
            self.gpu_triangulator = create_gpu_triangulator(
                use_gpu=True,
                quality_threshold=0.2,
                enable_caching=True
            )
            
            self.gpu_acceleration_enabled = True
            print("GPU acceleration initialized successfully")
            
        except Exception as e:
            print(f"GPU acceleration initialization failed: {e}")
            print("Falling back to CPU-only processing")
    
    def process_frame(self) -> Optional[PipelineResults]:
        """
        フレーム処理実行
        
        Returns:
            処理結果
        """
        if not self._initialized:
            return None
        
        start_time = time.perf_counter()
        
        try:
            # フレーム取得
            frame_data = self.camera.get_frame()
            if frame_data is None:
                return None
            
            # 基本処理
            results = self._process_basic_pipeline(frame_data)
            if not results:
                return None
            
            # メッシュ生成
            if self.config.enable_mesh_generation:
                self._process_mesh_generation(results)
            
            # 衝突検出
            if self.config.enable_collision_detection:
                self._process_collision_detection(results)
            
            # 音響合成
            if self.config.enable_audio_synthesis:
                self._process_audio_synthesis(results)
            
            # パフォーマンス統計更新
            total_time = time.perf_counter() - start_time
            self._update_performance_stats(total_time)
            results.performance_stats = self.perf_stats.copy()
            
            self.frame_counter += 1
            return results
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None
    
    def _process_basic_pipeline(self, frame_data: FrameData) -> Optional[PipelineResults]:
        """基本パイプライン処理"""
        try:
            results = PipelineResults(frame_data=frame_data)
            
            # 深度フィルタ適用
            filtered_depth_image = None
            if self.depth_filter and hasattr(frame_data, 'depth_image') and frame_data.depth_image is not None:
                filtered_depth_image = self.depth_filter.apply_filter(frame_data.depth_image)
            else:
                filtered_depth_image = frame_data.depth_image
            
            # 点群変換
            if self.pointcloud_converter and filtered_depth_image is not None:
                points = self.pointcloud_converter.depth_to_pointcloud(filtered_depth_image)
                frame_data.points = points
            
            # 手検出（2D）
            if self.hands_2d and frame_data.color_frame is not None:
                # カラー抽出エラーカウンターチェック
                if self.color_extraction_disabled:
                    # エラー上限に達している場合はスキップ
                    results.hands_2d = []
                else:
                    # オリジナルコードと同じカラー画像処理
                    color_image = self._extract_color_image_for_mediapipe(frame_data)
                    if color_image is not None:
                        # 正常に取得できた場合はエラーカウンターをリセット
                        self.color_extraction_errors = 0
                        hands_2d = self.hands_2d.detect_hands(color_image)
                        results.hands_2d = hands_2d
                    else:
                        # エラーカウンターを増加
                        self.color_extraction_errors += 1
                        results.hands_2d = []
                        
                        # エラー上限チェック
                        if self.color_extraction_errors >= self.max_color_extraction_errors:
                            print(f"Color extraction failed {self.max_color_extraction_errors} times, disabling MediaPipe processing")
                            self.color_extraction_disabled = True
            else:
                results.hands_2d = []
            
            # 3D投影
            if self.projector_3d and results.hands_2d and filtered_depth_image is not None:
                hands_3d = []
                for hand_2d in results.hands_2d:
                    hand_3d = self.projector_3d.project_to_3d(
                        hand_2d, filtered_depth_image, self.camera.depth_intrinsics
                    )
                    if hand_3d:
                        hands_3d.append(hand_3d)
                results.hands_3d = hands_3d
            
            # 3Dトラッキング
            if self.tracker and results.hands_3d:
                tracked_hands = self.tracker.update(results.hands_3d)
                results.tracked_hands = tracked_hands
                self.current_tracked_hands = tracked_hands
            
            return results
            
        except Exception as e:
            print(f"Basic pipeline processing error: {e}")
            return None
    
    def _process_mesh_generation(self, results: PipelineResults) -> None:
        """メッシュ生成処理（メモリ監視付き）"""
        try:
            mesh_start = time.perf_counter()
            
            # メモリ使用量チェック
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:  # メモリ使用率85%超過で警告
                print(f"Warning: High memory usage ({memory_percent:.1f}%), skipping mesh generation")
                results.mesh = self.current_mesh  # 既存メッシュを再利用
                return
            
            # メッシュ更新判定
            should_update_mesh = (
                self.force_mesh_update_requested or
                (self.frame_counter - self.last_mesh_update >= self.config.mesh_update_interval) or
                (len(self.current_tracked_hands) == 0 and 
                 self.frame_counter - self.last_mesh_update >= self.config.max_mesh_skip_frames)
            )
            
            if should_update_mesh and results.frame_data and results.frame_data.points is not None:
                mesh = None
                points = results.frame_data.points
                
                # 点数制限（メモリ保護）
                if len(points) > 100000:  # 10万点を超える場合はダウンサンプリング
                    import random
                    indices = random.sample(range(len(points)), 100000)
                    points = points[indices]
                    print(f"Point cloud downsampled to {len(points)} points for memory protection")
                
                # LODメッシュ生成を優先
                if self.lod_mesh_generator:
                    try:
                        hand_positions = [h.palm_center for h in self.current_tracked_hands if h.palm_center is not None]
                        mesh = self.lod_mesh_generator.generate_mesh(
                            points,
                            hand_positions=hand_positions
                        )
                    except Exception as e:
                        print(f"LOD mesh generation failed: {e}")
                        # GPU メモリ不足の場合は明示的にクリア
                        if "memory" in str(e).lower() or "alloc" in str(e).lower():
                            print("GPU memory issue detected, forcing cleanup")
                            import gc
                            gc.collect()
                            # CUDA キャッシュクリア（利用可能な場合）
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except ImportError:
                                pass
                
                # フォールバック: 従来の三角分割
                if mesh is None and self.triangulator:
                    try:
                        # 地形投影
                        if self.projector:
                            projected_points = self.projector.project_points(points)
                            # 投影点数も制限
                            if len(projected_points) > 50000:
                                import random
                                indices = random.sample(range(len(projected_points)), 50000)
                                projected_points = projected_points[indices]
                            
                            mesh = self.triangulator.triangulate(projected_points)
                            
                            # メッシュ簡略化
                            if mesh and self.simplifier:
                                mesh = self.simplifier.simplify(mesh)
                    except Exception as e:
                        print(f"Fallback mesh generation failed: {e}")
                        # メモリ関連エラーの場合は強制クリーンアップ
                        if "memory" in str(e).lower() or "alloc" in str(e).lower():
                            print("Memory allocation failed, forcing cleanup")
                            import gc
                            gc.collect()
                
                if mesh:
                    # メッシュサイズチェック（メモリ保護）
                    if hasattr(mesh, 'vertices') and len(mesh.vertices) > 50000:
                        print(f"Warning: Large mesh ({len(mesh.vertices)} vertices), may cause memory issues")
                    
                    self.current_mesh = mesh
                    results.mesh = mesh
                    self.last_mesh_update = self.frame_counter
                    self.force_mesh_update_requested = False
                else:
                    # メッシュ生成失敗時は既存メッシュを保持
                    results.mesh = self.current_mesh
            else:
                # 既存メッシュを使用
                results.mesh = self.current_mesh
            
            mesh_time = time.perf_counter() - mesh_start
            self.perf_stats['mesh_generation_time'] = mesh_time
            
        except Exception as e:
            print(f"Mesh generation processing error: {e}")
            # エラー時も既存メッシュを保持
            results.mesh = self.current_mesh
            # メモリ関連エラーの場合は緊急クリーンアップ
            if "memory" in str(e).lower() or "alloc" in str(e).lower():
                print("Critical memory error, forcing garbage collection")
                import gc
                gc.collect()
    
    def _process_collision_detection(self, results: PipelineResults) -> None:
        """衝突検出処理"""
        try:
            collision_start = time.perf_counter()
            
            collision_events = []
            
            if results.mesh:
                
                # 空間インデックス初期化（初回のみ）
                if self.spatial_index is None:
                    try:
                        from ..mesh.index import SpatialIndex, IndexType
                        self.spatial_index = SpatialIndex(mesh=results.mesh, index_type=IndexType.OCTREE)
                    except Exception as e:
                        print(f"SpatialIndex initialization failed: {e}")
                        self.spatial_index = None
                
                # 衝突検索器初期化（初回のみ）
                if self.collision_searcher is None and self.spatial_index is not None:
                    try:
                        from ..collision.search import CollisionSearcher
                        self.collision_searcher = CollisionSearcher(
                            spatial_index=self.spatial_index,
                            default_radius=self.config.sphere_radius
                        )
                    except Exception as e:
                        print(f"CollisionSearcher initialization failed: {e}")
                        self.collision_searcher = None
                
                # 球-三角形衝突テスト初期化（初回のみ）
                if self.collision_tester is None:
                    try:
                        from ..collision.sphere_tri import SphereTriangleCollision
                        self.collision_tester = SphereTriangleCollision(
                            mesh=results.mesh
                        )
                    except Exception as e:
                        print(f"SphereTriangleCollision initialization failed: {e}")
                        self.collision_tester = None
                
                # 空間インデックス更新
                if self.spatial_index:
                    self.spatial_index.build_index(results.mesh)
                
                # 各手について衝突検出（コンポーネントが利用可能な場合のみ）
                if self.collision_searcher and self.collision_tester and self.spatial_index:
                    for hand in results.tracked_hands:
                        if hand.palm_center is not None:
                            try:
                                # 候補三角形検索
                                candidates = self.collision_searcher.search_candidates(
                                    hand.palm_center, self.spatial_index
                                )
                                
                                # 詳細衝突判定
                                for triangle in candidates:
                                    if self.collision_tester.test_collision(hand.palm_center, triangle):
                                        # 簡易版CollisionEventを作成
                                        collision_event = self._create_simple_collision_event(hand, triangle)
                                        collision_events.append(collision_event)
                                        self.event_queue.add_event(collision_event)
                            except Exception as e:
                                print(f"Collision detection error for hand {hand.hand_id}: {e}")
                                continue
            
            results.collision_events = collision_events
            self.current_collision_points = collision_events
            
            collision_time = time.perf_counter() - collision_start
            self.perf_stats['collision_detection_time'] = collision_time
            self.perf_stats['collision_events_count'] = len(collision_events)
            
        except Exception as e:
            print(f"Collision detection processing error: {e}")
    
    def _process_audio_synthesis(self, results: PipelineResults) -> None:
        """音響合成処理"""
        try:
            audio_start = time.perf_counter()
            
            notes_played = 0
            
            if (self.audio_enabled and self.audio_mapper and 
                self.audio_synthesizer and self.voice_manager):
                
                current_time = time.time()
                
                for event in results.collision_events:
                    hand_id = event.get('hand_id', 'unknown')
                    
                    # クールダウンチェック
                    last_trigger = self.last_audio_trigger_time.get(hand_id, 0)
                    if current_time - last_trigger >= self.audio_cooldown_time:
                        # 音響マッピング
                        note_info = self.audio_mapper.map_collision_event(event)
                        
                        if note_info:
                            # 音声合成（統一されたAPI使用）
                            try:
                                # allocate_and_play関数を使用
                                success = allocate_and_play(
                                    voice_manager=self.voice_manager,
                                    note_info=note_info
                                )
                                if success:
                                    notes_played += 1
                                    self.last_audio_trigger_time[hand_id] = current_time
                            except (ImportError, AttributeError):
                                # フォールバック: 従来のAPI
                                voice = self.voice_manager.allocate_voice()
                                if voice is not None:
                                    self.audio_synthesizer.play_note(voice, note_info)
                                    notes_played += 1
                                    self.last_audio_trigger_time[hand_id] = current_time
            
            audio_time = time.perf_counter() - audio_start
            self.perf_stats['audio_synthesis_time'] = audio_time
            self.perf_stats['audio_notes_played'] = notes_played
            
        except Exception as e:
            print(f"Audio synthesis processing error: {e}")
    
    def _update_performance_stats(self, total_time: float) -> None:
        """パフォーマンス統計更新"""
        self.perf_stats['frame_count'] = self.frame_counter
        self.perf_stats['total_pipeline_time'] = total_time
    
    def force_mesh_update(self) -> None:
        """メッシュ強制更新要求"""
        self.force_mesh_update_requested = True
    
    def update_config(self, config: HandledPipelineConfig) -> None:
        """設定更新"""
        self.config = config
        
        # 動的設定更新
        if self.collision_tester:
            self.collision_tester.radius = config.sphere_radius
        
        if self.audio_mapper:
            self.audio_mapper.scale = config.audio_scale
            self.audio_mapper.default_instrument = config.audio_instrument
        
        if self.voice_manager:
            self.voice_manager.max_polyphony = config.audio_polyphony
            # master_volumeは音響シンセサイザー側で管理
            if self.audio_synthesizer:
                self.audio_synthesizer.set_master_volume(config.audio_master_volume)
    
    def _extract_color_image_for_mediapipe(self, frame_data):
        """MediaPipe用カラー画像抽出（MJPG対応・C-contiguous完全対応版）"""
        try:
            # エラー制限チェック（サーキットブレーカーパターン）
            if self.color_extraction_disabled:
                return None
                
            if frame_data is None:
                return None
                
            if not hasattr(frame_data, 'color_frame') or frame_data.color_frame is None:
                return None
                
            if not hasattr(self, 'camera') or self.camera is None:
                return None
                
            if not hasattr(self.camera, 'has_color') or not self.camera.has_color:
                return None
            
            import cv2
            from ..types import OBFormat
            
            # カラーフレームから安全にデータを取得
            try:
                color_frame = frame_data.color_frame
                
                # フレーム形式を確認
                frame_format = getattr(color_frame, 'get_format', lambda: OBFormat.RGB)()
                

                
                if str(frame_format) == "OBFormat.MJPG" or str(frame_format) == "MJPG":
                    # MJPG形式の場合：JPEGデコードが必要
                    try:
                        # get_data()でJPEGバイナリデータを取得
                        jpeg_data = color_frame.get_data()
                        if jpeg_data is None:
                            if not hasattr(self, '_mjpg_no_data_error_shown'):
                                print("MJPG: No data from color frame")
                                self._mjpg_no_data_error_shown = True
                            return None
                        
                        # データサイズを確認
                        data_size = 0
                        jpeg_bytes = None
                        
                        # バイナリデータをnumpy配列に変換（複数の方法で試行）
                        try:
                            if hasattr(jpeg_data, 'tobytes'):
                                jpeg_bytes = jpeg_data.tobytes()
                            elif hasattr(jpeg_data, '__bytes__'):
                                jpeg_bytes = bytes(jpeg_data)
                            elif hasattr(jpeg_data, '__array__'):
                                jpeg_array_raw = np.array(jpeg_data, dtype=np.uint8)
                                jpeg_bytes = jpeg_array_raw.tobytes()
                            else:
                                # 最後の手段：直接bytes()を試行
                                jpeg_bytes = bytes(jpeg_data)
                            
                            data_size = len(jpeg_bytes)
                            
                        except Exception as data_convert_error:
                            if not hasattr(self, '_mjpg_data_convert_error_shown'):
                                print(f"MJPG data conversion failed: {data_convert_error}")
                                self._mjpg_data_convert_error_shown = True
                            return None
                        
                        # データサイズが妥当かチェック
                        if data_size < 100:  # JPEGは最低でも100バイト以上必要
                            if not hasattr(self, '_mjpg_small_data_error_shown'):
                                print(f"MJPG data too small: {data_size} bytes")
                                self._mjpg_small_data_error_shown = True
                            return None
                        
                        # OpenCVでJPEGデコード
                        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        bgr_image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
                        
                        if bgr_image is None:
                            if not hasattr(self, '_mjpg_opencv_decode_error_shown'):
                                print(f"OpenCV MJPG decode failed (data size: {data_size} bytes)")
                                self._mjpg_opencv_decode_error_shown = True
                            return None
                        
                        # 画像サイズを確認
                        if bgr_image.shape[0] == 0 or bgr_image.shape[1] == 0:
                            if not hasattr(self, '_mjpg_zero_size_error_shown'):
                                print(f"MJPG decoded to zero size image: {bgr_image.shape}")
                                self._mjpg_zero_size_error_shown = True
                            return None
                        
                        # BGRからRGBに変換
                        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                        
                        # C-contiguous配列として確実に作成
                        rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
                        
                        # 成功メッセージは初回のみ表示
                        if not hasattr(self, '_mjpg_success_shown'):
                            print(f"✅ MJPG processing successful: {rgb_image.shape} ({data_size} bytes)")
                            self._mjpg_success_shown = True
                        
                        return rgb_image
                        
                    except Exception as e:
                        # エラーメッセージは初回のみ表示
                        if not hasattr(self, '_mjpg_processing_error_shown'):
                            print(f"MJPG processing error: {e}")
                            self._mjpg_processing_error_shown = True
                        return None
                
                elif str(frame_format) in ["OBFormat.RGB", "RGB", "OBFormat.BGR", "BGR"]:
                    # RGB/BGR形式の場合：直接データを取得
                    try:
                        # get_data()で生データを取得
                        raw_data = color_frame.get_data()
                        if raw_data is None:
                            return None
                        
                        # カメラの解像度情報を取得
                        width = getattr(color_frame, 'get_width', lambda: 640)()
                        height = getattr(color_frame, 'get_height', lambda: 480)()
                        
                        # numpy配列に変換（C-contiguous保証）
                        if hasattr(raw_data, 'tobytes'):
                            data_bytes = raw_data.tobytes()
                        else:
                            data_bytes = bytes(raw_data)
                        
                        # RGB/BGR配列として再構築
                        color_array = np.frombuffer(data_bytes, dtype=np.uint8)
                        color_array = color_array.reshape((height, width, 3))
                        
                        # C-contiguous配列として確実に作成
                        color_array = np.ascontiguousarray(color_array, dtype=np.uint8)
                        
                        # BGR→RGB変換が必要な場合
                        if str(frame_format) in ["OBFormat.BGR", "BGR"]:
                            color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)
                            color_array = np.ascontiguousarray(color_array, dtype=np.uint8)
                        
                        return color_array
                        
                    except Exception as e:
                        # エラーメッセージは初回のみ表示
                        if not hasattr(self, '_rgb_bgr_processing_error_shown'):
                            print(f"RGB/BGR processing error: {e}")
                            self._rgb_bgr_processing_error_shown = True
                        return None
                
                else:
                    # サポートされていない形式（初回のみ表示）
                    if not hasattr(self, '_unsupported_format_error_shown'):
                        print(f"Unsupported color format: {frame_format}")
                        self._unsupported_format_error_shown = True
                    return None
                    
            except Exception as e:
                # エラーカウンターの更新
                self.color_extraction_errors += 1
                
                # エラーメッセージは初回のみ表示
                if not hasattr(self, '_color_frame_access_error_shown'):
                    print(f"Color frame access error: {e}")
                    self._color_frame_access_error_shown = True
                
                # エラーが多すぎる場合はサーキットブレーカーを作動
                if self.color_extraction_errors >= self.max_color_extraction_errors:
                    if not hasattr(self, '_circuit_breaker_triggered'):
                        print(f"⚠️  Color extraction circuit breaker triggered after {self.max_color_extraction_errors} errors")
                        self._circuit_breaker_triggered = True
                    self.color_extraction_disabled = True
                
                return None
                
        except Exception as e:
            # エラーカウンターの更新
            self.color_extraction_errors += 1
            
            # エラーメッセージは初回のみ表示
            if not hasattr(self, '_color_extraction_error_shown'):
                print(f"Color image extraction error: {e}")
                self._color_extraction_error_shown = True
            
            # エラーが多すぎる場合はサーキットブレーカーを作動
            if self.color_extraction_errors >= self.max_color_extraction_errors:
                if not hasattr(self, '_circuit_breaker_triggered'):
                    print(f"⚠️  Color extraction circuit breaker triggered after {self.max_color_extraction_errors} errors")
                    self._circuit_breaker_triggered = True
                self.color_extraction_disabled = True
            
            return None
    
    def _create_simple_collision_event(self, hand, triangle):
        """簡易版CollisionEventを作成"""
        from ..collision.events import CollisionEvent, CollisionIntensity, EventType
        from ..types import CollisionType
        
        # 基本的な衝突イベントを作成（必要最小限のパラメータ）
        return CollisionEvent(
            event_id=f"collision_{int(time.time() * 1000000)}",
            event_type=EventType.COLLISION_START,
            timestamp=time.time(),
            duration_ms=0.0,
            
            contact_position=hand.palm_center.copy(),
            hand_position=hand.palm_center.copy(),
            surface_normal=np.array([0.0, 1.0, 0.0]),  # デフォルト法線
            
            intensity=CollisionIntensity.MEDIUM,
            velocity=hand.speed if hasattr(hand, 'speed') else 0.1,
            penetration_depth=0.01,  # デフォルト値
            contact_area=0.001,  # デフォルト値
            
            pitch_hint=max(0.0, min(1.0, hand.palm_center[1])),  # Y座標を正規化
            timbre_hint=0.5,  # デフォルト値
            spatial_position=np.array([hand.palm_center[0], 0.0, hand.palm_center[2]]),
            
            triangle_index=0,  # デフォルト値
            hand_id=hand.hand_id,
            collision_type=CollisionType.FACE_COLLISION if hasattr(CollisionType, 'FACE_COLLISION') else None
        )
    
    def cleanup(self) -> None:
        """クリーンアップ処理"""
        try:
            if self.audio_synthesizer:
                self.audio_synthesizer.cleanup()
            
            if self.camera:
                self.camera.cleanup()
            
            self._initialized = False
            
        except Exception as e:
            print(f"Pipeline cleanup error: {e}")