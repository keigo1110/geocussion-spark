#!/usr/bin/env python3
"""
パイプライン制御レイヤー（Clean Architecture適用）
責務の分離: UI < コントローラー < ドメインロジック < データアクセス
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# リソース管理
from ..resource_manager import ManagedResource, ResourceManager

# 各フェーズのドメインロジック
from ..input.stream import OrbbecCamera, FrameData
from ..input.pointcloud import PointCloudConverter  
from ..input.depth_filter import DepthFilter, FilterType
from ..detection.hands2d import MediaPipeHandsWrapper
from ..detection.hands3d import Hand3DProjector, DepthInterpolationMethod
from ..detection.tracker import Hand3DTracker, TrackedHand
from ..mesh.projection import PointCloudProjector, ProjectionMethod
from ..mesh.delaunay import DelaunayTriangulator
from ..mesh.simplify import MeshSimplifier
from ..collision.search import CollisionSearcher
from ..collision.sphere_tri import SphereTriangleCollision
from ..collision.events import CollisionEventQueue
from ..mesh.index import SpatialIndex, IndexType
from ..sound.mapping import AudioMapper, ScaleType, InstrumentType
from ..sound.synth import AudioSynthesizer
from ..sound.voice_mgr import VoiceManager


@dataclass
class PipelineConfiguration:
    """パイプライン全体の設定"""
    # 入力設定
    enable_filter: bool = True
    enable_hand_detection: bool = True
    enable_tracking: bool = True
    min_detection_confidence: float = 0.7
    use_gpu_mediapipe: bool = False
    
    # メッシュ生成設定
    enable_mesh_generation: bool = True
    mesh_update_interval: int = 5
    max_mesh_skip_frames: int = 60
    mesh_resolution: float = 0.01  # 1cm
    mesh_quality_threshold: float = 0.3
    mesh_reduction: float = 0.7  # 70%削減
    
    # 衝突検出設定
    enable_collision_detection: bool = True
    enable_collision_visualization: bool = True
    sphere_radius: float = 0.05  # 5cm
    
    # 音響合成設定
    enable_audio_synthesis: bool = False
    audio_scale: ScaleType = ScaleType.PENTATONIC
    audio_instrument: InstrumentType = InstrumentType.MARIMBA
    audio_polyphony: int = 4
    audio_master_volume: float = 0.7
    audio_cooldown_time: float = 0.15  # 150ms


@dataclass
class PipelineState:
    """パイプライン実行時状態"""
    frame_count: int = 0
    frame_counter: int = 0
    last_mesh_update: int = -999
    force_mesh_update_requested: bool = False
    
    # 現在のデータ
    current_mesh: Optional[Any] = None
    current_collision_points: List = field(default_factory=list)
    current_tracked_hands: List[TrackedHand] = field(default_factory=list)
    
    # 音響クールダウン管理
    last_audio_trigger_time: Dict[str, float] = field(default_factory=dict)
    
    # パフォーマンス統計
    perf_stats: Dict[str, Any] = field(default_factory=lambda: {
        'frame_count': 0,
        'mesh_generation_time': 0.0,
        'collision_detection_time': 0.0,
        'audio_synthesis_time': 0.0,
        'collision_events_count': 0,
        'audio_notes_played': 0,
        'total_pipeline_time': 0.0
    })


class IPipelineObserver(ABC):
    """パイプラインイベント観察者インターフェース"""
    
    @abstractmethod
    def on_frame_processed(self, frame_data: FrameData, results: Dict[str, Any]) -> None:
        """フレーム処理完了時のコールバック"""
        pass
    
    @abstractmethod
    def on_collision_detected(self, collision_events: List) -> None:
        """衝突検出時のコールバック"""
        pass
    
    @abstractmethod
    def on_performance_update(self, stats: Dict[str, Any]) -> None:
        """パフォーマンス統計更新時のコールバック"""
        pass


class GeocussionPipelineController(ManagedResource):
    """
    Geocussion統合パイプラインコントローラー
    Clean Architecture適用: 責務分離されたドメインロジック制御
    """
    
    def __init__(self, config: PipelineConfiguration):
        """
        初期化
        
        Args:
            config: パイプライン設定
        """
        super().__init__(resource_id="pipeline_controller")
        self.config = config
        self.state = PipelineState()
        self.observers: List[IPipelineObserver] = []
        
        # コンポーネント（ManagedResourceとして管理）
        self.camera: Optional[OrbbecCamera] = None
        self.pointcloud_converter: Optional[PointCloudConverter] = None
        self.depth_filter: Optional[DepthFilter] = None
        
        # 手検出コンポーネント
        self.hands_2d: Optional[MediaPipeHandsWrapper] = None
        self.projector_3d: Optional[Hand3DProjector] = None
        self.tracker: Optional[Hand3DTracker] = None
        
        # メッシュ生成コンポーネント
        self.projector: Optional[PointCloudProjector] = None
        self.triangulator: Optional[DelaunayTriangulator] = None
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
        
        self._components_initialized = False
    
    def initialize(self) -> bool:
        """
        パイプライン初期化
        
        Returns:
            成功した場合True
        """
        try:
            # 入力システム初期化
            if not self._initialize_input_system():
                return False
            
            # 手検出システム初期化
            if self.config.enable_hand_detection:
                if not self._initialize_hand_detection_system():
                    print("Warning: Hand detection initialization failed")
                    self.config.enable_hand_detection = False
            
            # メッシュ生成システム初期化
            if self.config.enable_mesh_generation:
                self._initialize_mesh_generation_system()
            
            # 衝突検出システム初期化
            if self.config.enable_collision_detection:
                self._initialize_collision_detection_system()
            
            # 音響生成システム初期化
            if self.config.enable_audio_synthesis:
                self._initialize_audio_synthesis_system()
            
            self._components_initialized = True
            print("Pipeline controller initialized successfully")
            return True
            
        except Exception as e:
            print(f"Pipeline initialization error: {e}")
            return False
    
    def _initialize_input_system(self) -> bool:
        """入力システム初期化"""
        # カメラ初期化
        self.camera = OrbbecCamera(enable_color=True)
        if not self.camera.initialize() or not self.camera.start():
            print("Failed to initialize camera")
            return False
        
        # 点群コンバーター初期化
        if self.camera.depth_intrinsics:
            self.pointcloud_converter = PointCloudConverter(self.camera.depth_intrinsics)
        else:
            print("No depth intrinsics available")
            return False
        
        # 深度フィルタ初期化
        if self.config.enable_filter:
            self.depth_filter = DepthFilter(
                filter_types=[FilterType.COMBINED],
                temporal_alpha=0.3,
                bilateral_sigma_color=50.0
            )
        
        return True
    
    def _initialize_hand_detection_system(self) -> bool:
        """手検出システム初期化"""
        try:
            # 2D手検出初期化
            self.hands_2d = MediaPipeHandsWrapper(
                use_gpu=self.config.use_gpu_mediapipe,
                max_num_hands=2,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=0.5
            )
            
            # 3D投影初期化
            if self.camera.depth_intrinsics:
                self.projector_3d = Hand3DProjector(
                    camera_intrinsics=self.camera.depth_intrinsics,
                    interpolation_method=DepthInterpolationMethod.NEAREST,
                    min_confidence_3d=0.3
                )
            else:
                return False
            
            # トラッカー初期化
            if self.config.enable_tracking:
                self.tracker = Hand3DTracker(
                    max_lost_frames=10,
                    min_track_length=5,
                    max_assignment_distance=0.3
                )
            
            return True
            
        except Exception as e:
            print(f"Hand detection initialization error: {e}")
            return False
    
    def _initialize_mesh_generation_system(self) -> None:
        """メッシュ生成システム初期化"""
        self.projector = PointCloudProjector(
            resolution=self.config.mesh_resolution,
            method=ProjectionMethod.MEDIAN_HEIGHT,
            fill_holes=True
        )
        
        self.triangulator = DelaunayTriangulator(
            adaptive_sampling=True,
            boundary_points=True,
            quality_threshold=self.config.mesh_quality_threshold
        )
        
        self.simplifier = MeshSimplifier(
            target_reduction=self.config.mesh_reduction,
            preserve_boundary=True
        )
    
    def _initialize_collision_detection_system(self) -> None:
        """衝突検出システム初期化"""
        # CollisionSearcherとSphereTriangleCollisionはメッシュ生成後に作成
        self.collision_searcher = None
        self.collision_tester = None
    
    def _initialize_audio_synthesis_system(self) -> None:
        """音響生成システム初期化"""
        try:
            # AudioSynthesizer
            self.audio_synthesizer = AudioSynthesizer(
                sample_rate=44100,
                buffer_size=256,
                channels=2,
                default_scale=self.config.audio_scale,
                default_instrument=self.config.audio_instrument,
                polyphony=self.config.audio_polyphony,
                master_volume=self.config.audio_master_volume
            )
            
            # AudioMapper
            self.audio_mapper = AudioMapper(
                scale=self.config.audio_scale,
                instrument=self.config.audio_instrument
            )
            
            # VoiceManager  
            self.voice_manager = VoiceManager(
                synthesizer=self.audio_synthesizer,
                mapper=self.audio_mapper
            )
            
            print("Audio synthesis system initialized")
            
        except Exception as e:
            print(f"Audio initialization error: {e}")
    
    def process_frame(self) -> Dict[str, Any]:
        """
        単一フレーム処理（メインパイプライン実行）
        
        Returns:
            処理結果辞書
        """
        if not self._components_initialized:
            return {}
        
        pipeline_start = time.perf_counter()
        
        # フレームデータ取得
        frame_data = self.camera.get_frame()
        if frame_data is None:
            return {}
        
        results = {
            'frame_data': frame_data,
            'hands_2d': [],
            'hands_3d': [],
            'tracked_hands': [],
            'mesh': None,
            'collision_events': [],
            'audio_events': []
        }
        
        # 手検出フェーズ
        if self.config.enable_hand_detection:
            hands_2d, hands_3d, tracked_hands = self._process_hand_detection(frame_data)
            results.update({
                'hands_2d': hands_2d,
                'hands_3d': hands_3d, 
                'tracked_hands': tracked_hands
            })
            self.state.current_tracked_hands = tracked_hands
        
        # メッシュ生成フェーズ
        if self.config.enable_mesh_generation:
            mesh = self._process_mesh_generation(frame_data)
            if mesh:
                results['mesh'] = mesh
                self.state.current_mesh = mesh
        
        # 衝突検出フェーズ
        if self.config.enable_collision_detection and self.state.current_mesh:
            collision_events = self._process_collision_detection(self.state.current_tracked_hands)
            results['collision_events'] = collision_events
            self.state.current_collision_points = collision_events
        
        # 音響生成フェーズ
        if self.config.enable_audio_synthesis and results['collision_events']:
            audio_events = self._process_audio_generation(results['collision_events'])
            results['audio_events'] = audio_events
        
        # パフォーマンス統計更新
        total_time = time.perf_counter() - pipeline_start
        self._update_performance_stats(total_time)
        
        # 観察者に通知
        self._notify_observers(frame_data, results)
        
        return results
    
    def _process_hand_detection(self, frame_data: FrameData) -> tuple:
        """手検出処理"""
        hands_2d = []
        hands_3d = []
        tracked_hands = []
        
        if self.hands_2d and frame_data.color is not None:
            # 2D手検出
            hands_2d = self.hands_2d.process_frame(frame_data.color)
            
            # 3D投影
            if self.projector_3d and frame_data.depth is not None:
                hands_3d = self.projector_3d.project_hands(hands_2d, frame_data.depth)
                
                # トラッキング
                if self.tracker:
                    tracked_hands = self.tracker.update(hands_3d)
        
        return hands_2d, hands_3d, tracked_hands
    
    def _process_mesh_generation(self, frame_data: FrameData) -> Optional[Any]:
        """メッシュ生成処理"""
        # メッシュ更新条件判定
        hands_detected = len(self.state.current_tracked_hands) > 0
        frames_since_update = self.state.frame_counter - self.state.last_mesh_update
        
        should_update = (
            self.state.force_mesh_update_requested or
            (not hands_detected and frames_since_update >= self.config.mesh_update_interval) or
            frames_since_update >= self.config.max_mesh_skip_frames
        )
        
        if not should_update:
            return None
        
        mesh_start = time.perf_counter()
        
        try:
            # 点群変換
            if self.pointcloud_converter and frame_data.depth is not None:
                points_3d = self.pointcloud_converter.convert(
                    frame_data.depth, frame_data.color
                )
                
                # 深度フィルタ適用
                if self.depth_filter:
                    filtered_depth = self.depth_filter.apply(frame_data.depth)
                    points_3d = self.pointcloud_converter.convert(
                        filtered_depth, frame_data.color
                    )
                
                # メッシュ生成パイプライン
                heightmap = self.projector.project(points_3d)
                mesh = self.triangulator.triangulate(heightmap)
                simplified_mesh = self.simplifier.simplify(mesh)
                
                # 状態更新
                self.state.last_mesh_update = self.state.frame_counter
                self.state.force_mesh_update_requested = False
                
                # パフォーマンス統計更新
                mesh_time = time.perf_counter() - mesh_start
                self.state.perf_stats['mesh_generation_time'] += mesh_time
                
                return simplified_mesh
                
        except Exception as e:
            print(f"Mesh generation error: {e}")
        
        return None
    
    def _process_collision_detection(self, tracked_hands: List[TrackedHand]) -> List:
        """衝突検出処理"""
        collision_start = time.perf_counter()
        collision_events = []
        
        try:
            if self.collision_searcher and self.collision_tester and self.state.current_mesh:
                # 空間インデックス構築（必要に応じて）
                if not self.spatial_index:
                    self.spatial_index = SpatialIndex(self.state.current_mesh, index_type=IndexType.BVH)
                
                # CollisionSearcher初期化（必要に応じて）
                if not self.collision_searcher:
                    self.collision_searcher = CollisionSearcher(
                        self.spatial_index,
                        default_radius=self.config.sphere_radius
                    )
                

                
                # 各手について衝突検出
                for hand in tracked_hands:
                    # トラッキング状態をチェック（属性が存在する場合のみ）
                    is_trackable = True
                    if hasattr(hand, 'tracking_state'):
                        is_trackable = hand.tracking_state.is_stable()
                    
                    if is_trackable and hasattr(hand, 'position') and hand.position is not None:
                        # 新しいAPIを使用
                        search_result = self.collision_searcher.search_near_hand(hand)
                        candidates = search_result.triangle_indices
                        
                        # 三角形インデックスから実際の三角形頂点を取得
                        for triangle_idx in candidates:
                            triangle_vertices = self.state.current_mesh.vertices[
                                self.state.current_mesh.triangles[triangle_idx]
                            ]
                            
                            # シンプルな衝突テスト関数を使用
                            from ..collision.sphere_tri import check_sphere_triangle
                            contact_point = check_sphere_triangle(
                                hand.position, self.config.sphere_radius, triangle_vertices
                            )
                            
                            if contact_point is not None:
                                collision_events.append({
                                    'hand_id': hand.hand_id,
                                    'position': hand.position,
                                    'triangle': triangle_idx,
                                    'contact_point': contact_point,
                                    'timestamp': time.time()
                                })
                
                # イベントキューに追加
                for event in collision_events:
                    self.event_queue.add_event(event)
                
                # パフォーマンス統計更新
                collision_time = time.perf_counter() - collision_start
                self.state.perf_stats['collision_detection_time'] += collision_time
                self.state.perf_stats['collision_events_count'] += len(collision_events)
                
        except Exception as e:
            print(f"Collision detection error: {e}")
        
        return collision_events
    
    def _process_audio_generation(self, collision_events: List) -> List:
        """音響生成処理"""
        audio_start = time.perf_counter()
        audio_events = []
        
        try:
            if self.voice_manager:
                current_time = time.time()
                
                for event in collision_events:
                    hand_id = event['hand_id']
                    
                    # クールダウンチェック
                    last_trigger = self.state.last_audio_trigger_time.get(hand_id, 0)
                    if current_time - last_trigger >= self.config.audio_cooldown_time:
                        # 音響生成
                        note_event = self.voice_manager.trigger_collision_sound(
                            event['position'], hand_id
                        )
                        if note_event:
                            audio_events.append(note_event)
                            self.state.last_audio_trigger_time[hand_id] = current_time
                            self.state.perf_stats['audio_notes_played'] += 1
                
                # パフォーマンス統計更新
                audio_time = time.perf_counter() - audio_start
                self.state.perf_stats['audio_synthesis_time'] += audio_time
                
        except Exception as e:
            print(f"Audio generation error: {e}")
        
        return audio_events
    
    def _update_performance_stats(self, total_time: float) -> None:
        """パフォーマンス統計更新"""
        self.state.frame_count += 1
        self.state.frame_counter += 1
        self.state.perf_stats['frame_count'] = self.state.frame_count
        self.state.perf_stats['total_pipeline_time'] += total_time
        
        # 観察者に通知
        for observer in self.observers:
            observer.on_performance_update(self.state.perf_stats.copy())
    
    def add_observer(self, observer: IPipelineObserver) -> None:
        """観察者追加"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: IPipelineObserver) -> None:
        """観察者削除"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def _notify_observers(self, frame_data: FrameData, results: Dict[str, Any]) -> None:
        """観察者に通知"""
        for observer in self.observers:
            observer.on_frame_processed(frame_data, results)
            if results['collision_events']:
                observer.on_collision_detected(results['collision_events'])
    
    def force_mesh_update(self) -> None:
        """メッシュ強制更新要求"""
        self.state.force_mesh_update_requested = True
    
    def update_configuration(self, new_config: PipelineConfiguration) -> None:
        """設定更新"""
        self.config = new_config
        # 必要に応じてコンポーネント再初期化
    
    def get_state(self) -> PipelineState:
        """現在の状態取得"""
        return self.state
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return self.state.perf_stats.copy()
    
    # ManagedResource実装
    @property
    def resource_type(self) -> str:
        """リソースタイプ"""
        return "GeocussionPipelineController"
    
    def initialize_resource(self) -> bool:
        """リソース初期化"""
        return self.initialize()
    
    def cleanup(self) -> None:
        """リソースクリーンアップ（ManagedResource用）"""
        if self.camera:
            self.camera.cleanup()
        if self.audio_synthesizer:
            self.audio_synthesizer.cleanup()
    
    def cleanup_resource(self) -> None:
        """リソースクリーンアップ（外部呼び出し用）"""
        self.cleanup()
    
    def get_memory_usage(self) -> int:
        """メモリ使用量推定（MB）"""
        usage = 50  # 基本使用量
        if self.state.current_mesh:
            usage += 20  # メッシュデータ
        if self.state.current_tracked_hands:
            usage += len(self.state.current_tracked_hands) * 5  # 手データ
        return usage