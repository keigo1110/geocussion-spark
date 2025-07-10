#!/usr/bin/env python3
"""
統合パイプライン処理（Clean Architecture適用）
責務: 入力→検出→メッシュ→衝突→音響の全フェーズ統合処理

このファイルは後方互換性のためのラッパーです。
実際の処理は pipeline パッケージ内のモジュールで行われます。
"""

import time
import psutil
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

# 新しいパイプラインモジュール
from .pipeline.orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResults as NewPipelineResults
from .pipeline.input_stage import InputStageConfig
from .pipeline.detection_stage import DetectionStageConfig
from .pipeline.mesh_stage import MeshStageConfig
from .pipeline.collision_stage import CollisionStageConfig
from .pipeline.audio_stage import AudioStageConfig

# 既存の型定義（後方互換性のため）
from ..data_types import FrameData
from ..sound.mapping import ScaleType, InstrumentType
from ..input.stream import OrbbecCamera
from ..input.depth_filter import FilterType
from ..detection.hands3d import DepthInterpolationMethod
from ..detection.tracker import TrackedHand
from ..mesh.projection import ProjectionMethod
from ..mesh.index import IndexType
from ..sound.voice_mgr import StealStrategy
from .. import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResults:
    """パイプライン処理結果（後方互換性のため）"""
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
    """統合パイプライン設定（後方互換性のため）"""
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
    
    # ヘッドレスモード設定
    headless_mode: bool = False
    
    def to_pipeline_config(self) -> PipelineConfig:
        """新しいPipelineConfigに変換"""
        return PipelineConfig(
            input=InputStageConfig(
                enable_filter=self.enable_filter,
                filter_type=FilterType.COMBINED,
                enable_voxel_downsampling=self.enable_voxel_downsampling,
                voxel_size=self.voxel_size
            ),
            detection=DetectionStageConfig(
                enable_hand_detection=self.enable_hand_detection,
                enable_tracking=self.enable_tracking,
                min_detection_confidence=self.min_detection_confidence,
                use_gpu_mediapipe=self.use_gpu_mediapipe
            ),
            mesh=MeshStageConfig(
                enable_mesh_generation=self.enable_mesh_generation,
                mesh_update_interval=self.mesh_update_interval,
                max_mesh_skip_frames=self.max_mesh_skip_frames,
                projection_method=ProjectionMethod.MEAN_HEIGHT,
                mesh_resolution=self.mesh_resolution,
                mesh_quality=self.mesh_quality_threshold,
                mesh_reduction=self.mesh_reduction,
                enable_gpu_triangulation=self.enable_gpu_acceleration
            ),
            collision=CollisionStageConfig(
                enable_collision_detection=self.enable_collision_detection,
                sphere_radius=self.sphere_radius,
                enable_gpu_collision=self.enable_gpu_acceleration
            ),
            audio=AudioStageConfig(
                enable_audio_synthesis=self.enable_audio_synthesis,
                audio_scale=self.audio_scale,
                audio_instrument=self.audio_instrument,
                audio_polyphony=self.audio_polyphony,
                audio_master_volume=self.audio_master_volume
            )
        )


class HandledPipeline:
    """
    統合パイプライン処理クラス（後方互換性ラッパー）
    
    このクラスは既存のインターフェースを維持しながら、
    新しいパイプラインオーケストレーターを内部で使用します。
    """
    
    def __init__(self, config: HandledPipelineConfig):
        """統合パイプライン初期化"""
        logger.info("統合パイプライン初期化中...")
        
        # 設定保存
        self.config = config
        
        # エラーカウンター（後方互換性のため）
        self.color_extraction_errors = 0
        self.max_color_extraction_errors = 10
        self.color_extraction_disabled = False
        
        # カメラ
        self.camera: Optional[OrbbecCamera] = None
        
        # 新しいオーケストレーターを内部で使用
        pipeline_config = config.to_pipeline_config()
        self._orchestrator = PipelineOrchestrator(pipeline_config, self.camera)
        
        # パフォーマンス統計（後方互換性のため）
        self._total_frames = 0
        self._total_processing_time = 0.0
        self._stage_times = {
            'filter': [],
            'detection': [],
            'projection': [],
            'tracking': [],
            'mesh': [],
            'collision': [],
            'audio': []
        }
        
        logger.info("統合パイプラインの作成が完了しました")
    
    def initialize(self, camera: Optional[OrbbecCamera] = None) -> bool:
        """
        パイプラインを初期化
        
        Args:
            camera: OrbbecCameraインスタンス（オプション）
            
        Returns:
            初期化に成功した場合True
        """
        self.camera = camera
        if camera:
            self._orchestrator.camera = camera
            self._orchestrator.input_stage.camera = camera
        
        # オーケストレーターを初期化
        if not self._orchestrator.initialize():
            logger.error("パイプラインの初期化に失敗しました")
            return False
        
        logger.info("統合パイプラインの初期化が完了しました")
        return True
    
    def process_frame(self, frame_data: Optional[FrameData] = None) -> PipelineResults:
        """
        フレームを処理（統合パイプライン）
        
        Args:
            frame_data: 処理するフレームデータ（Noneの場合はカメラから取得）
            
        Returns:
            パイプライン処理結果
        """
        # 新しいオーケストレーターで処理
        new_results = self._orchestrator.process_frame(frame_data)
        
        # 結果を既存の形式に変換（後方互換性のため）
        results = PipelineResults(
            frame_data=new_results.frame_data,
            hands_2d=new_results.hands_2d,
            hands_3d=new_results.hands_3d,
            tracked_hands=new_results.tracked_hands,
            collision_events=new_results.collision_events,
            performance_stats={
                'total_ms': new_results.processing_time_ms,
                'stage_times': new_results.stage_times,
                'active_voices': new_results.active_voices
            }
        )
        
        # メッシュ情報を追加
        if new_results.mesh_vertices is not None:
            results.mesh = {
                'vertices': new_results.mesh_vertices,
                'triangles': new_results.mesh_triangles,
                'colors': new_results.mesh_colors
            }
        
        # パフォーマンス統計更新（後方互換性のため）
        self._total_frames += 1
        self._total_processing_time += new_results.processing_time_ms
        
        return results
    
    def force_mesh_update(self) -> None:
        """次のフレームで強制的にメッシュを更新"""
        self._orchestrator.force_mesh_update()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        設定を動的に更新
        
        Args:
            updates: 更新する設定項目の辞書
        """
        # 既存の設定を更新
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # オーケストレーターに伝達
        orchestrator_updates = {}
        
        # キーマッピング
        key_mapping = {
            'enable_filter': 'input_enable_filter',
            'enable_hand_detection': 'detection_enable_hand_detection',
            'enable_tracking': 'detection_enable_tracking',
            'min_detection_confidence': 'detection_min_detection_confidence',
            'enable_mesh_generation': 'mesh_enable_mesh_generation',
            'mesh_update_interval': 'mesh_mesh_update_interval',
            'sphere_radius': 'collision_sphere_radius',
            'enable_collision_detection': 'collision_enable_collision_detection',
            'enable_audio_synthesis': 'audio_enable_audio_synthesis',
            'audio_scale': 'audio_audio_scale',
            'audio_instrument': 'audio_audio_instrument',
            'audio_master_volume': 'audio_audio_master_volume'
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in updates:
                orchestrator_updates[new_key] = updates[old_key]
        
        if orchestrator_updates:
            self._orchestrator.update_config(orchestrator_updates)
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        logger.info("統合パイプラインをクリーンアップ中...")
        self._orchestrator.cleanup()
        logger.info("統合パイプラインのクリーンアップが完了しました")
    
    # 後方互換性のためのプロパティ
    @property
    def hands_2d(self):
        """MediaPipeハンドラッパーへのアクセス（後方互換性）"""
        return self._orchestrator.detection_stage.hand_detector
    
    @property
    def projector_3d(self):
        """3Dプロジェクターへのアクセス（後方互換性）"""
        return self._orchestrator.detection_stage.hand_projector
    
    @property
    def tracker(self):
        """トラッカーへのアクセス（後方互換性）"""
        return self._orchestrator.detection_stage.hand_tracker
    
    @property
    def synthesizer(self):
        """シンセサイザーへのアクセス（後方互換性）"""
        return self._orchestrator.audio_stage.synthesizer