#!/usr/bin/env python3
"""
パイプラインオーケストレーター: 各ステージを統合

すべてのステージを管理し、データフローを制御します。
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import psutil

from .base import PipelineStage
from .input_stage import InputStage, InputStageConfig
from .detection_stage import DetectionStage, DetectionStageConfig
from .mesh_stage import MeshStage, MeshStageConfig
from .collision_stage import CollisionStage, CollisionStageConfig
from .audio_stage import AudioStage, AudioStageConfig

from ...types import FrameData
from ...input.stream import OrbbecCamera
from ...sound.mapping import ScaleType, InstrumentType
from ...input.depth_filter import FilterType
from ... import get_logger

# イベントシステム
from ..events import (
    get_event_dispatcher,
    FrameProcessedEvent,
    StageCompletedEvent,
    ErrorEvent,
    MeshUpdatedEvent,
    CollisionDetectedEvent,
    AudioTriggeredEvent,
    HandsDetectedEvent,
    PointCloudGeneratedEvent
)


@dataclass
class PipelineConfig:
    """統合パイプライン設定"""
    # 各ステージの設定
    input: InputStageConfig = field(default_factory=InputStageConfig)
    detection: DetectionStageConfig = field(default_factory=DetectionStageConfig)
    mesh: MeshStageConfig = field(default_factory=MeshStageConfig)
    collision: CollisionStageConfig = field(default_factory=CollisionStageConfig)
    audio: AudioStageConfig = field(default_factory=AudioStageConfig)
    
    # パフォーマンス設定
    enable_performance_tracking: bool = True
    performance_log_interval: int = 100  # フレーム数


@dataclass
class PipelineResults:
    """パイプライン処理結果"""
    frame_data: Optional[FrameData] = None
    point_cloud: Optional[Any] = None
    colors: Optional[Any] = None
    hands_2d: List[Any] = field(default_factory=list)
    hands_3d: List[Any] = field(default_factory=list)
    tracked_hands: List[Any] = field(default_factory=list)
    mesh_vertices: Optional[Any] = None
    mesh_triangles: Optional[Any] = None
    mesh_colors: Optional[Any] = None
    collision_events: List[Any] = field(default_factory=list)
    active_collisions: Dict[int, List[Any]] = field(default_factory=dict)
    active_voices: int = 0
    processing_time_ms: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)


class PipelineOrchestrator:
    """パイプラインオーケストレーター"""
    
    def __init__(self, config: PipelineConfig, camera: Optional[OrbbecCamera] = None) -> None:
        """
        初期化
        
        Args:
            config: パイプライン設定
            camera: カメラインスタンス
        """
        self.config = config
        self.camera = camera
        self.logger = get_logger(__name__)
        
        # 各ステージを作成
        self.input_stage = InputStage(config.input, camera)
        self.detection_stage = DetectionStage(config.detection)
        self.mesh_stage = MeshStage(config.mesh)
        self.collision_stage = CollisionStage(config.collision)
        self.audio_stage = AudioStage(config.audio)
        
        # カメラ内部パラメータをDetectionStageに渡す
        if camera and hasattr(camera, 'depth_intrinsics'):
            self.detection_stage.camera_intrinsics = camera.depth_intrinsics
        else:
            # カメラがない場合はInputStageから取得を試みる
            self.detection_stage.camera_intrinsics = None
        
        # イベントディスパッチャー
        self.event_dispatcher = get_event_dispatcher()
        
        # パフォーマンス統計
        self._frame_count = 0
        self._total_time = 0.0
        self._stage_total_times = {
            'input': 0.0,
            'detection': 0.0,
            'mesh': 0.0,
            'collision': 0.0,
            'audio': 0.0
        }
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """すべてのステージを初期化"""
        self.logger.info("パイプラインオーケストレーターを初期化中...")
        
        # 各ステージの初期化
        stages = [
            ('input', self.input_stage),
            ('detection', self.detection_stage),
            ('mesh', self.mesh_stage),
            ('collision', self.collision_stage),
            ('audio', self.audio_stage)
        ]
        
        for stage_name, stage in stages:
            if not stage.initialize():
                self.logger.error(f"{stage_name}ステージの初期化に失敗しました")
                return False
            self.logger.info(f"{stage_name}ステージを初期化しました")
            
            # InputStage初期化後、カメラintrinsicsをDetectionStageに設定
            if stage_name == 'input' and self.detection_stage.camera_intrinsics is None:
                intrinsics = self.input_stage.get_camera_intrinsics()
                if intrinsics:
                    self.detection_stage.camera_intrinsics = intrinsics
                    self.logger.info("カメラintrinsicsをDetectionStageに設定しました")
        
        self._initialized = True
        self.logger.info("パイプラインオーケストレーターの初期化が完了しました")
        return True
    
    def process_frame(self, frame_data: Optional[FrameData] = None) -> PipelineResults:
        """
        フレームを処理
        
        Args:
            frame_data: 入力フレームデータ（Noneの場合は内部カメラから取得）
            
        Returns:
            処理結果
        """
        if not self._initialized:
            self.logger.error("パイプラインが初期化されていません")
            return PipelineResults()
        
        start_time = time.perf_counter()
        results = PipelineResults()
        stage_times = {}
        
        try:
            # 1. 入力ステージ
            stage_start = time.perf_counter()
            input_result = self.input_stage.process(frame_data)
            stage_times['input'] = (time.perf_counter() - stage_start) * 1000
            
            if not input_result.success:
                self.logger.warning(f"入力ステージエラー: {input_result.error_message}")
                # エラーイベント発行
                self.event_dispatcher.publish(ErrorEvent(
                    stage_name="input",
                    error_message=input_result.error_message or "Unknown error",
                    error_type="INPUT_ERROR",
                    recoverable=True
                ))
                return results
            
            results.frame_data = input_result.frame_data
            results.point_cloud = input_result.point_cloud
            results.colors = input_result.colors
            
            # フレーム処理イベント発行
            if input_result.frame_data:
                self.event_dispatcher.publish(FrameProcessedEvent(
                    frame_data=input_result.frame_data,
                    frame_number=self._frame_count
                ))
            
            # 点群生成イベント発行
            if input_result.point_cloud is not None:
                self.event_dispatcher.publish(PointCloudGeneratedEvent(
                    point_cloud=input_result.point_cloud,
                    colors=input_result.colors
                ))
            
            # ステージ完了イベント発行
            self.event_dispatcher.publish(StageCompletedEvent(
                stage_name="input",
                processing_time_ms=stage_times['input'],
                success=True
            ))
            
            # 2. 検出ステージ
            if self.config.detection.enable_hand_detection:
                stage_start = time.perf_counter()
                detection_result = self.detection_stage.process(
                    input_result.frame_data,
                    input_result.color_image
                )
                stage_times['detection'] = (time.perf_counter() - stage_start) * 1000
                
                if detection_result.success:
                    results.hands_2d = detection_result.hands_2d
                    results.hands_3d = detection_result.hands_3d
                    results.tracked_hands = detection_result.tracked_hands
                    
                    # 手検出イベント発行
                    if detection_result.hands_3d:
                        self.event_dispatcher.publish(HandsDetectedEvent(
                            hands_2d=detection_result.hands_2d,
                            hands_3d=detection_result.hands_3d,
                            tracked_hands=detection_result.tracked_hands
                        ))
                    
                    # ステージ完了イベント発行
                    self.event_dispatcher.publish(StageCompletedEvent(
                        stage_name="detection",
                        processing_time_ms=stage_times['detection'],
                        success=True
                    ))
                else:
                    # エラーイベント発行
                    self.event_dispatcher.publish(ErrorEvent(
                        stage_name="detection",
                        error_message=detection_result.error_message or "Detection failed",
                        error_type="DETECTION_ERROR",
                        recoverable=True
                    ))
            
            # 3. メッシュステージ
            if self.config.mesh.enable_mesh_generation and input_result.point_cloud is not None:
                stage_start = time.perf_counter()
                mesh_result = self.mesh_stage.process(
                    input_result.point_cloud,
                    input_result.colors
                )
                stage_times['mesh'] = (time.perf_counter() - stage_start) * 1000
                
                if mesh_result.success:
                    if mesh_result.mesh_updated:
                        results.mesh_vertices = mesh_result.vertices
                        results.mesh_triangles = mesh_result.triangles
                        results.mesh_colors = mesh_result.colors
                        
                        # メッシュ更新イベント発行
                        self.event_dispatcher.publish(MeshUpdatedEvent(
                            vertices=mesh_result.vertices,
                            triangles=mesh_result.triangles,
                            colors=mesh_result.colors
                        ))
                    
                    # ステージ完了イベント発行
                    self.event_dispatcher.publish(StageCompletedEvent(
                        stage_name="mesh",
                        processing_time_ms=stage_times['mesh'],
                        success=True
                    ))
                else:
                    # エラーイベント発行
                    self.event_dispatcher.publish(ErrorEvent(
                        stage_name="mesh",
                        error_message=mesh_result.error_message or "Mesh generation failed",
                        error_type="MESH_ERROR",
                        recoverable=True
                    ))
            
            # 4. 衝突検出ステージ
            if (self.config.collision.enable_collision_detection and 
                results.tracked_hands and 
                results.mesh_vertices is not None):
                stage_start = time.perf_counter()
                collision_result = self.collision_stage.process(
                    results.tracked_hands,
                    results.mesh_vertices,
                    results.mesh_triangles,
                    mesh_updated=results.mesh_triangles is not None
                )
                stage_times['collision'] = (time.perf_counter() - stage_start) * 1000
                
                if collision_result.success:
                    results.collision_events = collision_result.collision_events
                    results.active_collisions = collision_result.active_collisions
                    
                    # 衝突検出イベント発行
                    if collision_result.collision_events:
                        self.event_dispatcher.publish(CollisionDetectedEvent(
                            collision_events=collision_result.collision_events,
                            active_collisions=collision_result.active_collisions
                        ))
                    
                    # ステージ完了イベント発行
                    self.event_dispatcher.publish(StageCompletedEvent(
                        stage_name="collision",
                        processing_time_ms=stage_times['collision'],
                        success=True
                    ))
                else:
                    # エラーイベント発行
                    self.event_dispatcher.publish(ErrorEvent(
                        stage_name="collision",
                        error_message=collision_result.error_message or "Collision detection failed",
                        error_type="COLLISION_ERROR",
                        recoverable=True
                    ))
            
            # 5. 音響ステージ
            if self.config.audio.enable_audio_synthesis and results.collision_events:
                stage_start = time.perf_counter()
                audio_result = self.audio_stage.process(results.collision_events)
                stage_times['audio'] = (time.perf_counter() - stage_start) * 1000
                
                if audio_result.success:
                    results.active_voices = audio_result.active_voices
                    
                    # 音響トリガーイベント発行（各衝突イベントに対して）
                    for collision_event in results.collision_events:
                        # Note: AudioStageは実際の音符情報を返さないので、簡易的に作成
                        self.event_dispatcher.publish(AudioTriggeredEvent(
                            collision_event=collision_event,
                            note=60,  # TODO: 実際の音符を取得
                            velocity=collision_event.force * 127,
                            duration=0.5
                        ))
                    
                    # ステージ完了イベント発行
                    self.event_dispatcher.publish(StageCompletedEvent(
                        stage_name="audio",
                        processing_time_ms=stage_times['audio'],
                        success=True
                    ))
                else:
                    # エラーイベント発行
                    self.event_dispatcher.publish(ErrorEvent(
                        stage_name="audio",
                        error_message=audio_result.error_message or "Audio synthesis failed",
                        error_type="AUDIO_ERROR",
                        recoverable=True
                    ))
            
            # 処理時間計算
            results.processing_time_ms = (time.perf_counter() - start_time) * 1000
            results.stage_times = stage_times
            
            # パフォーマンス統計更新
            self._update_performance_stats(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"パイプライン処理エラー: {e}")
            import traceback
            traceback.print_exc()
            
            # 致命的エラーイベント発行
            self.event_dispatcher.publish(ErrorEvent(
                stage_name="pipeline",
                error_message=str(e),
                error_type="FATAL_ERROR",
                recoverable=False
            ))
            
            return results
    
    def _update_performance_stats(self, results: PipelineResults) -> None:
        """パフォーマンス統計を更新"""
        self._frame_count += 1
        self._total_time += results.processing_time_ms
        
        for stage, time_ms in results.stage_times.items():
            if stage in self._stage_total_times:
                self._stage_total_times[stage] += time_ms
        
        # 定期的にログ出力
        if (self.config.enable_performance_tracking and 
            self._frame_count % self.config.performance_log_interval == 0):
            avg_time = self._total_time / self._frame_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            
            self.logger.info(f"パフォーマンス統計 (フレーム {self._frame_count}):")
            self.logger.info(f"  平均処理時間: {avg_time:.2f} ms ({fps:.1f} FPS)")
            
            for stage, total_time in self._stage_total_times.items():
                avg_stage_time = total_time / self._frame_count
                percentage = (avg_stage_time / avg_time * 100) if avg_time > 0 else 0
                self.logger.info(f"  {stage}: {avg_stage_time:.2f} ms ({percentage:.1f}%)")
            
            # メモリ使用量
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"  メモリ使用量: {memory_mb:.1f} MB")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """設定を動的に更新"""
        # 各ステージの設定を更新
        for key, value in updates.items():
            if key.startswith('input_'):
                self.input_stage.update_config({key[6:]: value})
            elif key.startswith('detection_'):
                self.detection_stage.update_config({key[10:]: value})
            elif key.startswith('mesh_'):
                self.mesh_stage.update_config({key[5:]: value})
            elif key.startswith('collision_'):
                self.collision_stage.update_config({key[10:]: value})
            elif key.startswith('audio_'):
                self.audio_stage.update_config({key[6:]: value})
    
    def force_mesh_update(self) -> None:
        """次のフレームで強制的にメッシュを更新"""
        self.mesh_stage.force_update()
    
    def cleanup(self) -> None:
        """すべてのステージをクリーンアップ"""
        self.logger.info("パイプラインオーケストレーターをクリーンアップ中...")
        
        stages = [
            self.input_stage,
            self.detection_stage,
            self.mesh_stage,
            self.collision_stage,
            self.audio_stage
        ]
        
        for stage in stages:
            try:
                stage.cleanup()
            except Exception as e:
                self.logger.error(f"ステージクリーンアップエラー: {e}")
        
        self._initialized = False
        self.logger.info("パイプラインオーケストレーターのクリーンアップが完了しました")