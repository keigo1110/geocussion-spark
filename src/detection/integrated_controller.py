#!/usr/bin/env python3
"""
統合手追跡コントローラー: 60fps手追跡・統一イベント・音響・衝突統合

MediaPipe + 60fps Kalman Tracker + 統一イベントシステム + 音響・衝突検出を
単一のコントローラーで統合管理します。
"""

import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from .tracker_60fps import (
    HighFrequencyHand3DTracker,
    HighFrequencyKalmanConfig,
    HighFrequencyTrackedHand,
    create_high_frequency_tracker
)
from .unified_events import (
    UnifiedHandEventStream,
    UnifiedHandEvent,
    UnifiedHandEventType,
    create_unified_event_stream
)
from ..data_types import CameraIntrinsics
from .. import get_logger

logger = get_logger(__name__)


@dataclass
class IntegratedControllerConfig:
    """統合コントローラー設定"""
    # 基本設定
    target_fps: int = 60
    mediapipe_detection_fps: int = 15
    collision_check_fps: int = 60
    audio_synthesis_fps: int = 60
    
    # Kalman Tracker設定
    kalman_config: Optional[HighFrequencyKalmanConfig] = None
    
    # 衝突検出設定
    collision_sphere_radius: float = 0.08  # 8cm
    collision_enabled: bool = True
    
    # 音響合成設定
    audio_enabled: bool = True
    audio_cooldown_ms: float = 50.0  # 50ms cooldown
    
    # パフォーマンス設定
    max_concurrent_hands: int = 4
    enable_prediction: bool = True
    enable_interpolation: bool = True
    
    # デバッグ設定
    enable_debug_logging: bool = False
    enable_performance_monitoring: bool = True
    
    # MediaPipe設定
    mediapipe_confidence: float = 0.7
    mediapipe_tracking_confidence: float = 0.5
    use_gpu_mediapipe: bool = False
    max_num_hands: int = 2
    
    # 統計更新間隔
    stats_update_interval: int = 60


class IntegratedHandTrackingController:
    """統合手追跡コントローラー"""
    
    def __init__(
        self,
        config: IntegratedControllerConfig,
        camera_intrinsics: CameraIntrinsics,
        hands_2d: 'MediaPipeHandsWrapper',
        hands_3d: 'Hand3DProjector',
        collision_searcher: Optional['CollisionSearcher'] = None,
        audio_mapper: Optional['AudioMapper'] = None,
        audio_synthesizer: Optional['AudioSynthesizer'] = None
    ):
        """初期化"""
        self.config = config
        self.camera_intrinsics = camera_intrinsics
        self.hands_2d = hands_2d
        self.hands_3d = hands_3d
        self.collision_searcher = collision_searcher
        self.audio_mapper = audio_mapper
        self.audio_synthesizer = audio_synthesizer
        
        # 60fps手追跡器
        kalman_config = config.kalman_config or HighFrequencyKalmanConfig(
            target_fps=config.target_fps,
            mediapipe_fps=config.mediapipe_detection_fps
        )
        self.tracker_60fps = create_high_frequency_tracker(
            target_fps=config.target_fps,
            mediapipe_fps=config.mediapipe_detection_fps
        )
        
        # 統一イベントストリーム
        self.event_stream = create_unified_event_stream(
            max_events=1000,
            event_retention_time=5.0
        )
        
        # タイミング制御
        self.last_mediapipe_time = 0.0
        self.last_collision_time = 0.0
        self.last_audio_time = 0.0
        
        # 音響クールダウン管理
        self.audio_cooldown_times: Dict[str, float] = {}
        
        # パフォーマンス統計
        self.frame_count = 0
        self.performance_stats = {
            'total_frames': 0,
            'mediapipe_executions': 0,
            'collision_checks': 0,
            'audio_triggers': 0,
            'average_fps': 0.0,
            'interpolation_rate': 0.0,
            'tracking_accuracy': 0.0
        }
        
        # 外部イベントリスナー
        self.external_listeners: List[Callable[[UnifiedHandEvent], None]] = []
        
        logger.info(f"Integrated controller initialized: {config.target_fps}fps target")

    def process_frame(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[UnifiedHandEvent]:
        """フレーム処理（60fps）"""
        if timestamp is None:
            timestamp = time.perf_counter()
        
        frame_start = timestamp
        events = []
        
        try:
            # 1. MediaPipe検出（必要に応じて）
            hands_3d = self._run_mediapipe_detection_if_needed(
                color_image, depth_image, timestamp
            )
            
            # 2. 60fps手追跡更新
            tracked_hands = self._update_60fps_tracking(hands_3d, timestamp)
            
            # 3. 衝突検出（必要に応じて）
            if self.config.collision_enabled and self._should_check_collision(timestamp):
                self._check_collisions(tracked_hands, timestamp)
                self.last_collision_time = timestamp
            
            # 4. 音響合成（必要に応じて）
            if self.config.audio_enabled and self._should_synthesize_audio(timestamp):
                self._synthesize_audio(tracked_hands, timestamp)
                self.last_audio_time = timestamp
            
            # 5. 統一イベント処理
            recent_events = self.event_stream.get_recent_events(max_age=0.1)
            for event in recent_events:
                self._handle_unified_event(event)
                events.append(event)
            
            # 6. パフォーマンス統計更新
            frame_time_ms = (time.perf_counter() - frame_start) * 1000
            self._update_performance_stats(frame_time_ms, hands_3d is not None)
            
            return events
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return []

    def _run_mediapipe_detection_if_needed(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        timestamp: float
    ) -> Optional[List['Hand3DResult']]:
        """必要に応じてMediaPipe検出を実行"""
        detection_interval = 1.0 / self.config.mediapipe_detection_fps
        
        if timestamp - self.last_mediapipe_time >= detection_interval:
            try:
                # 2D手検出
                hands_2d = self.hands_2d.detect_hands(color_image)
                
                if hands_2d:
                    # 3D投影
                    hands_3d = []
                    for hand_2d in hands_2d:
                        hand_3d = self.hands_3d.project_hand_to_3d(hand_2d, depth_image)
                        if hand_3d:
                            hands_3d.append(hand_3d)
                    
                    self.last_mediapipe_time = timestamp
                    self.performance_stats['mediapipe_executions'] += 1
                    
                    logger.debug(f"MediaPipe detection: {len(hands_2d)} 2D -> {len(hands_3d)} 3D")
                    return hands_3d
                    
            except Exception as e:
                logger.error(f"MediaPipe detection error: {e}")
        
        return None

    def _update_60fps_tracking(
        self,
        hands_3d: Optional[List['Hand3DResult']],
        timestamp: float
    ) -> List[HighFrequencyTrackedHand]:
        """60fps手追跡更新"""
        try:
            tracked_hands = self.tracker_60fps.update_60fps(hands_3d)
            
            # 統一イベントストリームに追跡更新を処理
            if tracked_hands:
                events = self.event_stream.process_tracking_update(tracked_hands)
                # イベントは自動的に発火される
            
            return tracked_hands
            
        except Exception as e:
            logger.error(f"60fps tracking error: {e}")
            return []

    def _should_check_collision(self, timestamp: float) -> bool:
        """衝突検出が必要かチェック"""
        collision_interval = 1.0 / self.config.collision_check_fps
        return timestamp - self.last_collision_time >= collision_interval

    def _check_collisions(
        self,
        tracked_hands: List[HighFrequencyTrackedHand],
        timestamp: float
    ) -> None:
        """衝突検出実行"""
        if not self.collision_searcher:
            return
        
        try:
            for hand in tracked_hands:
                if hand.position is None:
                    continue
                
                # 衝突検索
                search_result = self.collision_searcher._search_point(
                    hand.position,
                    self.config.collision_sphere_radius
                )
                
                if search_result and search_result.triangle_indices:
                    # 衝突イベント生成
                    collision_event = self._create_collision_event(hand, search_result, timestamp)
                    if collision_event:
                        self.event_stream.add_event(collision_event)
                        self.performance_stats['collision_checks'] += 1
                        
        except Exception as e:
            logger.error(f"Collision detection error: {e}")

    def _should_synthesize_audio(self, timestamp: float) -> bool:
        """音響合成が必要かチェック"""
        audio_interval = 1.0 / self.config.audio_synthesis_fps
        return timestamp - self.last_audio_time >= audio_interval

    def _synthesize_audio(
        self,
        tracked_hands: List[HighFrequencyTrackedHand],
        timestamp: float
    ) -> None:
        """音響合成実行"""
        if not self.audio_mapper or not self.audio_synthesizer:
            return
        
        try:
            # 衝突イベントから音響生成
            collision_events = self.event_stream.get_recent_events(
                event_types=[UnifiedHandEventType.COLLISION_DETECTED],
                max_age=0.1
            )
            
            for event in collision_events:
                if not self._is_audio_cooldown_active(event.hand_id, timestamp):
                    # 音響パラメータマッピング
                    # 衝突イベントを作成してマッピング
                    from ..collision.events import CollisionEvent, CollisionIntensity
                    collision_event = CollisionEvent(
                        event_id=f"collision_{event.hand_id}_{int(time.time()*1000)}",
                        hand_id=event.hand_id,
                        contact_position=event.position_3d,
                        velocity=np.linalg.norm(event.velocity_3d) if event.velocity_3d is not None else 0.0,
                        penetration_depth=event.collision_penetration or 0.01,
                        surface_normal=event.collision_normal or np.array([0, 1, 0]),
                        contact_area=0.01,
                        intensity=CollisionIntensity.MEDIUM,
                        timestamp=event.timestamp
                    )
                    audio_params = self.audio_mapper.map_collision_event(collision_event)
                    
                    # 音響再生
                    voice_id = self.audio_synthesizer.play_audio_parameters(audio_params)
                    
                    if voice_id:
                        self._set_audio_cooldown(event.hand_id, timestamp)
                        self.performance_stats['audio_triggers'] += 1
                        
                        # 音響イベント生成
                        audio_event = UnifiedHandEvent(
                            event_type=UnifiedHandEventType.AUDIO_TRIGGERED,
                            hand_id=event.hand_id,
                            timestamp=timestamp,
                            handedness=event.handedness,
                            position_3d=event.position_3d,
                            velocity_3d=event.velocity_3d,
                            acceleration_3d=event.acceleration_3d,
                            confidence=event.confidence,
                            tracking_quality=event.tracking_quality,
                            speed=event.speed,
                            acceleration_magnitude=event.acceleration_magnitude,
                            is_moving=event.is_moving,
                            is_accelerating=event.is_accelerating,
                            audio_frequency=audio_params.frequency,
                            audio_velocity=audio_params.velocity,
                            interpolated=event.interpolated
                        )
                        self.event_stream.add_event(audio_event)
                        
        except Exception as e:
            logger.error(f"Audio synthesis error: {e}")

    def _create_collision_event(
        self,
        hand: HighFrequencyTrackedHand,
        search_result: Any,
        timestamp: float
    ) -> Optional[Any]:
        """衝突イベント生成"""
        try:
            return UnifiedHandEvent(
                event_type=UnifiedHandEventType.COLLISION_DETECTED,
                hand_id=hand.id,
                timestamp=timestamp,
                handedness=hand.handedness,
                position_3d=hand.position.copy(),
                velocity_3d=hand.velocity.copy(),
                acceleration_3d=hand.acceleration.copy(),
                confidence=hand.confidence_tracking,
                tracking_quality=0.8,
                speed=hand.speed,
                acceleration_magnitude=hand.acceleration_magnitude,
                is_moving=hand.is_moving,
                is_accelerating=hand.is_accelerating,
                collision_position=hand.position.copy(),
                collision_normal=np.array([0, 1, 0]),
                collision_penetration=0.01,  # 簡易実装
                interpolated=hand.is_interpolated
            )
        except Exception as e:
            logger.error(f"Collision event creation error: {e}")
            return None

    def _is_audio_cooldown_active(self, hand_id: str, timestamp: float) -> bool:
        """音響クールダウンが有効かチェック"""
        last_time = self.audio_cooldown_times.get(hand_id, 0.0)
        cooldown_sec = self.config.audio_cooldown_ms / 1000.0
        return timestamp - last_time < cooldown_sec

    def _set_audio_cooldown(self, hand_id: str, timestamp: float) -> None:
        """音響クールダウン設定"""
        self.audio_cooldown_times[hand_id] = timestamp

    def _handle_unified_event(self, event: UnifiedHandEvent) -> None:
        """統一イベント処理"""
        # 外部リスナーに通知
        for listener in self.external_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")

    def _update_performance_stats(self, frame_time_ms: float, had_detection: bool) -> None:
        """パフォーマンス統計更新"""
        self.frame_count += 1
        self.performance_stats['total_frames'] = self.frame_count
        
        if self.frame_count % self.config.stats_update_interval == 0:
            # 統計計算
            tracker_stats = self.tracker_60fps.get_performance_stats()
            
            self.performance_stats['average_fps'] = tracker_stats.get('average_fps', 0.0)
            self.performance_stats['interpolation_rate'] = tracker_stats.get('interpolation_rate', 0.0)
            self.performance_stats['tracking_accuracy'] = tracker_stats.get('tracking_accuracy', 0.0)

    def add_event_listener(self, listener: Callable[[UnifiedHandEvent], None]) -> None:
        """外部イベントリスナー追加"""
        self.external_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[UnifiedHandEvent], None]) -> None:
        """外部イベントリスナー削除"""
        if listener in self.external_listeners:
            self.external_listeners.remove(listener)

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return self.performance_stats.copy()

    def get_current_hands(self) -> List[HighFrequencyTrackedHand]:
        """現在の手一覧取得"""
        return self.tracker_60fps._get_active_tracks()

    def get_recent_events(
        self,
        event_types: Optional[List[UnifiedHandEventType]] = None,
        max_age: Optional[float] = None
    ) -> List[UnifiedHandEvent]:
        """最近のイベント取得"""
        return self.event_stream.get_recent_events(event_types, max_age)

    def reset(self) -> None:
        """リセット"""
        self.tracker_60fps.reset()
        self.event_stream.clear()
        self.audio_cooldown_times.clear()
        self.frame_count = 0
        self.performance_stats = {
            'total_frames': 0,
            'mediapipe_executions': 0,
            'collision_checks': 0,
            'audio_triggers': 0,
            'average_fps': 0.0,
            'interpolation_rate': 0.0,
            'tracking_accuracy': 0.0
        }
        logger.info("Integrated controller reset")

    def stop(self) -> None:
        """停止"""
        self.cleanup()

    def cleanup(self) -> None:
        """クリーンアップ"""
        try:
            if self.audio_synthesizer:
                self.audio_synthesizer.stop_engine()
            
            self.event_stream.clear()
            self.external_listeners.clear()
            
            logger.info("Integrated controller cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# =============================================================================
# ファクトリー関数
# =============================================================================

def create_integrated_controller_from_instances(
    camera_intrinsics: CameraIntrinsics,
    hands_2d: 'MediaPipeHandsWrapper',
    hands_3d: 'Hand3DProjector',
    target_fps: int = 60,
    collision_searcher: Optional['CollisionSearcher'] = None,
    audio_mapper: Optional['AudioMapper'] = None,
    audio_synthesizer: Optional['AudioSynthesizer'] = None
) -> IntegratedHandTrackingController:
    """統合コントローラを作成（インスタンス受け渡し版）"""
    config = IntegratedControllerConfig(
        target_fps=target_fps,
        mediapipe_detection_fps=15,
        collision_check_fps=60,
        audio_synthesis_fps=60,
        collision_enabled=collision_searcher is not None,
        audio_enabled=audio_mapper is not None and audio_synthesizer is not None
    )
    
    return IntegratedHandTrackingController(
        config=config,
        camera_intrinsics=camera_intrinsics,
        hands_2d=hands_2d,
        hands_3d=hands_3d,
        collision_searcher=collision_searcher,
        audio_mapper=audio_mapper,
        audio_synthesizer=audio_synthesizer
    )


def create_high_performance_controller(
    camera_intrinsics: CameraIntrinsics,
    hands_2d: 'MediaPipeHandsWrapper',
    hands_3d: 'Hand3DProjector',
    collision_searcher: 'CollisionSearcher',
    audio_mapper: 'AudioMapper',
    audio_synthesizer: 'AudioSynthesizer'
) -> IntegratedHandTrackingController:
    """高性能統合コントローラーを作成"""
    config = IntegratedControllerConfig(
        target_fps=60,
        mediapipe_detection_fps=20,  # 高頻度検出
        collision_check_fps=60,
        audio_synthesis_fps=60,
        collision_enabled=True,
        audio_enabled=True,
        audio_cooldown_ms=30.0,  # 短いクールダウン
        enable_prediction=True,
        enable_interpolation=True,
        enable_performance_monitoring=True
    )
    
    return IntegratedHandTrackingController(
        config=config,
        camera_intrinsics=camera_intrinsics,
        hands_2d=hands_2d,
        hands_3d=hands_3d,
        collision_searcher=collision_searcher,
        audio_mapper=audio_mapper,
        audio_synthesizer=audio_synthesizer
    )


def create_integrated_controller(
    config: IntegratedControllerConfig,
    camera_intrinsics: CameraIntrinsics,
    collision_searcher: Optional['CollisionSearcher'] = None,
    audio_mapper: Optional['AudioMapper'] = None,
    audio_synthesizer: Optional['AudioSynthesizer'] = None
) -> IntegratedHandTrackingController:
    """統合コントローラーを作成（簡易版）"""
    from ..hands2d import MediaPipeHandsWrapper
    from ..hands3d import Hand3DProjector
    
    # MediaPipe 2D検出器
    hands_2d = MediaPipeHandsWrapper(
        use_gpu=config.use_gpu_mediapipe,
        max_num_hands=config.max_num_hands,
        min_detection_confidence=config.mediapipe_confidence,
        min_tracking_confidence=config.mediapipe_tracking_confidence
    )
    
    # 3D投影器
    hands_3d = Hand3DProjector(
        camera_intrinsics=camera_intrinsics,
        depth_scale=1000.0,
        min_confidence_3d=0.3
    )
    
    return IntegratedHandTrackingController(
        config=config,
        camera_intrinsics=camera_intrinsics,
        hands_2d=hands_2d,
        hands_3d=hands_3d,
        collision_searcher=collision_searcher,
        audio_mapper=audio_mapper,
        audio_synthesizer=audio_synthesizer
    ) 