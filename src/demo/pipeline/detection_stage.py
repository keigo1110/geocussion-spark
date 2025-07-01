#!/usr/bin/env python3
"""
検出ステージ: 手検出とトラッキング

MediaPipeを使用した2D手検出、3D投影、トラッキングを担当します。
"""

from typing import Optional, List, Dict, Any
import numpy as np
from dataclasses import dataclass

from .base import PipelineStage, StageResult
from ...types import FrameData, Hand2D, Hand3D
from ...detection.hands2d import MediaPipeHandsWrapper
from ...detection.hands3d import DepthInterpolationMethod
from ...detection.tracker import Hand3DTracker, TrackedHand


@dataclass
class DetectionStageConfig:
    """検出ステージの設定"""
    enable_hand_detection: bool = True
    enable_tracking: bool = True
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    use_gpu_mediapipe: bool = False
    max_num_hands: int = 5
    depth_interpolation: DepthInterpolationMethod = DepthInterpolationMethod.LINEAR


@dataclass
class DetectionStageResult(StageResult):
    """検出ステージの処理結果"""
    hands_2d: List[Hand2D] = None
    hands_3d: List[Hand3D] = None
    tracked_hands: List[TrackedHand] = None
    
    def __post_init__(self):
        if self.hands_2d is None:
            self.hands_2d = []
        if self.hands_3d is None:
            self.hands_3d = []
        if self.tracked_hands is None:
            self.tracked_hands = []


class DetectionStage(PipelineStage):
    """検出ステージの実装"""
    
    def __init__(self, config: DetectionStageConfig) -> None:
        """
        初期化
        
        Args:
            config: 検出ステージ設定
        """
        super().__init__(config)
        self.config: DetectionStageConfig = config
        self.hand_detector: Optional[MediaPipeHandsWrapper] = None
        self.hand_projector: Optional[Hand3DProjector] = None
        self.hand_tracker: Optional[Hand3DTracker] = None
        self.camera_intrinsics: Optional[Any] = None  # カメラ内部パラメータ
        
    def initialize(self) -> bool:
        """ステージの初期化"""
        if not self.config.enable_hand_detection:
            self.logger.info("手検出は無効化されています")
            self._initialized = True
            return True
            
        try:
            # MediaPipe手検出器初期化
            self.hand_detector = MediaPipeHandsWrapper(
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                max_num_hands=self.config.max_num_hands,
                use_gpu=self.config.use_gpu_mediapipe
            )
            self.logger.info("MediaPipe手検出器を初期化しました")
            
            # 3D投影器は最初のフレームで初期化
            # （カメラのintrinsicsが必要なため）
            self.logger.info("3D手投影器は最初のフレーム処理時に初期化されます")
            
            # トラッカー初期化
            if self.config.enable_tracking:
                self.hand_tracker = Hand3DTracker(
                    max_assignment_distance=0.2,
                    max_lost_frames=15,
                    min_track_length=3
                )
                self.logger.info("手トラッカーを初期化しました")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"検出ステージの初期化に失敗: {e}")
            return False
    
    def process(self, 
                frame_data: FrameData,
                color_image: Optional[np.ndarray] = None) -> DetectionStageResult:
        """
        手検出を実行
        
        Args:
            frame_data: フレームデータ
            color_image: RGB画像（MediaPipe用）
            
        Returns:
            検出結果
        """
        if not self._initialized:
            return DetectionStageResult(
                success=False,
                error_message="Stage not initialized"
            )
        
        if not self.config.enable_hand_detection:
            return DetectionStageResult(success=True)
        
        try:
            # 2D手検出
            hands_2d = []
            if self.hand_detector and color_image is not None:
                hands_2d = self.hand_detector.detect(color_image)
                if hands_2d:
                    self.logger.debug(f"{len(hands_2d)}個の手を検出しました")
            
            # 3D投影器の遅延初期化
            if self.hand_projector is None and self.camera_intrinsics is not None:
                from ...detection.hands3d import Hand3DProjector
                self.hand_projector = Hand3DProjector(
                    camera_intrinsics=self.camera_intrinsics,
                    interpolation_method=self.config.depth_interpolation
                )
                self.logger.info("3D手投影器を初期化しました")
            
            # 3D投影
            hands_3d = []
            if self.hand_projector and hands_2d and frame_data.depth_frame is not None:
                # 深度データを取得
                depth_data = frame_data.depth_frame.get_data() if hasattr(frame_data.depth_frame, 'get_data') else frame_data.depth_frame
                
                for hand_2d in hands_2d:
                    hand_3d = self.hand_projector.project(
                        hand_2d,
                        depth_data,
                        self.camera_intrinsics
                    )
                    if hand_3d:
                        hands_3d.append(hand_3d)
                self.logger.debug(f"{len(hands_3d)}個の手を3D投影しました")
            
            # トラッキング
            tracked_hands = []
            if self.hand_tracker and hands_3d:
                tracked_hands = self.hand_tracker.update(hands_3d)
                self.logger.debug(f"{len(tracked_hands)}個の手をトラッキング中")
            elif hands_3d and not self.config.enable_tracking:
                # トラッキング無効時は3D手をそのまま使用
                tracked_hands = [
                    TrackedHand(
                        id=i,
                        hand=hand,
                        confidence=1.0,
                        missing_frames=0,
                        is_new=True
                    )
                    for i, hand in enumerate(hands_3d)
                ]
            
            return DetectionStageResult(
                success=True,
                hands_2d=hands_2d,
                hands_3d=hands_3d,
                tracked_hands=tracked_hands
            )
            
        except Exception as e:
            self.logger.error(f"検出処理エラー: {e}")
            return DetectionStageResult(
                success=False,
                error_message=str(e)
            )
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.hand_detector:
            self.hand_detector.close()
            self.hand_detector = None
        if self.hand_projector:
            self.hand_projector = None
        if self.hand_tracker:
            self.hand_tracker = None
        self._initialized = False
        self.logger.info("検出ステージをクリーンアップしました")