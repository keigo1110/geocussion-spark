#!/usr/bin/env python3
"""
MediaPipe Hands 2D検出ラッパー
GPU対応・batch処理・パフォーマンス最適化対応
ROI トラッキングによるMediaPipe実行スキップ機能
"""
from __future__ import annotations  # postpone evaluation of type hints

import time
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    # テスト用モック定義
    class mp:
        class solutions:
            class hands:
                Hands = None
                HAND_CONNECTIONS = []
            class drawing_utils:
                draw_landmarks = lambda *args: None
                DrawingSpec = lambda *args: None

from .. import get_logger
from ..data_types import HandednessType, HandLandmark, HandROI, HandDetectionResult, ROITrackingStats

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# OpenCV compatibility shim
# Some OpenCV builds (e.g. pip wheels without legacy modules) do not provide the
# top-level attribute ``cv2.Tracker`` that is referenced in type annotations
# below.  The absence of this symbol triggers an ``AttributeError`` *during
# module import* (because type annotations are evaluated eagerly in Python <3.11
# unless `from __future__ import annotations` is used).  We therefore (1) delay
# evaluation via the future import above and (2) register a lightweight fallback
# alias so that dynamic checks like ``hasattr(cv2, 'Tracker')`` succeed later.
# -----------------------------------------------------------------------------
if not hasattr(cv2, "Tracker"):
    # Try to reuse the implementation living in the legacy namespace, or else
    # create a dummy placeholder to satisfy type-checking / isinstance tests.
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "Tracker"):
        cv2.Tracker = cv2.legacy.Tracker  # type: ignore
    else:
        class _DummyTracker:  # type: ignore
            """Minimal placeholder when OpenCV is built without the legacy tracker."""
            pass
        cv2.Tracker = _DummyTracker  # type: ignore


class MediaPipeHandsWrapper:
    """MediaPipe Hands ラッパークラス（ROIトラッキング対応）"""
    
    def __init__(
        self,
        use_gpu: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        # ROI トラッキング設定
        enable_roi_tracking: bool = True,
        tracker_type: str = "KCF",           # "KCF" or "MOSSE" 
        skip_interval: int = 4,              # N フレームに1回 MediaPipe実行
        roi_confidence_threshold: float = 0.6,
        max_tracking_age: int = 15,
        roi_expansion_factor: float = 1.2
    ):
        """
        初期化
        
        Args:
            use_gpu: GPU使用フラグ
            max_num_hands: 最大検出手数
            min_detection_confidence: 検出信頼度閾値
            min_tracking_confidence: トラッキング信頼度閾値
            model_complexity: モデル複雑度（0:lite, 1:full）
            enable_roi_tracking: ROIトラッキングの有効化
            tracker_type: トラッカー種類 ("KCF", "MOSSE")
            skip_interval: MediaPipe実行間隔 (フレーム数)
            roi_confidence_threshold: ROIトラッキング信頼度閾値
            max_tracking_age: 強制MediaPipe実行までのフレーム数
            roi_expansion_factor: ROI拡張係数
        """
        self.use_gpu = use_gpu
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # ROI トラッキング設定
        self.enable_roi_tracking = enable_roi_tracking
        self.tracker_type = tracker_type
        self.skip_interval = skip_interval
        self.roi_confidence_threshold = roi_confidence_threshold
        self.max_tracking_age = max_tracking_age
        self.roi_expansion_factor = roi_expansion_factor
        
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.is_initialized = False
        
        # ROI トラッキング状態
        self.current_trackers: Dict[str, cv2.Tracker] = {}
        self.current_rois: Dict[str, HandROI] = {}
        self.frame_count = 0
        self.last_mediapipe_frame = -1
        self.roi_tracking_stats = ROITrackingStats()
        
        # パフォーマンス統計
        self.performance_stats = {
            'total_frames': 0,
            'detection_time_ms': 0.0,
            'avg_detection_time_ms': 0.0,
            'hands_detected': 0,
            'detection_rate': 0.0
        }
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """MediaPipe初期化"""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available, using mock implementation")
            return False
        
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            def _create_hands_instance(use_gpu: bool):
                return self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    model_complexity=self.model_complexity if use_gpu else 0  # CPU時は lite
                )

            # 1) GPU でトライ
            init_success = False
            try:
                self.hands = _create_hands_instance(use_gpu=self.use_gpu)
                init_success = True
            except (RuntimeError, ValueError, Exception) as e:
                if self.use_gpu:
                    logger.warning(f"MediaPipe GPU initialization failed: {e}. Falling back to CPU mode.")
                else:
                    logger.warning(f"MediaPipe CPU initialization error: {e}")

            # 2) CPU フォールバック
            if not init_success:
                try:
                    self.hands = _create_hands_instance(use_gpu=False)
                    self.use_gpu = False  # 状態を更新
                    init_success = True
                except Exception as e:
                    logger.error(f"MediaPipe CPU fallback also failed: {e}")
                    init_success = False

            if init_success:
                self.is_initialized = True
                roi_status = f" (ROI tracking: {'enabled' if self.enable_roi_tracking else 'disabled'})"
                logger.info(f"MediaPipe Hands initialized ({'GPU' if self.use_gpu else 'CPU'} mode){roi_status}")
                return True
            else:
                return False
            
        except ImportError as e:
            # MediaPipeモジュールのインポートエラー
            logger.error(f"MediaPipe import error: {e}")
            return False
        except (RuntimeError, OSError) as e:
            # GPU/ハードウェアエラー（復旧可能）
            logger.warning(f"MediaPipe hardware initialization error: {e}")
            return False
        except Exception as e:
            # その他の予期しないエラー
            logger.error(f"Unexpected MediaPipe initialization error: {e}")
            return False
    
    def detect_hands(self, image: np.ndarray) -> List[HandDetectionResult]:
        """
        手検出実行（ROI トラッキング対応）
        
        Args:
            image: 入力画像 (BGR format)
            
        Returns:
            検出された手のリスト
        """
        if not self.is_initialized:
            return []
        
        self.roi_tracking_stats.total_frames += 1
        self.frame_count += 1
        
        # ROI トラッキングが有効で、MediaPipe をスキップできる場合
        if self.enable_roi_tracking and not self._should_run_mediapipe():
            return self._update_roi_tracking(image)
        
        # MediaPipe を実行
        return self._run_mediapipe_detection(image)
    
    def _should_run_mediapipe(self) -> bool:
        """MediaPipe手検出を実行すべきかを判定"""
        # 初回は必ずMediaPipe実行
        if self.frame_count <= 1:
            logger.debug("First frame: running MediaPipe")
            return True
        
        # 強制実行間隔チェック
        frames_since_mediapipe = self.frame_count - self.last_mediapipe_frame
        if frames_since_mediapipe >= self.skip_interval:
            logger.debug(f"Skip interval reached ({frames_since_mediapipe} >= {self.skip_interval}): running MediaPipe")
            return True
        
        # 古いトラッキングは強制更新
        if frames_since_mediapipe >= self.max_tracking_age:
            logger.debug(f"Max tracking age reached ({frames_since_mediapipe} >= {self.max_tracking_age}): running MediaPipe")
            return True
        
        # 有効なトラッカーがあれば MediaPipe をスキップ
        if self.current_trackers:
            logger.debug(f"Active trackers: {len(self.current_trackers)}, skipping MediaPipe")
            return False
        
        # トラッカーがない場合はMediaPipe実行
        logger.debug("No active trackers: running MediaPipe")
        return True
    
    def _run_mediapipe_detection(self, image: np.ndarray) -> List[HandDetectionResult]:
        """MediaPipe 手検出を実行"""
        start_time = time.perf_counter()
        
        self.last_mediapipe_frame = self.frame_count
        self.roi_tracking_stats.mediapipe_executions += 1
        
        try:
            # BGR -> RGB 変換
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # MediaPipe処理
            results = self.hands.process(rgb_image)
            
            # 結果変換
            hand_results = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_result = self._convert_to_hand_result(
                        hand_landmarks, 
                        handedness, 
                        image.shape[:2],
                        hand_id=f"hand_{i}"
                    )
                    if hand_result:
                        hand_results.append(hand_result)
            
            # ROI トラッキング用のトラッカーを更新
            if self.enable_roi_tracking:
                self._update_trackers_from_mediapipe(image, hand_results)
            
            # 統計更新
            detection_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(detection_time, len(hand_results))
            self.roi_tracking_stats.total_mediapipe_time_ms += detection_time
            
            return hand_results
            
        except cv2.error as e:
            # OpenCV画像処理エラー
            logger.warning(f"OpenCV error during hand detection: {e}")
            return []
        except (RuntimeError, ValueError) as e:
            # MediaPipe処理エラー
            logger.warning(f"MediaPipe processing error: {e}")
            return []
        except Exception as e:
            # その他の予期しないエラー
            logger.error(f"Unexpected MediaPipe hand detection error: {e}")
            return []
    
    def _update_roi_tracking(self, image: np.ndarray) -> List[HandDetectionResult]:
        """ROI トラッキングで手位置を更新"""
        start_time = time.perf_counter()
        
        hand_results = []
        failed_trackers = []
        
        for hand_id, tracker in self.current_trackers.items():
            success, bbox = tracker.update(image)
            
            if success:
                x, y, w, h = map(int, bbox)
                confidence = self._estimate_tracking_confidence(image, bbox)
                
                if confidence >= self.roi_confidence_threshold:
                    # トラッキング成功
                    roi = HandROI(
                        x=x, y=y, width=w, height=h,
                        confidence=confidence,
                        last_updated_frame=self.frame_count,
                        hand_id=hand_id
                    )
                    self.current_rois[hand_id] = roi
                    
                    # HandDetectionResult を生成（簡略版）
                    hand_result = self._create_tracked_hand_result(roi, hand_id)
                    if hand_result:
                        hand_results.append(hand_result)
                    
                    self.roi_tracking_stats.tracking_successes += 1
                    logger.debug(f"ROI tracking success for {hand_id}: confidence={confidence:.3f}")
                else:
                    # 信頼度不足
                    failed_trackers.append(hand_id)
                    logger.debug(f"ROI tracking failed for {hand_id}: low confidence {confidence:.3f}")
            else:
                # トラッキング失敗
                failed_trackers.append(hand_id)
                logger.debug(f"ROI tracking failed for {hand_id}: tracker.update() failed")
        
        # 失敗したトラッカーを削除
        for hand_id in failed_trackers:
            if hand_id in self.current_trackers:
                del self.current_trackers[hand_id]
            if hand_id in self.current_rois:
                del self.current_rois[hand_id]
            self.roi_tracking_stats.tracking_failures += 1
        
        # 統計更新
        tracking_time = (time.perf_counter() - start_time) * 1000
        self.roi_tracking_stats.total_tracking_time_ms += tracking_time
        
        return hand_results
    
    def _update_trackers_from_mediapipe(self, image: np.ndarray, hand_results: List[HandDetectionResult]):
        """MediaPipe結果からトラッカーを更新"""
        # 既存のトラッカーをクリア
        self.current_trackers.clear()
        self.current_rois.clear()
        
        for hand_result in hand_results:
            hand_id = hand_result.id
            x, y, w, h = hand_result.bounding_box
            
            # ROI を少し拡張
            center_x, center_y = x + w // 2, y + h // 2
            expanded_w = int(w * self.roi_expansion_factor)
            expanded_h = int(h * self.roi_expansion_factor)
            expanded_x = max(0, center_x - expanded_w // 2)
            expanded_y = max(0, center_y - expanded_h // 2)
            
            # 画像境界でクランプ
            height, width = image.shape[:2]
            expanded_w = min(expanded_w, width - expanded_x)
            expanded_h = min(expanded_h, height - expanded_y)
            
            if expanded_w > 20 and expanded_h > 20:  # 最小サイズチェック
                # トラッカーを初期化
                tracker = self._create_tracker()
                if tracker and tracker.init(image, (expanded_x, expanded_y, expanded_w, expanded_h)):
                    self.current_trackers[hand_id] = tracker
                    self.current_rois[hand_id] = HandROI(
                        x=expanded_x, y=expanded_y, 
                        width=expanded_w, height=expanded_h,
                        hand_id=hand_id,
                        last_updated_frame=self.frame_count
                    )
                    logger.debug(f"Initialized ROI tracker for {hand_id}: ({expanded_x}, {expanded_y}, {expanded_w}, {expanded_h})")
    
    def _create_tracker(self) -> Optional[cv2.Tracker]:
        """OpenCVトラッカーを作成"""
        try:
            if self.tracker_type == "KCF":
                return cv2.TrackerKCF_create()
            elif self.tracker_type == "MOSSE":
                return cv2.legacy.TrackerMOSSE_create()
            elif self.tracker_type == "CSRT":
                return cv2.TrackerCSRT_create()
            else:
                logger.warning(f"Unknown tracker type: {self.tracker_type}, using KCF")
                return cv2.TrackerKCF_create()
        except (AttributeError, cv2.error) as e:
            # OpenCVトラッカー作成エラー
            logger.warning(f"Failed to create tracker {self.tracker_type}: {e}")
            return None
        except Exception as e:
            # その他の予期しないエラー
            logger.error(f"Unexpected tracker creation error: {e}")
            return None
    
    def _estimate_tracking_confidence(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
        """トラッキング信頼度を推定"""
        try:
            x, y, w, h = map(int, bbox)
            
            # 画像境界チェック
            height, width = image.shape[:2]
            if x < 0 or y < 0 or x + w > width or y + h > height or w <= 0 or h <= 0:
                return 0.0
            
            # ROI領域の画像統計を使用
            roi_region = image[y:y+h, x:x+w]
            if roi_region.size == 0:
                return 0.0
            
            # 画像の分散を信頼度の指標として使用
            variance = np.var(roi_region)
            normalized_confidence = min(1.0, variance / 1000.0)
            
            return max(0.0, normalized_confidence)
            
        except (IndexError, ValueError) as e:
            # 画像境界エラー
            logger.debug(f"Tracking confidence estimation error: {e}")
            return 0.0
        except Exception as e:
            # その他の予期しないエラー
            logger.warning(f"Unexpected tracking confidence error: {e}")
            return 0.0
    
    def _create_tracked_hand_result(self, roi: HandROI, hand_id: str) -> Optional[HandDetectionResult]:
        """ROI から HandDetectionResult を生成（簡略版）"""
        try:
            # 簡略版のランドマーク（手の中心のみ）
            center_x_norm = (roi.x + roi.width / 2) / 640.0  # 仮の画像幅
            center_y_norm = (roi.y + roi.height / 2) / 480.0  # 仮の画像高さ
            
            landmarks = [
                HandLandmark(x=center_x_norm, y=center_y_norm, z=0.0)
                for _ in range(21)  # MediaPipe の標準ランドマーク数
            ]
            
            return HandDetectionResult(
                id=hand_id,
                landmarks=landmarks,
                handedness=HandednessType.UNKNOWN,
                confidence=roi.confidence,
                bounding_box=roi.bbox,
                timestamp_ms=time.time() * 1000,
                is_tracked=True
            )
            
        except (IndexError, ValueError) as e:
            # ランドマーク生成エラー
            logger.debug(f"Failed to create tracked hand result: {e}")
            return None
        except Exception as e:
            # その他の予期しないエラー
            logger.warning(f"Unexpected tracked hand result creation error: {e}")
            return None

    def detect_hands_batch(self, images: List[np.ndarray]) -> List[List[HandDetectionResult]]:
        """
        バッチ手検出（ROI トラッキング未対応）
        
        Args:
            images: 入力画像リスト
            
        Returns:
            各画像の検出結果リスト
        """
        if not self.is_initialized:
            return [[] for _ in images]
        
        results = []
        for image in images:
            # バッチ処理ではROIトラッキングを無効化
            hand_results = self._run_mediapipe_detection(image)
            results.append(hand_results)
        
        return results
    
    def _convert_to_hand_result(
        self, 
        landmarks, 
        handedness, 
        image_shape: Tuple[int, int],
        hand_id: str = ""
    ) -> Optional[HandDetectionResult]:
        """MediaPipe結果をHandDetectionResultに変換"""
        try:
            height, width = image_shape
            
            # ランドマーク変換
            hand_landmarks = []
            x_coords, y_coords = [], []
            
            for landmark in landmarks.landmark:
                hand_landmark = HandLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=getattr(landmark, 'visibility', 1.0)
                )
                hand_landmarks.append(hand_landmark)
                x_coords.append(landmark.x * width)
                y_coords.append(landmark.y * height)
            
            if not x_coords or not y_coords:
                return None
            
            # バウンディングボックス計算
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            bbox_x, bbox_y = int(min_x), int(min_y)
            bbox_w, bbox_h = int(max_x - min_x), int(max_y - min_y)
            
            # 手の左右判定
            handedness_label = handedness.classification[0].label
            hand_type = HandednessType.LEFT if handedness_label == "Left" else HandednessType.RIGHT
            
            # 信頼度
            confidence = handedness.classification[0].score
            
            # ID生成
            if not hand_id:
                hand_id = f"{hand_type.value.lower()}_{int(time.time() * 1000) % 10000}"
            
            return HandDetectionResult(
                id=hand_id,
                landmarks=hand_landmarks,
                handedness=hand_type,
                confidence=confidence,
                bounding_box=(bbox_x, bbox_y, bbox_w, bbox_h),
                timestamp_ms=time.time() * 1000,
                is_tracked=False
            )
            
        except (IndexError, AttributeError, ValueError) as e:
            # MediaPipe結果変換エラー
            logger.debug(f"MediaPipe result conversion error: {e}")
            return None
        except Exception as e:
            # その他の予期しないエラー
            logger.warning(f"Unexpected MediaPipe conversion error: {e}")
            return None

    def get_roi_tracking_stats(self) -> ROITrackingStats:
        """ROI トラッキング統計を取得"""
        return self.roi_tracking_stats
    
    def reset_roi_tracking(self):
        """ROI トラッキング状態をリセット"""
        self.current_trackers.clear()
        self.current_rois.clear()
        self.frame_count = 0
        self.last_mediapipe_frame = -1
        self.roi_tracking_stats = ROITrackingStats()
        logger.info("ROI tracking reset")

    def draw_landmarks(
        self, 
        image: np.ndarray, 
        hand_result: HandDetectionResult,
        draw_connections: bool = True
    ) -> np.ndarray:
        """ランドマーク描画"""
        if not self.is_initialized:
            return image

        try:
            height, width = image.shape[:2]

            # ROI トラッキング結果: バウンディングボックスのみ
            if hand_result.is_tracked:
                x, y, w, h = hand_result.bounding_box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(image, f"Tracked {hand_result.confidence:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                return image

            # --- 軽量 OpenCV 描画 ---
            pts = []
            for lm in hand_result.landmarks:
                cx, cy = int(lm.x * width), int(lm.y * height)
                pts.append((cx, cy))
                cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

            # 簡易骨格線（親指とそれ以外をざっくり接続）
            basic_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),   # 親指
                (5, 6), (6, 7), (7, 8),            # 人差し指
                (9,10), (10,11), (11,12),          # 中指
                (13,14), (14,15), (15,16),         # 薬指
                (17,18), (18,19), (19,20),         # 小指
                (0,5), (5,9), (9,13), (13,17), (17,0)  # 手のひら
            ] if draw_connections and len(pts) == 21 else []

            for a, b in basic_connections:
                cv2.line(image, pts[a], pts[b], (0, 255, 0), 1)

            # ラベル
            x, y, w_box, h_box = hand_result.bounding_box
            label = f"{hand_result.handedness.value} {hand_result.id}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        except Exception as e:
            logger.warning(f"Landmark drawing error: {e}")
        
        return image

    def _update_stats(self, detection_time_ms: float, num_hands: int) -> None:
        """統計情報を更新"""
        self.performance_stats['total_frames'] += 1
        self.performance_stats['detection_time_ms'] = detection_time_ms
        self.performance_stats['hands_detected'] += num_hands
        
        # 移動平均計算
        total_frames = self.performance_stats['total_frames']
        prev_avg = self.performance_stats['avg_detection_time_ms']
        self.performance_stats['avg_detection_time_ms'] = (
            (prev_avg * (total_frames - 1) + detection_time_ms) / total_frames
        )
        
        # 検出率計算
        self.performance_stats['detection_rate'] = (
            self.performance_stats['hands_detected'] / total_frames
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        return self.performance_stats.copy()
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        self.performance_stats = {
            'total_frames': 0,
            'detection_time_ms': 0.0,
            'avg_detection_time_ms': 0.0,
            'hands_detected': 0,
            'detection_rate': 0.0
        }
    
    def update_parameters(self, **kwargs) -> None:
        """検出パラメータを動的更新"""
        restart_needed = False
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['use_gpu', 'max_num_hands', 'model_complexity']:
                    restart_needed = True
        
        if restart_needed and self.is_initialized:
            self.close()
            self._initialize()
    
    def close(self) -> None:
        """リソース解放"""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        self.is_initialized = False
    
    def __enter__(self):
        """コンテキストマネージャー: 開始"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー: 終了"""
        self.close()


# ユーティリティ関数

def create_mock_hand_result() -> HandDetectionResult:
    """テスト用のモック手検出結果を作成"""
    landmarks = []
    for i in range(21):  # MediaPipeの手ランドマークは21点
        landmarks.append(HandLandmark(
            x=0.5 + 0.1 * np.sin(i * 0.3),
            y=0.5 + 0.1 * np.cos(i * 0.3),
            z=0.0,
            visibility=1.0
        ))
    
    timestamp = time.perf_counter() * 1000
    return HandDetectionResult(
        id=f"Mock_{int(timestamp)}",
        landmarks=landmarks,
        handedness=HandednessType.RIGHT,
        confidence=0.95,
        bounding_box=(100, 100, 200, 200),
        timestamp_ms=timestamp
    )


def filter_hands_by_confidence(
    hands: List[HandDetectionResult], 
    min_confidence: float = 0.8
) -> List[HandDetectionResult]:
    """信頼度でフィルタリング"""
    return [hand for hand in hands if hand.confidence >= min_confidence]


def get_dominant_hand(hands: List[HandDetectionResult]) -> Optional[HandDetectionResult]:
    """最も信頼度の高い手を取得"""
    if not hands:
        return None
    return max(hands, key=lambda h: h.confidence) 