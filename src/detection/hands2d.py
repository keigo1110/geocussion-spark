#!/usr/bin/env python3
"""
MediaPipe Hands 2D検出ラッパー
GPU対応・batch処理・パフォーマンス最適化対応
"""

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


class HandednessType(Enum):
    """手の左右判定"""
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


@dataclass
class HandLandmark:
    """手のランドマーク座標"""
    x: float  # 0-1の正規化座標
    y: float  # 0-1の正規化座標
    z: float  # 深度情報（相対値）
    visibility: float = 1.0  # 可視性スコア


@dataclass
class HandDetectionResult:
    """手検出結果"""
    id: str  # 手のID（トラッキング用）
    landmarks: List[HandLandmark]
    handedness: HandednessType
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    timestamp_ms: float
    
    @property
    def center_point(self) -> Tuple[float, float]:
        """手の中心点を計算"""
        if not self.landmarks:
            return (0.0, 0.0)
        avg_x = sum(lm.x for lm in self.landmarks) / len(self.landmarks)
        avg_y = sum(lm.y for lm in self.landmarks) / len(self.landmarks)
        return (avg_x, avg_y)
    
    @property
    def palm_center(self) -> Tuple[float, float]:
        """手のひら中心を計算（ランドマーク0, 5, 9, 13, 17の平均）"""
        if len(self.landmarks) < 21:
            return self.center_point
        palm_indices = [0, 5, 9, 13, 17]  # 手首・各指の付け根
        avg_x = sum(self.landmarks[i].x for i in palm_indices) / len(palm_indices)
        avg_y = sum(self.landmarks[i].y for i in palm_indices) / len(palm_indices)
        return (avg_x, avg_y)


class MediaPipeHandsWrapper:
    """MediaPipe Hands ラッパークラス"""
    
    def __init__(
        self,
        use_gpu: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        初期化
        
        Args:
            use_gpu: GPU使用フラグ
            max_num_hands: 最大検出手数
            min_detection_confidence: 検出信頼度閾値
            min_tracking_confidence: トラッキング信頼度閾値
            model_complexity: モデル複雑度（0:lite, 1:full）
        """
        self.use_gpu = use_gpu
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.is_initialized = False
        
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
            print("Warning: MediaPipe not available, using mock implementation")
            return False
        
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            # GPU/CPU設定
            if self.use_gpu:
                # GPU使用時の設定
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    model_complexity=self.model_complexity
                )
            else:
                # CPU使用時の設定
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    model_complexity=0  # CPU時はliteモデル使用
                )
            
            self.is_initialized = True
            print(f"MediaPipe Hands initialized ({'GPU' if self.use_gpu else 'CPU'} mode)")
            return True
            
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            return False
    
    def detect_hands(self, image: np.ndarray) -> List[HandDetectionResult]:
        """
        手検出実行
        
        Args:
            image: 入力画像 (BGR format)
            
        Returns:
            検出された手のリスト
        """
        if not self.is_initialized:
            return []
        
        start_time = time.perf_counter()
        
        try:
            # BGR -> RGB 変換
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # MediaPipe処理
            results = self.hands.process(rgb_image)
            
            # 結果変換
            hand_results = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_result = self._convert_to_hand_result(
                        hand_landmarks, 
                        handedness, 
                        image.shape[:2]
                    )
                    if hand_result:
                        hand_results.append(hand_result)
            
            # 統計更新
            detection_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(detection_time, len(hand_results))
            
            return hand_results
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return []
    
    def detect_hands_batch(self, images: List[np.ndarray]) -> List[List[HandDetectionResult]]:
        """
        バッチ処理での手検出
        
        Args:
            images: 入力画像リスト
            
        Returns:
            各画像の検出結果リスト
        """
        batch_results = []
        batch_start_time = time.perf_counter()
        
        for image in images:
            results = self.detect_hands(image)
            batch_results.append(results)
        
        batch_time = (time.perf_counter() - batch_start_time) * 1000
        print(f"Batch processing: {len(images)} images in {batch_time:.1f}ms "
              f"({batch_time/len(images):.1f}ms/image)")
        
        return batch_results
    
    def _convert_to_hand_result(
        self, 
        landmarks, 
        handedness, 
        image_shape: Tuple[int, int]
    ) -> Optional[HandDetectionResult]:
        """MediaPipe結果をHandDetectionResultに変換"""
        try:
            height, width = image_shape
            
            # ランドマーク変換
            hand_landmarks = []
            for landmark in landmarks.landmark:
                hand_landmarks.append(HandLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=getattr(landmark, 'visibility', 1.0)
                ))
            
            # 手の左右判定
            handedness_label = handedness.classification[0].label
            handedness_type = HandednessType.LEFT if handedness_label == "Left" else HandednessType.RIGHT
            confidence = handedness.classification[0].score
            
            # バウンディングボックス計算
            x_coords = [lm.x * width for lm in hand_landmarks]
            y_coords = [lm.y * height for lm in hand_landmarks]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # 手のID生成 (handedness + timestamp)
            timestamp = time.perf_counter() * 1000
            hand_id = f"{handedness_type.value}_{int(timestamp)}"
            
            return HandDetectionResult(
                id=hand_id,
                landmarks=hand_landmarks,
                handedness=handedness_type,
                confidence=confidence,
                bounding_box=bounding_box,
                timestamp_ms=timestamp
            )
            
        except Exception as e:
            print(f"Result conversion error: {e}")
            return None
    
    def draw_landmarks(
        self, 
        image: np.ndarray, 
        hand_result: HandDetectionResult,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        検出結果を画像に描画
        
        Args:
            image: 描画対象画像
            hand_result: 手検出結果
            draw_connections: 接続線を描画するか
            
        Returns:
            描画済み画像
        """
        if not self.is_initialized or not hand_result.landmarks:
            return image
        
        try:
            height, width = image.shape[:2]
            
            # ランドマーク描画
            for i, landmark in enumerate(hand_result.landmarks):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # 点描画
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
                # インデックス描画
                cv2.putText(image, str(i), (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # 接続線描画
            if draw_connections and MEDIAPIPE_AVAILABLE:
                # MediaPipeの接続定義を使用
                connections = self.mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    start_landmark = hand_result.landmarks[start_idx]
                    end_landmark = hand_result.landmarks[end_idx]
                    
                    start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
                    end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
                    
                    cv2.line(image, start_point, end_point, (255, 0, 0), 2)
            
            # 手の情報表示
            bbox = hand_result.bounding_box
            cv2.rectangle(image, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 2)
            
            cv2.putText(image, f"{hand_result.handedness.value} ({hand_result.confidence:.2f})",
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return image
            
        except Exception as e:
            print(f"Drawing error: {e}")
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