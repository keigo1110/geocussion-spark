#!/usr/bin/env python3
"""
60fps対応カルマンフィルタベースの3D手トラッカー
MediaPipe検出結果を60fpsで補間し、音響・衝突パイプラインに滑らかな座標を供給
"""

import time
import uuid
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment
import threading
from collections import deque

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hands3d import Hand3DResult, Hand3DLandmark

from ..data_types import HandednessType, TrackingState
from .tracker import KalmanFilterConfig, TrackedHand
from src import get_logger

logger = get_logger(__name__)


class HandTrackingState(Enum):
    """手トラッキング状態のFSM"""
    SEARCHING = "searching"      # 手を探索中
    DETECTING = "detecting"      # MediaPipe検出中
    TRACKING = "tracking"        # Kalman追跡中
    LOST = "lost"               # 手を見失った
    RECOVERED = "recovered"      # 手を再発見


@dataclass
class HighFrequencyKalmanConfig:
    """60fps対応カルマンフィルタ設定"""
    # 基本設定
    target_fps: int = 60
    mediapipe_fps: int = 15  # MediaPipeの実際の実行頻度
    
    # プロセスノイズ（60fps用に調整）
    process_noise_position: float = 0.02  # m（30fps時の0.05から調整）
    process_noise_velocity: float = 0.05  # m/s（30fps時の0.1から調整）
    process_noise_acceleration: float = 0.1  # m/s²（新規追加）
    
    # 観測ノイズ
    observation_noise: float = 0.03  # m（MediaPipe精度考慮）
    
    # 初期共分散
    initial_position_variance: float = 0.05  # m²
    initial_velocity_variance: float = 0.2   # (m/s)²
    initial_acceleration_variance: float = 0.5  # (m/s²)²
    
    # 補間設定
    enable_interpolation: bool = True
    interpolation_confidence_threshold: float = 0.7
    max_interpolation_frames: int = 8  # 最大補間フレーム数
    
    # 予測設定
    enable_prediction: bool = True
    prediction_horizon_ms: float = 33.3  # 2フレーム先まで予測（60fps）


@dataclass
class HighFrequencyTrackedHand:
    """60fps対応トラッキング済み手情報"""
    id: str
    handedness: HandednessType
    fsm_state: HandTrackingState
    tracking_state: TrackingState
    
    # 位置・速度・加速度（60fps更新）
    position: np.ndarray  # 3D位置 [x, y, z]
    velocity: np.ndarray  # 3D速度 [vx, vy, vz]
    acceleration: np.ndarray  # 3D加速度 [ax, ay, az]
    
    # 信頼度
    confidence_2d: float
    confidence_3d: float
    confidence_tracking: float
    confidence_interpolation: float  # 補間信頼度
    
    # トラッキング履歴
    last_seen_time: float
    last_detection_time: float  # 最後のMediaPipe検出時刻
    track_length: int
    lost_frames: int
    interpolation_frames: int  # 連続補間フレーム数
    
    # 手のサイズとジェスチャー
    hand_size: float
    landmarks_3d: Optional[List['Hand3DLandmark']] = None
    palm_normal: Optional[np.ndarray] = None
    
    # 拡張カルマンフィルタ状態（9次元：位置・速度・加速度）
    state_vector: np.ndarray = field(init=False)
    covariance_matrix: np.ndarray = field(init=False)
    
    # 履歴バッファ（予測精度向上）
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __post_init__(self):
        # 拡張状態ベクトル初期化 [x, y, z, vx, vy, vz, ax, ay, az]
        self.state_vector = np.array([
            self.position[0], self.position[1], self.position[2],
            self.velocity[0], self.velocity[1], self.velocity[2],
            self.acceleration[0], self.acceleration[1], self.acceleration[2]
        ])
        
        # 拡張共分散行列初期化
        self.covariance_matrix = np.eye(9) * 0.1
        
        # 履歴初期化
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
    
    @property
    def speed(self) -> float:
        """速度の大きさ"""
        return np.linalg.norm(self.velocity)
    
    @property
    def acceleration_magnitude(self) -> float:
        """加速度の大きさ"""
        return np.linalg.norm(self.acceleration)
    
    @property
    def is_moving(self) -> bool:
        """動いているかどうか（60fps用に閾値調整）"""
        return self.speed > 0.03  # 3cm/s 以上で動いているとみなす
    
    @property
    def is_accelerating(self) -> bool:
        """加速しているかどうか"""
        return self.acceleration_magnitude > 0.1  # 0.1m/s² 以上で加速中
    
    @property
    def palm_center(self) -> np.ndarray:
        """手のひら中心位置"""
        return self.position
    
    @property
    def hand_id(self) -> str:
        """手のID"""
        return self.id
    
    @property
    def is_interpolated(self) -> bool:
        """現在の状態が補間によるものか"""
        return self.interpolation_frames > 0
    
    @property
    def bounding_box(self) -> tuple:
        """バウンディングボックス（互換性のため）"""
        # 3D位置から推定バウンディングボックスを作成
        # 手のサイズを使用して適切なバウンディングボックスを計算
        size = int(self.hand_size * 1000)  # メートルからピクセル相当に変換
        center_x = int(self.position[0] * 100)  # 適当な座標変換
        center_y = int(self.position[1] * 100)
        
        # 最小サイズを保証
        size = max(size, 40)
        
        x = max(0, center_x - size // 2)
        y = max(0, center_y - size // 2)
        w = size
        h = size
        
        return (x, y, w, h)
    
    @property
    def is_tracked(self) -> bool:
        """60fps追跡による結果かどうか（互換性のため）"""
        return True  # 60fps追跡システムの結果は常にトラッキング済み
    
    @property
    def landmarks(self) -> List:
        """ランドマーク情報（互換性のため）"""
        # 簡易ランドマーク：手の中心のみ
        from ..data_types import HandLandmark
        center_landmark = HandLandmark(
            x=0.5,  # 正規化座標
            y=0.5,
            z=0.0,
            visibility=1.0
        )
        # MediaPipeの標準21点ランドマーク
        return [center_landmark for _ in range(21)]
    
    @property
    def timestamp_ms(self) -> float:
        """タイムスタンプ（ミリ秒）（互換性のため）"""
        return self.last_seen_time * 1000
    
    @property
    def confidence(self) -> float:
        """信頼度（互換性のため）"""
        return self.confidence_tracking 


class HighFrequencyHand3DTracker:
    """60fps対応カルマンフィルタベースの3D手トラッカー"""
    
    def __init__(
        self,
        config: HighFrequencyKalmanConfig = None,
        max_lost_frames: int = 30,  # 60fps対応で増加
        min_track_length: int = 10,  # 60fps対応で増加
        max_assignment_distance: float = 0.25,  # 25cm
    ):
        """
        初期化
        
        Args:
            config: 60fps対応カルマンフィルタ設定
            max_lost_frames: 最大消失フレーム数
            min_track_length: 最小トラック長
            max_assignment_distance: 最大割り当て距離
        """
        self.config = config or HighFrequencyKalmanConfig()
        self.max_lost_frames = max_lost_frames
        self.min_track_length = min_track_length
        self.max_assignment_distance = max_assignment_distance
        
        # フレーム間隔計算
        self.dt = 1.0 / self.config.target_fps  # 60fps = 16.67ms
        self.mediapipe_dt = 1.0 / self.config.mediapipe_fps  # MediaPipe間隔
        
        # トラッキング対象
        self.tracked_hands: Dict[str, HighFrequencyTrackedHand] = {}
        
        # 拡張カルマンフィルタ行列（9次元）
        self._setup_extended_kalman_matrices()
        
        # 統計情報
        self.performance_stats = {
            'total_updates': 0,
            'total_interpolations': 0,
            'total_predictions': 0,
            'tracking_time_ms': 0.0,
            'avg_tracking_time_ms': 0.0,
            'active_tracks': 0,
            'total_tracks_created': 0,
            'interpolation_accuracy': 0.0
        }
        
        # スレッドセーフティ
        self._lock = threading.RLock()
        
        logger.info(f"HighFrequencyHand3DTracker initialized: {self.config.target_fps}fps target")
    
    def _setup_extended_kalman_matrices(self) -> None:
        """拡張カルマンフィルタ行列の設定（9次元：位置・速度・加速度）"""
        dt = self.dt
        dt2 = dt * dt / 2
        
        # 拡張状態遷移行列（等加速度モデル）
        self.F = np.array([
            # 位置 = 位置 + 速度*dt + 加速度*dt²/2
            [1, 0, 0, dt, 0, 0, dt2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, dt2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, dt2],
            # 速度 = 速度 + 加速度*dt
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            # 加速度 = 加速度（定数モデル）
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 観測行列（位置のみ観測）
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])
        
        # プロセスノイズ共分散行列
        q_pos = self.config.process_noise_position ** 2
        q_vel = self.config.process_noise_velocity ** 2
        q_acc = self.config.process_noise_acceleration ** 2
        
        self.Q = np.diag([
            q_pos, q_pos, q_pos,  # 位置ノイズ
            q_vel, q_vel, q_vel,  # 速度ノイズ
            q_acc, q_acc, q_acc   # 加速度ノイズ
        ])
        
        # 観測ノイズ共分散行列
        r = self.config.observation_noise ** 2
        self.R = np.eye(3) * r
    
    def update_60fps(self, hands_3d: Optional[List['Hand3DResult']] = None) -> List[HighFrequencyTrackedHand]:
        """
        60fps更新メソッド
        
        Args:
            hands_3d: 新しい3D手検出結果（MediaPipe検出時のみ提供）
            
        Returns:
            60fps更新されたトラッキング済み手のリスト
        """
        with self._lock:
            start_time = time.perf_counter()
            
            try:
                if hands_3d is not None:
                    # MediaPipe検出結果が利用可能な場合
                    return self._update_with_detection(hands_3d)
                else:
                    # 補間・予測のみの場合
                    return self._update_interpolation_only()
                    
            except Exception as e:
                logger.error(f"60fps tracking update error: {e}")
                return []
            finally:
                # 統計更新
                tracking_time = (time.perf_counter() - start_time) * 1000
                self._update_stats(tracking_time, hands_3d is not None)
    
    def _update_with_detection(self, hands_3d: List['Hand3DResult']) -> List[HighFrequencyTrackedHand]:
        """MediaPipe検出結果を使用した更新"""
        # 1. 既存トラックの予測
        self._predict_all_tracks()
        
        # 2. データアソシエーション
        assignments = self._assign_detections_to_tracks(hands_3d)
        
        # 3. 検出結果による更新
        self._update_tracks_with_detection(assignments, hands_3d)
        
        # 4. 新しいトラック作成
        self._create_new_tracks(assignments, hands_3d)
        
        # 5. 消失トラック処理
        self._handle_lost_tracks()
        
        # 6. FSM状態更新
        self._update_fsm_states()
        
        return self._get_active_tracks()
    
    def _update_interpolation_only(self) -> List[HighFrequencyTrackedHand]:
        """補間・予測のみの更新"""
        # 1. 既存トラックの予測
        self._predict_all_tracks()
        
        # 2. 補間フレーム数を増加
        for hand in self.tracked_hands.values():
            if hand.fsm_state == HandTrackingState.TRACKING:
                hand.interpolation_frames += 1
        
        # 3. 長時間補間のトラックを処理
        self._handle_long_interpolation()
        
        # 4. FSM状態更新
        self._update_fsm_states()
        
        return self._get_active_tracks()
    
    def _predict_all_tracks(self) -> None:
        """全トラックの予測ステップ"""
        current_time = time.perf_counter()
        
        for hand in self.tracked_hands.values():
            if hand.fsm_state in [HandTrackingState.TRACKING, HandTrackingState.LOST]:
                # カルマンフィルタ予測
                hand.state_vector = self.F @ hand.state_vector
                hand.covariance_matrix = (
                    self.F @ hand.covariance_matrix @ self.F.T + self.Q
                )
                
                # 状態更新
                hand.position = hand.state_vector[:3]
                hand.velocity = hand.state_vector[3:6]
                hand.acceleration = hand.state_vector[6:9]
                
                # 履歴更新
                hand.position_history.append(hand.position.copy())
                hand.velocity_history.append(hand.velocity.copy())
                
                # 信頼度減衰（補間時）
                if hand.interpolation_frames > 0:
                    decay_factor = 0.95 ** hand.interpolation_frames
                    hand.confidence_interpolation = hand.confidence_tracking * decay_factor
    
    def _assign_detections_to_tracks(self, hands_3d: List['Hand3DResult']) -> Dict[str, Optional[int]]:
        """検出結果とトラックの割り当て（Hungarian Algorithm）"""
        if not self.tracked_hands or not hands_3d:
            return {}
        
        track_ids = list(self.tracked_hands.keys())
        n_tracks = len(track_ids)
        n_detections = len(hands_3d)
        
        # コスト行列計算
        cost_matrix = np.full((n_tracks, n_detections), np.inf)
        
        for i, track_id in enumerate(track_ids):
            track = self.tracked_hands[track_id]
            for j, detection in enumerate(hands_3d):
                # 距離コスト
                distance = np.linalg.norm(track.position - detection.palm_center_3d)
                
                # 手の種類一致ボーナス
                handedness_bonus = 0.0
                if hasattr(detection, 'handedness') and track.handedness == detection.handedness:
                    handedness_bonus = -0.05  # 5cmのボーナス
                
                cost_matrix[i, j] = distance + handedness_bonus
        
        # Hungarian Algorithm
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            assignments = {}
            for i, track_id in enumerate(track_ids):
                assignments[track_id] = None
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= self.max_assignment_distance:
                    track_id = track_ids[row]
                    assignments[track_id] = col
            
            return assignments
            
        except Exception as e:
            logger.warning(f"Assignment algorithm failed: {e}")
            return {track_id: None for track_id in track_ids}
    
    def _update_tracks_with_detection(
        self,
        assignments: Dict[str, Optional[int]],
        hands_3d: List['Hand3DResult']
    ) -> None:
        """検出結果によるトラック更新"""
        current_time = time.perf_counter()
        
        for track_id, detection_idx in assignments.items():
            hand = self.tracked_hands[track_id]
            
            if detection_idx is not None:
                # 検出結果による更新
                detection = hands_3d[detection_idx]
                observation = np.array(detection.palm_center_3d)
                
                # カルマンフィルタ更新
                y = observation - self.H @ hand.state_vector  # 残差
                S = self.H @ hand.covariance_matrix @ self.H.T + self.R  # 残差共分散
                K = hand.covariance_matrix @ self.H.T @ np.linalg.inv(S)  # カルマンゲイン
                
                hand.state_vector += K @ y
                hand.covariance_matrix = (
                    np.eye(9) - K @ self.H
                ) @ hand.covariance_matrix
                
                # 状態更新
                hand.position = hand.state_vector[:3]
                hand.velocity = hand.state_vector[3:6]
                hand.acceleration = hand.state_vector[6:9]
                
                # メタデータ更新
                hand.confidence_2d = getattr(detection, 'confidence_2d', 0.8)
                hand.confidence_3d = getattr(detection, 'confidence_3d', 0.8)
                hand.confidence_tracking = min(1.0, hand.confidence_tracking + 0.1)
                hand.landmarks_3d = getattr(detection, 'landmarks_3d', None)
                hand.palm_normal = getattr(detection, 'palm_normal', None)
                
                hand.last_seen_time = current_time
                hand.last_detection_time = current_time
                hand.track_length += 1
                hand.lost_frames = 0
                hand.interpolation_frames = 0  # 検出でリセット
                
                # FSM状態更新
                if hand.fsm_state == HandTrackingState.LOST:
                    hand.fsm_state = HandTrackingState.RECOVERED
                elif hand.fsm_state in [HandTrackingState.DETECTING, HandTrackingState.RECOVERED]:
                    hand.fsm_state = HandTrackingState.TRACKING
                
            else:
                # 検出されなかった場合
                hand.lost_frames += 1
                if hand.lost_frames > self.max_lost_frames:
                    hand.fsm_state = HandTrackingState.LOST
    
    def _create_new_tracks(
        self,
        assignments: Dict[str, Optional[int]],
        hands_3d: List['Hand3DResult']
    ) -> None:
        """新しいトラックの作成"""
        assigned_detections = set(assignments.values())
        assigned_detections.discard(None)
        
        for i, detection in enumerate(hands_3d):
            if i not in assigned_detections:
                # 新しいトラック作成
                track_id = f"hf_hand_{uuid.uuid4().hex[:8]}"
                
                position = np.array(detection.palm_center_3d)
                velocity = np.zeros(3)
                acceleration = np.zeros(3)
                
                new_hand = HighFrequencyTrackedHand(
                    id=track_id,
                    handedness=getattr(detection, 'handedness', HandednessType.UNKNOWN),
                    fsm_state=HandTrackingState.DETECTING,
                    tracking_state=TrackingState.INITIALIZING,
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration,
                    confidence_2d=getattr(detection, 'confidence_2d', 0.8),
                    confidence_3d=getattr(detection, 'confidence_3d', 0.8),
                    confidence_tracking=0.5,
                    confidence_interpolation=1.0,
                    last_seen_time=time.perf_counter(),
                    last_detection_time=time.perf_counter(),
                    track_length=1,
                    lost_frames=0,
                    interpolation_frames=0,
                    hand_size=0.08,  # 8cm default
                    landmarks_3d=getattr(detection, 'landmarks_3d', None),
                    palm_normal=getattr(detection, 'palm_normal', None)
                )
                
                # 初期共分散設定
                new_hand.covariance_matrix = np.diag([
                    self.config.initial_position_variance,
                    self.config.initial_position_variance,
                    self.config.initial_position_variance,
                    self.config.initial_velocity_variance,
                    self.config.initial_velocity_variance,
                    self.config.initial_velocity_variance,
                    self.config.initial_acceleration_variance,
                    self.config.initial_acceleration_variance,
                    self.config.initial_acceleration_variance
                ])
                
                self.tracked_hands[track_id] = new_hand
                self.performance_stats['total_tracks_created'] += 1
                
                logger.debug(f"Created new 60fps track: {track_id}")
    
    def _handle_lost_tracks(self) -> None:
        """消失トラックの処理"""
        to_remove = []
        
        for track_id, hand in self.tracked_hands.items():
            if hand.fsm_state == HandTrackingState.LOST:
                if hand.lost_frames > self.max_lost_frames * 2:  # 完全削除
                    to_remove.append(track_id)
                    logger.debug(f"Removing lost track: {track_id}")
        
        for track_id in to_remove:
            del self.tracked_hands[track_id]
    
    def _handle_long_interpolation(self) -> None:
        """長時間補間トラックの処理"""
        for hand in self.tracked_hands.values():
            if hand.interpolation_frames > self.config.max_interpolation_frames:
                hand.fsm_state = HandTrackingState.LOST
                logger.debug(f"Track {hand.id} lost due to long interpolation")
    
    def _update_fsm_states(self) -> None:
        """FSM状態の更新"""
        for hand in self.tracked_hands.values():
            # トラッキング状態の更新
            if hand.fsm_state == HandTrackingState.TRACKING:
                if hand.track_length >= self.min_track_length:
                    hand.tracking_state = TrackingState.TRACKING
                else:
                    hand.tracking_state = TrackingState.INITIALIZING
            elif hand.fsm_state == HandTrackingState.LOST:
                hand.tracking_state = TrackingState.LOST
    
    def _get_active_tracks(self) -> List[HighFrequencyTrackedHand]:
        """アクティブなトラックを取得"""
        return [
            hand for hand in self.tracked_hands.values()
            if hand.fsm_state in [
                HandTrackingState.TRACKING,
                HandTrackingState.DETECTING,
                HandTrackingState.RECOVERED
            ]
        ]
    
    def _update_stats(self, tracking_time_ms: float, has_detection: bool) -> None:
        """統計更新"""
        self.performance_stats['total_updates'] += 1
        self.performance_stats['tracking_time_ms'] = tracking_time_ms
        
        if has_detection:
            self.performance_stats['total_predictions'] += 1
        else:
            self.performance_stats['total_interpolations'] += 1
        
        self.performance_stats['active_tracks'] = len(self._get_active_tracks())
        
        # 移動平均
        total = self.performance_stats['total_updates']
        prev_avg = self.performance_stats['avg_tracking_time_ms']
        self.performance_stats['avg_tracking_time_ms'] = (
            (prev_avg * (total - 1) + tracking_time_ms) / total
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        stats = self.performance_stats.copy()
        stats['interpolation_ratio'] = (
            stats['total_interpolations'] / max(1, stats['total_updates'])
        )
        return stats
    
    def reset(self) -> None:
        """トラッカーをリセット"""
        with self._lock:
            self.tracked_hands.clear()
            self.performance_stats = {
                'total_updates': 0,
                'total_interpolations': 0,
                'total_predictions': 0,
                'tracking_time_ms': 0.0,
                'avg_tracking_time_ms': 0.0,
                'active_tracks': 0,
                'total_tracks_created': 0,
                'interpolation_accuracy': 0.0
            }
            logger.info("60fps tracker reset")


def create_60fps_tracker(
    target_fps: int = 60,
    mediapipe_fps: int = 15
) -> HighFrequencyHand3DTracker:
    """60fps対応トラッカーの作成"""
    config = HighFrequencyKalmanConfig(
        target_fps=target_fps,
        mediapipe_fps=mediapipe_fps
    )
    return HighFrequencyHand3DTracker(config)


def create_high_frequency_tracker(
    target_fps: int = 60,
    mediapipe_fps: int = 15,
    **kwargs
) -> HighFrequencyHand3DTracker:
    """高頻度手追跡器を作成（外部インターフェース）"""
    config = HighFrequencyKalmanConfig(
        target_fps=target_fps,
        mediapipe_fps=mediapipe_fps,
        **kwargs
    )
    return HighFrequencyHand3DTracker(config)


def convert_to_legacy_tracked_hand(hf_hand: HighFrequencyTrackedHand) -> TrackedHand:
    """60fps対応TrackedHandを既存のTrackedHandに変換（互換性のため）"""
    return TrackedHand(
        id=hf_hand.id,
        handedness=hf_hand.handedness,
        state=hf_hand.tracking_state,
        position=hf_hand.position,
        velocity=hf_hand.velocity,
        acceleration=hf_hand.acceleration,
        confidence_2d=hf_hand.confidence_2d,
        confidence_3d=hf_hand.confidence_3d,
        confidence_tracking=hf_hand.confidence_tracking,
        last_seen_time=hf_hand.last_seen_time,
        track_length=hf_hand.track_length,
        lost_frames=hf_hand.lost_frames,
        hand_size=hf_hand.hand_size,
        landmarks_3d=hf_hand.landmarks_3d,
        palm_normal=hf_hand.palm_normal
    ) 