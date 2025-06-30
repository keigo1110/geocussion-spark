#!/usr/bin/env python3
"""
カルマンフィルタベースの3D手トラッカー
位置平滑化・速度推定・複数手同時トラッキング
"""

import time
import uuid
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hands3d import Hand3DResult, Hand3DLandmark

from ..types import HandednessType, TrackingState
from src import get_logger

logger = get_logger(__name__)


@dataclass
class KalmanFilterConfig:
    """カルマンフィルタ設定"""
    # プロセスノイズ（位置の変動）
    process_noise_position: float = 0.05  # m
    process_noise_velocity: float = 0.1   # m/s
    
    # 観測ノイズ（3D投影の不確実性）
    observation_noise: float = 0.05  # m
    
    # 初期共分散
    initial_position_variance: float = 0.1  # m^2
    initial_velocity_variance: float = 0.5  # (m/s)^2


@dataclass
class TrackedHand:
    """トラッキング済み手情報"""
    id: str
    handedness: HandednessType
    state: TrackingState
    position: np.ndarray  # 3D位置 [x, y, z]
    velocity: np.ndarray  # 3D速度 [vx, vy, vz]
    acceleration: np.ndarray  # 3D加速度 [ax, ay, az]
    confidence_2d: float
    confidence_3d: float
    confidence_tracking: float  # トラッキング信頼度
    
    # トラッキング履歴
    last_seen_time: float
    track_length: int
    lost_frames: int
    
    # 手のサイズとジェスチャー
    hand_size: float  # 手のサイズ推定値
    landmarks_3d: Optional[List['Hand3DLandmark']] = None
    palm_normal: Optional[np.ndarray] = None
    
    # カルマンフィルタ状態
    state_vector: np.ndarray = field(init=False)
    covariance_matrix: np.ndarray = field(init=False)
    
    def __post_init__(self):
        # 状態ベクトル初期化 [x, y, z, vx, vy, vz]
        self.state_vector = np.array([
            self.position[0], self.position[1], self.position[2],
            self.velocity[0], self.velocity[1], self.velocity[2]
        ])
        
        # 共分散行列初期化
        self.covariance_matrix = np.eye(6) * 0.1
    
    @property
    def speed(self) -> float:
        """速度の大きさ"""
        return np.linalg.norm(self.velocity)
    
    @property
    def is_moving(self) -> bool:
        """動いているかどうか"""
        return self.speed > 0.05  # 5cm/s 以上で動いているとみなす
    
    @property
    def palm_center(self) -> np.ndarray:
        """手のひら中心位置（位置と同じ）"""
        return self.position
    
    @property
    def hand_id(self) -> str:
        """手のID（idと同じ）"""
        return self.id


class Hand3DTracker:
    """カルマンフィルタベースの3D手トラッカー"""
    
    def __init__(
        self,
        kalman_config: KalmanFilterConfig = None,
        max_lost_frames: int = 10,
        min_track_length: int = 5,
        max_assignment_distance: float = 0.3,  # 30cm
        dt: float = 1.0/30.0  # 30fps仮定
    ):
        """
        初期化
        
        Args:
            kalman_config: カルマンフィルタ設定
            max_lost_frames: 最大消失フレーム数
            min_track_length: 最小トラック長
            max_assignment_distance: 最大割り当て距離
            dt: フレーム間隔
        """
        self.kalman_config = kalman_config or KalmanFilterConfig()
        self.max_lost_frames = max_lost_frames
        self.min_track_length = min_track_length
        self.max_assignment_distance = max_assignment_distance
        self.dt = dt
        
        # トラッキング対象
        self.tracked_hands: Dict[str, TrackedHand] = {}
        
        # カルマンフィルタ行列
        self._setup_kalman_matrices()
        
        # 統計情報
        self.performance_stats = {
            'total_updates': 0,
            'tracking_time_ms': 0.0,
            'avg_tracking_time_ms': 0.0,
            'active_tracks': 0,
            'total_tracks_created': 0
        }
    
    def _setup_kalman_matrices(self) -> None:
        """カルマンフィルタ行列の設定"""
        # 状態遷移行列 (等速度モデル)
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 観測行列（位置のみ観測）
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # プロセスノイズ共分散行列
        q_pos = self.kalman_config.process_noise_position ** 2
        q_vel = self.kalman_config.process_noise_velocity ** 2
        
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        
        # 観測ノイズ共分散行列
        r = self.kalman_config.observation_noise ** 2
        self.R = np.eye(3) * r
    
    def update(self, hands_3d: List['Hand3DResult']) -> List[TrackedHand]:
        """
        トラッキング更新
        
        Args:
            hands_3d: 新しい3D手検出結果
            
        Returns:
            更新されたトラッキング済み手のリスト
        """
        start_time = time.perf_counter()
        
        try:
            # 1. 予測ステップ（既存トラックの予測）
            self._predict_existing_tracks()
            
            # 2. データアソシエーション（検出結果とトラックの対応付け）
            assignments = self._assign_detections_to_tracks(hands_3d)
            
            # 3. 更新ステップ
            self._update_assigned_tracks(assignments, hands_3d)
            
            # 4. 新しいトラック作成
            self._create_new_tracks(assignments, hands_3d)
            
            # 5. 消失トラックの処理
            self._handle_lost_tracks()
            
            # 6. アクティブなトラックのフィルタリング（TERMINATED状態を除外）
            active_tracks = [
                track for track in self.tracked_hands.values()
                if track.state in [TrackingState.TRACKING, TrackingState.INITIALIZING]
            ]
            
            # 7. 統計更新
            tracking_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(tracking_time)
            
            return active_tracks
            
        except Exception as e:
            logger.warning(f"Tracking update error: {e}")
            return []
    
    def _predict_existing_tracks(self) -> None:
        """既存トラックの予測"""
        current_time = time.perf_counter()
        
        for track in self.tracked_hands.values():
            if track.state == TrackingState.TERMINATED:
                continue
            
            # カルマンフィルタ予測
            track.state_vector = self.F @ track.state_vector
            track.covariance_matrix = (
                self.F @ track.covariance_matrix @ self.F.T + self.Q
            )
            
            # 予測値から位置・速度を更新
            track.position = track.state_vector[:3]
            track.velocity = track.state_vector[3:]
            
            # 加速度更新（速度の変化率）
            dt = current_time - track.last_seen_time
            if dt > 0:
                # 簡易的な加速度推定
                prev_velocity = track.velocity
                track.acceleration = (track.velocity - prev_velocity) / dt
    
    def _assign_detections_to_tracks(
        self, 
        hands_3d: List['Hand3DResult']
    ) -> Dict[str, Optional[int]]:
        """
        検出結果とトラックの対応付け（ハンガリアンアルゴリズム）
        
        Returns:
            {track_id: detection_index or None}
        """
        if not hands_3d:
            # 検出結果がない場合、全てのアクティブトラックをNoneに割り当て
            active_tracks = [
                (track_id, track) for track_id, track in self.tracked_hands.items()
                if track.state in [TrackingState.TRACKING, TrackingState.INITIALIZING, TrackingState.LOST]
            ]
            return {track_id: None for track_id, _ in active_tracks}
        
        # アクティブなトラックのリスト
        active_tracks = [
            (track_id, track) for track_id, track in self.tracked_hands.items()
            if track.state in [TrackingState.TRACKING, TrackingState.INITIALIZING, TrackingState.LOST]
        ]
        
        if not active_tracks:
            # 初回検出時はアクティブトラックが存在しないため、空の辞書を返す
            # この後_create_new_tracksで新しいトラックが作成される
            return {}
        
        # コスト行列作成
        cost_matrix = np.full((len(active_tracks), len(hands_3d)), np.inf)
        
        for i, (track_id, track) in enumerate(active_tracks):
            for j, hand_3d in enumerate(hands_3d):
                # 距離コスト
                distance = np.linalg.norm(
                    track.position - np.array(hand_3d.palm_center_3d)
                )
                
                # 手の左右一致ボーナス
                handedness_match = track.handedness == hand_3d.handedness
                handedness_bonus = -0.05 if handedness_match else 0.1
                
                # 信頼度ボーナス
                confidence_bonus = -0.02 * hand_3d.confidence_3d
                
                total_cost = distance + handedness_bonus + confidence_bonus
                
                if distance <= self.max_assignment_distance:
                    cost_matrix[i, j] = total_cost
        
        # 全ての要素がinfの場合は空の辞書を返す
        if np.all(cost_matrix == np.inf):
            return {track_id: None for track_id, _ in active_tracks}
        
        # ハンガリアンアルゴリズムで最適割り当て
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            # cost matrixに問題がある場合は全てをNoneに設定
            logger.error(f"Assignment algorithm failed: {e}, resetting assignments")
            return {track_id: None for track_id, _ in active_tracks}
        
        assignments = {}
        used_detections = set()
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < np.inf:
                track_id = active_tracks[row][0]
                assignments[track_id] = col
                used_detections.add(col)
        
        # 割り当てられなかったトラック
        for track_id, _ in active_tracks:
            if track_id not in assignments:
                assignments[track_id] = None
        
        return assignments
    
    def _update_assigned_tracks(
        self,
        assignments: Dict[str, Optional[int]],
        hands_3d: List['Hand3DResult']
    ) -> None:
        """割り当てられたトラックの更新"""
        current_time = time.perf_counter()
        
        for track_id, detection_idx in assignments.items():
            if detection_idx is None:
                # 検出されなかった場合
                track = self.tracked_hands[track_id]
                track.lost_frames += 1
                
                if track.lost_frames > self.max_lost_frames:
                    # max_lost_framesを超えたらLOST状態に変更
                    # _handle_lost_tracks()で後処理される
                    track.state = TrackingState.LOST
                
                continue
            
            # 検出された場合
            track = self.tracked_hands[track_id]
            hand_3d = hands_3d[detection_idx]
            
            # カルマンフィルタ更新
            observation = np.array(hand_3d.palm_center_3d)
            
            # 更新ステップ
            y = observation - self.H @ track.state_vector  # 残差
            S = self.H @ track.covariance_matrix @ self.H.T + self.R  # 残差共分散
            K = track.covariance_matrix @ self.H.T @ np.linalg.inv(S)  # カルマンゲイン
            
            track.state_vector += K @ y
            track.covariance_matrix = (
                np.eye(6) - K @ self.H
            ) @ track.covariance_matrix
            
            # トラック情報更新
            track.position = track.state_vector[:3]
            track.velocity = track.state_vector[3:]
            track.confidence_2d = hand_3d.confidence_2d
            track.confidence_3d = hand_3d.confidence_3d
            track.landmarks_3d = hand_3d.landmarks_3d
            track.palm_normal = hand_3d.palm_normal
            
            track.last_seen_time = current_time
            track.track_length += 1
            track.lost_frames = 0
            
            # トラッキング信頼度更新
            track.confidence_tracking = min(
                track.confidence_3d * (track.track_length / self.min_track_length),
                1.0
            )
            
            # 状態更新
            if track.state == TrackingState.INITIALIZING:
                if track.track_length >= self.min_track_length:
                    track.state = TrackingState.TRACKING
            elif track.state == TrackingState.LOST:
                track.state = TrackingState.TRACKING
    
    def _create_new_tracks(
        self,
        assignments: Dict[str, Optional[int]],
        hands_3d: List['Hand3DResult']
    ) -> None:
        """新しいトラックの作成"""
        used_indices = set(
            idx for idx in assignments.values() if idx is not None
        )
        
        for i, hand_3d in enumerate(hands_3d):
            if i not in used_indices:
                # 新しいトラック作成
                track_id = str(uuid.uuid4())
                
                new_track = TrackedHand(
                    id=track_id,
                    handedness=hand_3d.handedness,
                    state=TrackingState.INITIALIZING,
                    position=np.array(hand_3d.palm_center_3d),
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    confidence_2d=hand_3d.confidence_2d,
                    confidence_3d=hand_3d.confidence_3d,
                    confidence_tracking=hand_3d.confidence_3d,
                    last_seen_time=time.perf_counter(),
                    track_length=1,
                    lost_frames=0,
                    hand_size=0.0,
                    landmarks_3d=hand_3d.landmarks_3d,
                    palm_normal=hand_3d.palm_normal
                )
                
                # カルマンフィルタ初期化
                new_track.covariance_matrix = np.diag([
                    self.kalman_config.initial_position_variance,
                    self.kalman_config.initial_position_variance,
                    self.kalman_config.initial_position_variance,
                    self.kalman_config.initial_velocity_variance,
                    self.kalman_config.initial_velocity_variance,
                    self.kalman_config.initial_velocity_variance
                ])
                
                self.tracked_hands[track_id] = new_track
                self.performance_stats['total_tracks_created'] += 1
    
    def _handle_lost_tracks(self) -> None:
        """消失トラックの処理"""
        tracks_to_remove = []
        
        for track_id, track in self.tracked_hands.items():
            if track.state == TrackingState.LOST:
                if track.track_length < self.min_track_length:
                    # 短いトラックは削除
                    tracks_to_remove.append(track_id)
                else:
                    # 長いトラックは終了状態に
                    track.state = TrackingState.TERMINATED
        
        for track_id in tracks_to_remove:
            del self.tracked_hands[track_id]
    
    def get_tracked_hand(self, hand_id: str) -> Optional[TrackedHand]:
        """特定の手を取得"""
        return self.tracked_hands.get(hand_id)
    
    def get_dominant_hand(self) -> Optional[TrackedHand]:
        """最も信頼度の高い手を取得"""
        active_tracks = [
            track for track in self.tracked_hands.values()
            if track.state == TrackingState.TRACKING
        ]
        
        if not active_tracks:
            return None
        
        return max(active_tracks, key=lambda t: t.confidence_tracking)
    
    def reset(self) -> None:
        """トラッカーリセット"""
        self.tracked_hands.clear()
        self.performance_stats['total_tracks_created'] = 0
    
    def _update_stats(self, tracking_time_ms: float) -> None:
        """統計更新"""
        self.performance_stats['total_updates'] += 1
        self.performance_stats['tracking_time_ms'] = tracking_time_ms
        self.performance_stats['active_tracks'] = len([
            t for t in self.tracked_hands.values()
            if t.state == TrackingState.TRACKING
        ])
        
        # 移動平均
        total = self.performance_stats['total_updates']
        prev_avg = self.performance_stats['avg_tracking_time_ms']
        self.performance_stats['avg_tracking_time_ms'] = (
            (prev_avg * (total - 1) + tracking_time_ms) / total
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        return self.performance_stats.copy()
    
    def update_config(self, **kwargs) -> None:
        """設定の動的更新"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.kalman_config, key):
                setattr(self.kalman_config, key, value)
                # カルマンフィルタ行列を再設定
                self._setup_kalman_matrices()


# ユーティリティ関数

def create_test_tracker() -> Hand3DTracker:
    """テスト用トラッカー作成"""
    config = KalmanFilterConfig(
        process_noise_position=0.005,
        process_noise_velocity=0.05,
        observation_noise=0.01
    )
    
    return Hand3DTracker(
        kalman_config=config,
        max_lost_frames=5,
        min_track_length=3,
        max_assignment_distance=0.2
    )


def filter_stable_hands(
    tracked_hands: List[TrackedHand],
    min_confidence: float = 0.7,
    max_speed: float = 2.0  # m/s
) -> List[TrackedHand]:
    """安定した手のみフィルタリング"""
    return [
        hand for hand in tracked_hands
        if (hand.confidence_tracking >= min_confidence and
            hand.speed <= max_speed and
            hand.state == TrackingState.TRACKING)
    ]


def get_hand_velocities(tracked_hands: List[TrackedHand]) -> Dict[str, np.ndarray]:
    """手の速度情報を取得"""
    return {
        hand.id: hand.velocity
        for hand in tracked_hands
        if hand.state == TrackingState.TRACKING
    } 