#!/usr/bin/env python3
"""
連続衝突検出モジュール
高速手動作に対応した衝突検出改善機能
"""

import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

from ..constants import (
    COLLISION_ADAPTIVE_VELOCITY_COEFFICIENT,
    COLLISION_ADAPTIVE_MAX_RADIUS_EXTENSION,
    COLLISION_INTERPOLATION_MIN_SAMPLES,
    COLLISION_INTERPOLATION_MAX_SAMPLES,
    COLLISION_HAND_HISTORY_SIZE,
    HIGH_SPEED_VELOCITY_THRESHOLD,
    VERY_HIGH_SPEED_VELOCITY_THRESHOLD
)
from ..detection.tracker import TrackedHand
from ..data_types import SearchResult, CollisionInfo
from .. import get_logger

logger = get_logger(__name__)


@dataclass
class ContinuousCollisionConfig:
    """連続衝突検出設定"""
    enable_interpolation_sampling: bool = True
    enable_adaptive_radius: bool = True
    enable_velocity_prediction: bool = True
    max_interpolation_distance: float = 0.15  # 15cm以上の移動で補間を有効化
    min_frame_interval: float = 0.01  # 最小フレーム間隔（100fps対応）


class ContinuousCollisionDetector:
    """連続衝突検出器"""
    
    def __init__(
        self, 
        config: Optional[ContinuousCollisionConfig] = None,
        base_sphere_radius: float = 0.05
    ):
        self.config = config or ContinuousCollisionConfig()
        self.base_sphere_radius = base_sphere_radius
        
        # 手位置履歴の拡張管理
        self._hand_position_history: Dict[str, List[np.ndarray]] = {}
        self._hand_velocity_history: Dict[str, List[np.ndarray]] = {}
        self._last_frame_time: Dict[str, float] = {}
        
        # パフォーマンス統計
        self.stats = {
            'interpolation_samples_generated': 0,
            'adaptive_radius_applied': 0,
            'high_speed_detections': 0,
            'continuous_collision_time_ms': 0.0
        }
    
    def calculate_adaptive_radius(self, hand: TrackedHand) -> float:
        """速度に応じた適応的半径を計算"""
        if not self.config.enable_adaptive_radius:
            return self.base_sphere_radius
        
        # 速度の大きさを取得
        velocity = getattr(hand, 'velocity', np.zeros(3))
        vel_magnitude = float(np.linalg.norm(velocity))
        
        # 改善された適応的半径計算
        if vel_magnitude > VERY_HIGH_SPEED_VELOCITY_THRESHOLD:
            # 超高速時: より積極的な拡張
            coeff = COLLISION_ADAPTIVE_VELOCITY_COEFFICIENT * 1.5
            max_extension = COLLISION_ADAPTIVE_MAX_RADIUS_EXTENSION * 1.2
        elif vel_magnitude > HIGH_SPEED_VELOCITY_THRESHOLD:
            # 高速時: 標準拡張
            coeff = COLLISION_ADAPTIVE_VELOCITY_COEFFICIENT
            max_extension = COLLISION_ADAPTIVE_MAX_RADIUS_EXTENSION
        else:
            # 低速時: 最小限の拡張
            coeff = COLLISION_ADAPTIVE_VELOCITY_COEFFICIENT * 0.7
            max_extension = COLLISION_ADAPTIVE_MAX_RADIUS_EXTENSION * 0.5
        
        # 適応的半径計算
        radius_extension = min(max_extension, vel_magnitude * coeff)
        adaptive_radius = self.base_sphere_radius + radius_extension
        
        if radius_extension > 0.01:  # 1cm以上の拡張時
            self.stats['adaptive_radius_applied'] += 1
            logger.debug(f"Adaptive radius: {adaptive_radius*100:.1f}cm (vel: {vel_magnitude:.2f}m/s)")
        
        return adaptive_radius
    
    def generate_interpolation_samples(
        self, 
        hand: TrackedHand, 
        current_time: float
    ) -> List[np.ndarray]:
        """補間サンプリング位置を生成"""
        if not self.config.enable_interpolation_sampling or hand.position is None:
            return [np.array(hand.position)]
        
        hand_id = hand.id
        current_pos = np.array(hand.position)
        
        # 履歴を更新
        self._update_hand_history(hand_id, current_pos, current_time)
        
        # 履歴から前フレーム位置を取得
        history = self._hand_position_history.get(hand_id, [])
        if len(history) < 2:
            return [current_pos]
        
        prev_pos = history[-2]  # 一つ前の位置
        
        # 移動距離とサンプル数を計算
        movement_distance = np.linalg.norm(current_pos - prev_pos)
        
        if movement_distance < self.config.max_interpolation_distance:
            return [current_pos]  # 短距離移動は補間不要
        
        # 速度に基づく動的サンプル数
        velocity = getattr(hand, 'velocity', np.zeros(3))
        vel_magnitude = float(np.linalg.norm(velocity))
        
        if vel_magnitude > VERY_HIGH_SPEED_VELOCITY_THRESHOLD:
            num_samples = COLLISION_INTERPOLATION_MAX_SAMPLES
        elif vel_magnitude > HIGH_SPEED_VELOCITY_THRESHOLD:
            num_samples = (COLLISION_INTERPOLATION_MIN_SAMPLES + COLLISION_INTERPOLATION_MAX_SAMPLES) // 2
        else:
            num_samples = COLLISION_INTERPOLATION_MIN_SAMPLES
        
        # 線形補間でサンプル生成
        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.0
            interpolated_pos = prev_pos + t * (current_pos - prev_pos)
            samples.append(interpolated_pos)
        
        self.stats['interpolation_samples_generated'] += len(samples) - 1
        
        if vel_magnitude > HIGH_SPEED_VELOCITY_THRESHOLD:
            self.stats['high_speed_detections'] += 1
            logger.debug(f"High-speed detection: {vel_magnitude:.2f}m/s, {len(samples)} samples")
        
        return samples
    
    def detect_continuous_collision(
        self,
        hand: TrackedHand,
        collision_searcher: Any,
        collision_tester: Any,
        current_time: float
    ) -> Optional[Any]:
        """連続衝突検出を実行"""
        start_time = time.perf_counter()
        
        try:
            # 適応的半径を計算
            adaptive_radius = self.calculate_adaptive_radius(hand)
            
            # 補間サンプル位置を生成
            sample_positions = self.generate_interpolation_samples(hand, current_time)
            
            # 各サンプル位置で衝突検出
            for sample_pos in sample_positions:
                # 空間検索
                search_result = collision_searcher._search_point(sample_pos, adaptive_radius)
                
                if not search_result.triangle_indices:
                    continue
                
                # 詳細衝突検出
                collision_info = collision_tester.test_sphere_collision(
                    sample_pos, adaptive_radius, search_result
                )
                
                if collision_info and collision_info.has_collision:
                    # 最初の衝突で終了（早期発見）
                    logger.debug(f"Continuous collision detected at sample {np.array_str(sample_pos, precision=3)}")
                    return collision_info
            
            return None
            
        except Exception as e:
            logger.error(f"Continuous collision detection error: {e}")
            return None
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats['continuous_collision_time_ms'] += elapsed_ms
    
    def _update_hand_history(self, hand_id: str, position: np.ndarray, timestamp: float):
        """手位置履歴を更新"""
        if hand_id not in self._hand_position_history:
            self._hand_position_history[hand_id] = []
            self._hand_velocity_history[hand_id] = []
            self._last_frame_time[hand_id] = timestamp
        
        # 位置履歴を更新
        self._hand_position_history[hand_id].append(position.copy())
        if len(self._hand_position_history[hand_id]) > COLLISION_HAND_HISTORY_SIZE:
            self._hand_position_history[hand_id].pop(0)
        
        # 速度履歴を更新（時間差分から計算）
        if len(self._hand_position_history[hand_id]) >= 2:
            dt = timestamp - self._last_frame_time[hand_id]
            if dt > 0:
                prev_pos = self._hand_position_history[hand_id][-2]
                velocity = (position - prev_pos) / dt
                self._hand_velocity_history[hand_id].append(velocity)
                if len(self._hand_velocity_history[hand_id]) > COLLISION_HAND_HISTORY_SIZE // 2:
                    self._hand_velocity_history[hand_id].pop(0)
        
        self._last_frame_time[hand_id] = timestamp
    
    def cleanup_old_tracks(self, active_hand_ids: List[str]):
        """非アクティブな手のトラック履歴をクリーンアップ"""
        for hand_id in list(self._hand_position_history.keys()):
            if hand_id not in active_hand_ids:
                del self._hand_position_history[hand_id]
                if hand_id in self._hand_velocity_history:
                    del self._hand_velocity_history[hand_id]
                if hand_id in self._last_frame_time:
                    del self._last_frame_time[hand_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計をリセット"""
        for key in self.stats:
            self.stats[key] = 0
            

# ファクトリー関数
def create_continuous_collision_detector(
    enable_interpolation: bool = True,
    enable_adaptive_radius: bool = True,
    base_radius: float = 0.05
) -> ContinuousCollisionDetector:
    """連続衝突検出器を作成"""
    config = ContinuousCollisionConfig(
        enable_interpolation_sampling=enable_interpolation,
        enable_adaptive_radius=enable_adaptive_radius
    )
    return ContinuousCollisionDetector(config, base_radius) 