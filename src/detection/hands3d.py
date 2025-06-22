#!/usr/bin/env python3
"""
3D手投影ユーティリティ
2D手ランドマークを深度マップ経由で3D座標に変換
"""

import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import griddata

# 入力フェーズからインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.input.stream import CameraIntrinsics
from .hands2d import HandDetectionResult, HandLandmark, HandednessType


@dataclass
class Hand3DLandmark:
    """3D手ランドマーク座標"""
    x: float  # 3D空間座標 (m)
    y: float  # 3D空間座標 (m)
    z: float  # 3D空間座標 (m)
    confidence: float  # 3D推定信頼度
    depth_valid: bool  # 深度値が有効か
    
    @property
    def position(self) -> np.ndarray:
        """3D位置ベクトル"""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Hand3DLandmark') -> float:
        """他のランドマークとの距離"""
        return np.linalg.norm(self.position - other.position)


@dataclass
class Hand3DResult:
    """3D手検出結果"""
    id: str
    landmarks_3d: List[Tuple[float, float, float]]
    palm_center_3d: Tuple[float, float, float]
    handedness: HandednessType
    confidence: float

    @property
    def position(self):
        return np.array(self.palm_center_3d)


class DepthInterpolationMethod(Enum):
    """深度補間手法"""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    GAUSSIAN = "gaussian"


class Hand3DProjector:
    """2D→3D手投影クラス"""
    
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        depth_scale: float = 1000.0,
        interpolation_method: DepthInterpolationMethod = DepthInterpolationMethod.LINEAR,
        depth_filter_kernel_size: int = 3,
        max_depth_diff: float = 0.05,  # 5cm
        min_confidence_3d: float = 0.3,
        use_guided_filter: bool = False,
        **guided_filter_params
    ):
        """
        初期化
        
        Args:
            camera_intrinsics: カメラ内部パラメータ
            depth_scale: 深度スケール（mm→m変換）
            interpolation_method: 深度補間手法
            depth_filter_kernel_size: 深度フィルタカーネルサイズ
            max_depth_diff: 最大深度差閾値
            min_confidence_3d: 最小3D信頼度
            use_guided_filter: ガイドフィルタを使用するか
            **guided_filter_params: ガイドフィルタのパラメータ
        """
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale = depth_scale
        self.interpolation_method = interpolation_method
        self.depth_filter_kernel_size = depth_filter_kernel_size
        self.max_depth_diff = max_depth_diff
        self.min_confidence_3d = min_confidence_3d
        self.use_guided_filter = use_guided_filter
        self.guided_filter_params = guided_filter_params
        
        # パフォーマンス統計
        self.performance_stats = {
            'total_projections': 0,
            'projection_time_ms': 0.0,
            'avg_projection_time_ms': 0.0,
            'valid_landmarks_ratio': 0.0
        }
        
        # 事前計算されたインデックス
        self._precompute_indices()
    
    def _precompute_indices(self) -> None:
        """インデックスマップを事前計算"""
        height, width = self.camera_intrinsics.height, self.camera_intrinsics.width
        
        # ピクセル座標系
        self.pixel_coords = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing='xy'
        )
    
    def project_hand_to_3d(
        self,
        hand_2d: HandDetectionResult,
        depth_image: np.ndarray
    ) -> Optional[Hand3DResult]:
        """
        2D手検出結果を3Dに投影
        
        Args:
            hand_2d: 2D手検出結果
            depth_image: 深度画像 (uint16, mm)
            
        Returns:
            3D手検出結果またはNone
        """
        start_time = time.perf_counter()
        
        try:
            # 深度画像の前処理
            filtered_depth = self._preprocess_depth(depth_image)
            
            # 各ランドマークを3Dに投影
            landmarks_3d = []
            valid_count = 0
            
            for landmark_2d in hand_2d.landmarks:
                landmark_3d = self._project_landmark_to_3d(
                    landmark_2d, 
                    filtered_depth
                )
                landmarks_3d.append(landmark_3d)
                if landmark_3d.depth_valid:
                    valid_count += 1
            
            # 有効ランドマーク数による信頼度計算
            confidence_3d = valid_count / len(landmarks_3d) if landmarks_3d else 0.0
            
            if confidence_3d < self.min_confidence_3d:
                return None
            
            # 手のひら中心の3D座標計算
            palm_center_3d = self._calculate_palm_center_3d(landmarks_3d)
            
            # landmarks_3dをタプルに変換
            landmarks_tuples = []
            for lm in landmarks_3d:
                landmarks_tuples.append((lm.x, lm.y, lm.z))
            
            # 結果作成
            result = Hand3DResult(
                id=hand_2d.id,
                landmarks_3d=landmarks_tuples,
                palm_center_3d=palm_center_3d,
                handedness=hand_2d.handedness,
                confidence=hand_2d.confidence
            )
            
            # 統計更新
            projection_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(projection_time, confidence_3d)
            
            return result
            
        except Exception as e:
            print(f"3D projection error: {e}")
            return None
    
    def _preprocess_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """深度画像の前処理（NaN補間と平滑化）"""
        # uint16 → float32 変換とスケーリング
        depth_float = depth_image.astype(np.float32) / self.depth_scale
        
        # 無効値（0）をNaNに変換
        depth_float[depth_float == 0] = np.nan
        
        # NaNを補間 (Inpainting)
        nan_mask = np.isnan(depth_float).astype(np.uint8)
        if np.any(nan_mask):
            # OpenCVのinpaint関数を利用
            # 8bitに正規化してから処理
            max_val = np.nanmax(depth_float)
            if max_val > 0:
                norm_image = (depth_float / max_val * 255).astype(np.uint8)
                inpainted_norm = cv2.inpaint(norm_image, nan_mask, 3, cv2.INPAINT_TELEA)
                depth_float = (inpainted_norm.astype(np.float32) / 255.0) * max_val
        
        # ガウシアンフィルタで平滑化
        if self.depth_filter_kernel_size > 1:
            # カーネルサイズは奇数に
            ksize = self.depth_filter_kernel_size if self.depth_filter_kernel_size % 2 != 0 else self.depth_filter_kernel_size + 1
            depth_float = cv2.GaussianBlur(depth_float, (ksize, ksize), 0)
        
        return depth_float
    
    def _project_landmark_to_3d(
        self,
        landmark_2d: HandLandmark,
        depth_image: np.ndarray
    ) -> Hand3DLandmark:
        """2Dランドマークを3Dに投影"""
        # ピクセル座標
        u = landmark_2d.x * self.camera_intrinsics.width
        v = landmark_2d.y * self.camera_intrinsics.height
        
        # 深度値取得・補間
        depth_z, depth_confidence = self._get_interpolated_depth(
            u, v, depth_image
        )
        
        if np.isnan(depth_z) or depth_z <= 0:
            # 無効な深度値
            return Hand3DLandmark(
                x=0.0, y=0.0, z=0.0,
                confidence=0.0,
                depth_valid=False
            )
        
        # 3D座標計算
        x = (u - self.camera_intrinsics.cx) * depth_z / self.camera_intrinsics.fx
        y = -(v - self.camera_intrinsics.cy) * depth_z / self.camera_intrinsics.fy
        z = depth_z
        
        # NaNまたは無限大のチェック
        if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x) or np.isinf(y) or np.isinf(z):
            # 無効な3D座標の場合は無効なランドマークを返す
            return Hand3DLandmark(
                x=0.0, y=0.0, z=0.0,
                confidence=0.0,
                depth_valid=False
            )
        
        # 信頼度計算（MediaPipe可視性スコア × 深度信頼度）
        confidence_3d = landmark_2d.visibility * depth_confidence
        
        return Hand3DLandmark(
            x=x,
            y=y,
            z=z,
            confidence=confidence_3d,
            depth_valid=True
        )
    
    def _get_interpolated_depth(
        self,
        u: float,
        v: float,
        depth_image: np.ndarray
    ) -> Tuple[float, float]:
        """
        深度値の補間取得
        
        Returns:
            (depth_value, confidence)
        """
        height, width = depth_image.shape
        
        # 境界チェック
        if u < 0 or u >= width or v < 0 or v >= height:
            return np.nan, 0.0
        
        if self.interpolation_method == DepthInterpolationMethod.NEAREST:
            # 最近傍補間
            i, j = int(round(v)), int(round(u))
            if 0 <= i < height and 0 <= j < width:
                depth = depth_image[i, j]
                confidence = 1.0 if not np.isnan(depth) else 0.0
                return depth, confidence
            return np.nan, 0.0
        
        elif self.interpolation_method == DepthInterpolationMethod.LINEAR:
            # バイリニア補間
            i0, j0 = int(v), int(u)
            i1, j1 = min(i0 + 1, height - 1), min(j0 + 1, width - 1)
            
            # 4近傍の深度値取得
            depths = []
            weights = []
            
            for i, j in [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]:
                if 0 <= i < height and 0 <= j < width:
                    d = depth_image[i, j]
                    if not np.isnan(d):
                        # 距離重み計算
                        w = 1.0 / (1.0 + np.sqrt((v - i)**2 + (u - j)**2))
                        depths.append(d)
                        weights.append(w)
            
            if depths:
                weights = np.array(weights)
                weights /= weights.sum()
                interpolated_depth = np.average(depths, weights=weights)
                confidence = len(depths) / 4.0  # 有効点数による信頼度
                return interpolated_depth, confidence
            
            return np.nan, 0.0
        
        elif self.interpolation_method == DepthInterpolationMethod.GAUSSIAN:
            # ガウシアン重み補間
            kernel_size = 5
            half_kernel = kernel_size // 2
            
            i_center, j_center = int(v), int(u)
            total_weight = 0.0
            weighted_depth = 0.0
            valid_points = 0
            
            for di in range(-half_kernel, half_kernel + 1):
                for dj in range(-half_kernel, half_kernel + 1):
                    i, j = i_center + di, j_center + dj
                    if 0 <= i < height and 0 <= j < width:
                        d = depth_image[i, j]
                        if not np.isnan(d):
                            # ガウシアン重み
                            weight = np.exp(-(di**2 + dj**2) / (2 * 1.5**2))
                            weighted_depth += d * weight
                            total_weight += weight
                            valid_points += 1
            
            if total_weight > 0:
                interpolated_depth = weighted_depth / total_weight
                confidence = min(valid_points / (kernel_size**2), 1.0)
                return interpolated_depth, confidence
            
            return np.nan, 0.0
        
        # デフォルト: 最近傍
        return self._get_interpolated_depth(u, v, depth_image)
    
    def _calculate_palm_center_3d(
        self,
        landmarks_3d: List[Hand3DLandmark]
    ) -> Tuple[float, float, float]:
        """手のひら中心の3D座標計算"""
        if len(landmarks_3d) < 21:
            return (0.0, 0.0, 0.0)
        
        # 手のひらの主要ランドマーク（手首・各指の付け根）
        palm_indices = [0, 5, 9, 13, 17]
        valid_palm_points = []
        
        for idx in palm_indices:
            if landmarks_3d[idx].depth_valid:
                valid_palm_points.append(landmarks_3d[idx].position)
        
        if valid_palm_points:
            center = np.mean(valid_palm_points, axis=0)
            return (center[0], center[1], center[2])
        
        return (0.0, 0.0, 0.0)
    
    def project_hands_batch(
        self,
        hands_2d: List[HandDetectionResult],
        depth_image: np.ndarray
    ) -> List[Hand3DResult]:
        """複数手のバッチ3D投影"""
        results = []
        batch_start_time = time.perf_counter()
        
        for i, hand_2d in enumerate(hands_2d):
            result = self.project_hand_to_3d(hand_2d, depth_image)
            if result:
                results.append(result)
        
        batch_time = (time.perf_counter() - batch_start_time) * 1000
        if hands_2d:
            print(f"Batch 3D projection: {len(hands_2d)} hands in {batch_time:.1f}ms "
                  f"({batch_time/len(hands_2d):.1f}ms/hand) -> {len(results)} successful")
        
        return results
    
    def _update_stats(self, projection_time_ms: float, confidence_3d: float) -> None:
        """統計情報を更新"""
        self.performance_stats['total_projections'] += 1
        self.performance_stats['projection_time_ms'] = projection_time_ms
        
        # 移動平均計算
        total = self.performance_stats['total_projections']
        prev_avg = self.performance_stats['avg_projection_time_ms']
        self.performance_stats['avg_projection_time_ms'] = (
            (prev_avg * (total - 1) + projection_time_ms) / total
        )
        
        # 有効ランドマーク比率
        prev_ratio = self.performance_stats['valid_landmarks_ratio']
        self.performance_stats['valid_landmarks_ratio'] = (
            (prev_ratio * (total - 1) + confidence_3d) / total
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        return self.performance_stats.copy()
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        self.performance_stats = {
            'total_projections': 0,
            'projection_time_ms': 0.0,
            'avg_projection_time_ms': 0.0,
            'valid_landmarks_ratio': 0.0
        }
    
    def update_intrinsics(self, new_intrinsics: CameraIntrinsics) -> None:
        """カメラ内部パラメータを更新"""
        self.camera_intrinsics = new_intrinsics
        self._precompute_indices()


# ユーティリティ関数

def calculate_hand_size_3d(hand_3d: Hand3DResult) -> float:
    """3D手のサイズ（手首-中指先端距離）を計算"""
    if len(hand_3d.landmarks_3d) < 21:
        return 0.0
    
    wrist = hand_3d.landmarks_3d[0]
    middle_tip = hand_3d.landmarks_3d[12]
    
    if wrist.depth_valid and middle_tip.depth_valid:
        return wrist.distance_to(middle_tip)
    
    return 0.0


def filter_hands_3d_by_depth(
    hands_3d: List[Hand3DResult],
    min_depth: float = 0.1,
    max_depth: float = 2.0
) -> List[Hand3DResult]:
    """深度範囲でフィルタリング"""
    filtered = []
    for hand in hands_3d:
        palm_z = hand.palm_center_3d[2]
        if min_depth <= palm_z <= max_depth:
            filtered.append(hand)
    return filtered


def create_mock_hand_3d_result() -> Hand3DResult:
    """テスト用のモック3D手検出結果を作成"""
    landmarks_3d = []
    for i in range(21):
        landmarks_3d.append(Hand3DLandmark(
            x=0.1 + 0.05 * np.sin(i * 0.3),
            y=0.0 + 0.05 * np.cos(i * 0.3),
            z=0.8 + 0.02 * np.sin(i * 0.5),
            confidence=0.9,
            depth_valid=True
        ))
    
    return Hand3DResult(
        id="mock_id",
        landmarks_3d=[tuple(lm.position) for lm in landmarks_3d],
        palm_center_3d=(0.1, 0.0, 0.8),
        handedness=HandednessType.RIGHT,
        confidence=0.85
    ) 