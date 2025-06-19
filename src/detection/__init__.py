"""
手検出フェーズパッケージ
MediaPipe による 2D 手検出 → 深度マップ参照による 3D 座標化 → カルマンフィルタによる平滑化＆速度推定
"""

from .hands2d import (
    MediaPipeHandsWrapper,
    HandDetectionResult,
    HandLandmark,
    HandednessType,
    create_mock_hand_result,
    filter_hands_by_confidence,
    get_dominant_hand
)

from .hands3d import (
    Hand3DProjector,
    Hand3DResult,
    Hand3DLandmark,
    DepthInterpolationMethod,
    calculate_hand_size_3d,
    filter_hands_3d_by_depth,
    create_mock_hand_3d_result
)

from .tracker import (
    Hand3DTracker,
    TrackedHand,
    TrackingState,
    KalmanFilterConfig,
    create_test_tracker,
    filter_stable_hands,
    get_hand_velocities
)

__all__ = [
    # 2D手検出
    'MediaPipeHandsWrapper',
    'HandDetectionResult',
    'HandLandmark',
    'HandednessType',
    'create_mock_hand_result',
    'filter_hands_by_confidence',
    'get_dominant_hand',
    
    # 3D投影
    'Hand3DProjector',
    'Hand3DResult',
    'Hand3DLandmark',
    'DepthInterpolationMethod',
    'calculate_hand_size_3d',
    'filter_hands_3d_by_depth',
    'create_mock_hand_3d_result',
    
    # トラッキング
    'Hand3DTracker',
    'TrackedHand',
    'TrackingState',
    'KalmanFilterConfig',
    'create_test_tracker',
    'filter_stable_hands',
    'get_hand_velocities'
] 