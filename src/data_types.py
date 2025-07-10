#!/usr/bin/env python3
"""
共通型定義

アプリケーション全体で使用される型定義を一元管理し、
モジュール間の循環依存を解消します。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Any, Protocol, runtime_checkable, Union
import numpy as np
from abc import ABC, abstractmethod

# 型エイリアス
ArrayLike = Union[np.ndarray, List, Tuple]

# =============================================================================
# プロトコル定義（インターフェース）
# =============================================================================

@runtime_checkable
class Renderable(Protocol):
    """描画可能なオブジェクトのプロトコル"""
    
    def render(self, image: np.ndarray, intrinsics: 'CameraIntrinsics') -> np.ndarray:
        """オブジェクトを画像に描画"""
        ...


@runtime_checkable
class Trackable(Protocol):
    """追跡可能なオブジェクトのプロトコル"""
    
    @property
    def position(self) -> np.ndarray:
        """現在位置"""
        ...
    
    @property
    def velocity(self) -> np.ndarray:
        """速度ベクトル"""
        ...


@runtime_checkable
class Collidable(Protocol):
    """衝突可能なオブジェクトのプロトコル"""
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """境界ボックス（min, max）"""
        ...
    
    def intersects(self, other: 'Collidable') -> bool:
        """他のオブジェクトと交差するか"""
        ...


# =============================================================================
# 入力システム型定義
# =============================================================================

class OBFormat(Enum):
    """OrbbecSDKカラーフォーマット"""
    RGB = "RGB"
    BGR = "BGR"
    MJPG = "MJPG"


@dataclass
class FrameData:
    """フレームデータ構造（Orbbec SDK互換）"""
    depth_frame: Optional[Any] = None
    color_frame: Optional[Any] = None
    timestamp_ms: float = 0.0
    frame_number: int = 0
    points: Optional[np.ndarray] = None  # 点群データ
    
    @property
    def has_color(self) -> bool:
        """カラーデータが存在するか"""
        return self.color_frame is not None
    
    # 後方互換性のためのプロパティ
    @property
    def depth_image(self) -> Optional[np.ndarray]:
        """深度画像データ（後方互換性）"""
        if self.depth_frame is None:
            return None
        # OrbbecフレームからNumPy配列を取得する処理を想定
        return getattr(self.depth_frame, 'get_data', lambda: None)()
    
    @property
    def color_data(self) -> Optional[np.ndarray]:
        """カラーデータ（後方互換性）"""
        if self.color_frame is None:
            return None
        return getattr(self.color_frame, 'get_data', lambda: None)()
    
    @property
    def timestamp(self) -> float:
        """タイムスタンプ（秒）"""
        return self.timestamp_ms / 1000.0


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    def project_point(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """3D点を2D画像座標に投影"""
        x, y, z = point_3d
        if z <= 0:
            return -1, -1
        
        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)
        
        if 0 <= u < self.width and 0 <= v < self.height:
            return u, v
        return -1, -1


# =============================================================================
# 検出システム型定義
# =============================================================================

class HandednessType(Enum):
    """手の種類"""
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


class TrackingState(Enum):
    """トラッキング状態"""
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass
class HandLandmark:
    """手のランドマーク座標"""
    x: float  # 0-1の正規化座標
    y: float  # 0-1の正規化座標
    z: float  # 深度情報（相対値）
    visibility: float = 1.0  # 可視性スコア


@dataclass
class HandROI:
    """手領域の矩形情報"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    last_updated_frame: int = 0
    hand_id: str = ""
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """OpenCV tracker用のバウンディングボックス (x, y, w, h)"""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """矩形の中心座標"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def area(self) -> int:
        """矩形の面積"""
        return self.width * self.height


@dataclass
class HandDetectionResult:
    """手検出結果"""
    id: str  # 手のID（トラッキング用）
    landmarks: List[HandLandmark]
    handedness: HandednessType
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    timestamp_ms: float
    is_tracked: bool = False  # ROIトラッキングで生成されたかどうか
    
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


@dataclass
class ROITrackingStats:
    """ROIトラッキング統計情報"""
    total_frames: int = 0
    mediapipe_executions: int = 0
    tracking_successes: int = 0
    tracking_failures: int = 0
    total_tracking_time_ms: float = 0.0
    total_mediapipe_time_ms: float = 0.0
    
    @property
    def skip_ratio(self) -> float:
        """MediaPipe スキップ率"""
        if self.total_frames == 0:
            return 0.0
        return 1.0 - (self.mediapipe_executions / self.total_frames)
    
    @property
    def success_rate(self) -> float:
        """トラッキング成功率"""
        total_attempts = self.tracking_successes + self.tracking_failures
        if total_attempts == 0:
            return 0.0
        return self.tracking_successes / total_attempts


@dataclass
class Hand2D:
    """2D手検出結果（レガシー互換）"""
    landmarks: np.ndarray  # (21, 2) MediaPipe landmarks
    handedness: HandednessType
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class Hand3D:
    """3D手情報（レガシー互換）"""
    landmarks_3d: np.ndarray  # (21, 3) 3D landmarks
    position: np.ndarray      # (3,) 手の中心位置
    velocity: np.ndarray      # (3,) 速度ベクトル
    handedness: HandednessType
    confidence: float
    tracking_state: TrackingState
    tracking_id: int


# =============================================================================
# 衝突システム型定義
# =============================================================================

class SearchStrategy(Enum):
    """検索戦略の列挙"""
    SPHERE_QUERY = "sphere"
    FRUSTUM_QUERY = "frustum"
    ADAPTIVE_RADIUS = "adaptive"
    PREDICTIVE = "predictive"


class CollisionType(Enum):
    """衝突タイプの列挙"""
    NO_COLLISION = "none"
    VERTEX_COLLISION = "vertex"
    EDGE_COLLISION = "edge"
    FACE_COLLISION = "face"


@dataclass
class SearchResult:
    """検索結果データ構造"""
    triangle_indices: List[int]
    distances: List[float]
    search_time_ms: float
    query_point: np.ndarray
    search_radius: float
    num_nodes_visited: int
    
    @property
    def num_triangles(self) -> int:
        """検索された三角形数を取得"""
        return len(self.triangle_indices)
    
    @property
    def closest_triangle(self) -> Optional[int]:
        """最近傍三角形インデックスを取得"""
        if not self.triangle_indices:
            return None
        min_idx = np.argmin(self.distances)
        return self.triangle_indices[min_idx]
    
    @property
    def closest_distance(self) -> Optional[float]:
        """最近傍距離を取得"""
        if not self.distances:
            return None
        return min(self.distances)


@dataclass
class ContactPoint:
    """接触点情報"""
    position: np.ndarray
    normal: np.ndarray
    depth: float
    triangle_index: int
    barycentric: np.ndarray
    collision_type: CollisionType
    
    @property
    def penetration_vector(self) -> np.ndarray:
        """侵入ベクトルを取得"""
        return self.normal * self.depth


@dataclass
class CollisionInfo:
    """衝突情報の総合データ"""
    has_collision: bool
    contact_points: List[ContactPoint]
    closest_point: Optional[ContactPoint]
    total_penetration_depth: float
    collision_normal: np.ndarray
    collision_time_ms: float
    
    @property
    def num_contacts(self) -> int:
        """接触点数を取得"""
        return len(self.contact_points)
    
    @property
    def max_penetration_depth(self) -> float:
        """最大侵入深度を取得"""
        if not self.contact_points:
            return 0.0
        return max(cp.depth for cp in self.contact_points)


# =============================================================================
# メッシュシステム型定義
# =============================================================================

class IndexType(Enum):
    """空間インデックスタイプ"""
    OCTREE = "octree"
    KD_TREE = "kdtree"
    BVH = "bvh"
    UNIFORM_GRID = "uniform_grid"


@dataclass
class TriangleMeshInfo:
    """三角形メッシュ情報"""
    num_vertices: int
    num_triangles: int
    bounding_box: Tuple[np.ndarray, np.ndarray]  # (min, max)
    surface_area: float
    volume: float


# =============================================================================
# 音響システム型定義
# =============================================================================

class ScaleType(Enum):
    """音階タイプ"""
    CHROMATIC = "chromatic"
    MAJOR = "major"
    MINOR = "minor"
    PENTATONIC = "pentatonic"
    BLUES = "blues"


class InstrumentType(Enum):
    """楽器タイプ"""
    SINE = "sine"
    SAWTOOTH = "sawtooth"
    SQUARE = "square"
    TRIANGLE = "triangle"
    NOISE = "noise"
    BELL = "bell"
    ORGAN = "organ"


@dataclass
class AudioEvent:
    """音響イベント"""
    frequency: float
    amplitude: float
    duration: float
    instrument: InstrumentType
    timestamp: float
    spatial_position: Optional[np.ndarray] = None


# =============================================================================
# デバッグ・UI型定義
# =============================================================================

class ViewerMode(Enum):
    """ビューワーモード"""
    DEPTH_ONLY = "depth"
    COLOR_ONLY = "color"
    DUAL = "dual"
    MESH = "mesh"
    COLLISION = "collision"
    DEBUG = "debug"


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    fps: float
    frame_time_ms: float
    cpu_usage: float
    memory_usage: int  # bytes
    active_threads: int
    
    @property
    def memory_usage_mb(self) -> float:
        """メモリ使用量（MB）"""
        return self.memory_usage / (1024 * 1024) 