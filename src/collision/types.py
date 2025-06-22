#!/usr/bin/env python3
"""
衝突検出フェーズの共通データ構造

このモジュールは、衝突検出に関連するdataclassやEnumを定義し、
モジュール間の循環参照を防ぐために使用されます。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class SearchStrategy(Enum):
    """検索戦略の列挙"""
    SPHERE_QUERY = "sphere"
    FRUSTUM_QUERY = "frustum"
    ADAPTIVE_RADIUS = "adaptive"
    PREDICTIVE = "predictive"


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


class CollisionType(Enum):
    """衝突タイプの列挙"""
    NO_COLLISION = "none"
    VERTEX_COLLISION = "vertex"
    EDGE_COLLISION = "edge"
    FACE_COLLISION = "face"


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