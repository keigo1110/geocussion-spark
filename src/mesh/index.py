#!/usr/bin/env python3
"""
空間インデックス

高速な空間検索のためのBVH（Bounding Volume Hierarchy）や
KD-Treeなどのデータ構造を提供します。
衝突検出での三角形検索を高速化します。
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from enum import Enum
import numpy as np
from scipy.spatial import cKDTree

from .delaunay import TriangleMesh


class IndexType(Enum):
    """インデックスタイプの列挙"""
    BVH = "bvh"                # Bounding Volume Hierarchy
    KDTREE = "kdtree"          # KD-Tree
    OCTREE = "octree"          # Octree
    UNIFORM_GRID = "grid"      # 均等グリッド


@dataclass
class BoundingBox:
    """軸並行バウンディングボックス"""
    min_point: np.ndarray      # 最小点 (3,)
    max_point: np.ndarray      # 最大点 (3,)
    
    @property
    def center(self) -> np.ndarray:
        """中心点を取得"""
        return (self.min_point + self.max_point) * 0.5
    
    @property
    def size(self) -> np.ndarray:
        """サイズを取得"""
        return self.max_point - self.min_point
    
    @property
    def volume(self) -> float:
        """体積を取得"""
        size = self.size
        return size[0] * size[1] * size[2]
    
    def contains_point(self, point: np.ndarray) -> bool:
        """点が含まれるかチェック"""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def intersects_box(self, other: 'BoundingBox') -> bool:
        """他のバウンディングボックスと交差するかチェック"""
        return np.all(self.min_point <= other.max_point) and np.all(self.max_point >= other.min_point)
    
    def intersects_sphere(self, center: np.ndarray, radius: float) -> bool:
        """球と交差するかチェック"""
        # 最近接点を計算
        closest_point = np.clip(center, self.min_point, self.max_point)
        distance = np.linalg.norm(center - closest_point)
        return distance <= radius
    
    def expand(self, margin: float) -> 'BoundingBox':
        """マージンを追加してボックスを拡張"""
        return BoundingBox(
            min_point=self.min_point - margin,
            max_point=self.max_point + margin
        )
    
    @staticmethod
    def from_points(points: np.ndarray) -> 'BoundingBox':
        """点群からバウンディングボックスを作成"""
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        return BoundingBox(min_point, max_point)
    
    @staticmethod
    def union(box1: 'BoundingBox', box2: 'BoundingBox') -> 'BoundingBox':
        """2つのボックスの和集合"""
        min_point = np.minimum(box1.min_point, box2.min_point)
        max_point = np.maximum(box1.max_point, box2.max_point)
        return BoundingBox(min_point, max_point)


@dataclass
class BVHNode:
    """BVHノード"""
    bounding_box: BoundingBox
    triangle_indices: Optional[List[int]] = None  # リーフノードの三角形インデックス
    left_child: Optional['BVHNode'] = None
    right_child: Optional['BVHNode'] = None
    
    @property
    def is_leaf(self) -> bool:
        """リーフノードかどうか"""
        return self.triangle_indices is not None
    
    @property
    def num_triangles(self) -> int:
        """含まれる三角形数"""
        return len(self.triangle_indices) if self.triangle_indices else 0


class SpatialIndex:
    """空間インデックス基底クラス"""
    
    def __init__(
        self,
        mesh: TriangleMesh,
        index_type: IndexType = IndexType.BVH,
        max_triangles_per_leaf: int = 10,  # リーフノードあたりの最大三角形数
        max_depth: int = 20,               # 最大深度
        split_method: str = "median"       # 分割方法
    ):
        """
        初期化
        
        Args:
            mesh: 入力メッシュ
            index_type: インデックスタイプ
            max_triangles_per_leaf: リーフノードあたりの最大三角形数
            max_depth: 最大深度
            split_method: 分割方法 ("median", "mean", "sah")
        """
        self.mesh = mesh
        self.index_type = index_type
        self.max_triangles_per_leaf = max_triangles_per_leaf
        self.max_depth = max_depth
        self.split_method = split_method
        
        # インデックス構造
        self.root_node: Optional[BVHNode] = None
        self.kdtree: Optional[cKDTree] = None
        self.triangle_centers: Optional[np.ndarray] = None
        self.triangle_bounds: Optional[List[BoundingBox]] = None
        
        # パフォーマンス統計
        self.stats = {
            'build_time_ms': 0.0,
            'num_nodes': 0,
            'max_depth_reached': 0,
            'total_queries': 0,
            'total_query_time_ms': 0.0,
            'average_query_time_ms': 0.0
        }
        
        # インデックス構築
        self._build_index()
    
    def _build_index(self):
        """インデックスを構築"""
        start_time = time.perf_counter()
        
        if self.mesh.num_triangles == 0:
            return
        
        # 三角形の中心とバウンディングボックスを計算
        self.triangle_centers = self.mesh.get_triangle_centers()
        self.triangle_bounds = self._calculate_triangle_bounds()
        
        if self.index_type == IndexType.BVH:
            self.root_node = self._build_bvh()
        elif self.index_type == IndexType.KDTREE:
            self.kdtree = cKDTree(self.triangle_centers)
        else:
            # その他のインデックス（実装省略）
            self.root_node = self._build_bvh()  # デフォルトはBVH
        
        build_time = (time.perf_counter() - start_time) * 1000
        self.stats['build_time_ms'] = build_time
    
    def query_point(self, point: np.ndarray, radius: float = 0.1) -> List[int]:
        """
        点の近傍三角形を検索
        
        Args:
            point: 検索点
            radius: 検索半径
            
        Returns:
            近傍三角形のインデックスリスト
        """
        start_time = time.perf_counter()
        
        if self.index_type == IndexType.BVH:
            result = self._query_bvh_point(point, radius)
        elif self.index_type == IndexType.KDTREE:
            result = self._query_kdtree_point(point, radius)
        else:
            result = []
        
        # パフォーマンス統計更新
        query_time = (time.perf_counter() - start_time) * 1000
        self._update_query_stats(query_time)
        
        return result
    
    def query_sphere(self, center: np.ndarray, radius: float) -> List[int]:
        """
        球の近傍三角形を検索
        
        Args:
            center: 球の中心
            radius: 球の半径
            
        Returns:
            交差する三角形のインデックスリスト
        """
        return self.query_point(center, radius)
    
    def query_ray(self, origin: np.ndarray, direction: np.ndarray, max_distance: float = np.inf) -> List[int]:
        """
        レイと交差する三角形を検索
        
        Args:
            origin: レイの原点
            direction: レイの方向（正規化済み）
            max_distance: 最大距離
            
        Returns:
            交差する可能性のある三角形のインデックスリスト
        """
        start_time = time.perf_counter()
        
        if self.index_type == IndexType.BVH:
            result = self._query_bvh_ray(origin, direction, max_distance)
        else:
            # 簡単な実装: 全三角形をチェック
            result = list(range(self.mesh.num_triangles))
        
        # パフォーマンス統計更新
        query_time = (time.perf_counter() - start_time) * 1000
        self._update_query_stats(query_time)
        
        return result
    
    def _calculate_triangle_bounds(self) -> List[BoundingBox]:
        """各三角形のバウンディングボックスを計算"""
        bounds = []
        
        for triangle in self.mesh.triangles:
            triangle_vertices = self.mesh.vertices[triangle]
            bbox = BoundingBox.from_points(triangle_vertices)
            bounds.append(bbox)
        
        return bounds
    
    def _build_bvh(self) -> Optional[BVHNode]:
        """BVHを構築"""
        triangle_indices = list(range(self.mesh.num_triangles))
        self.stats['num_nodes'] = 0
        self.stats['max_depth_reached'] = 0
        
        return self._build_bvh_recursive(triangle_indices, 0)
    
    def _build_bvh_recursive(self, triangle_indices: List[int], depth: int) -> Optional[BVHNode]:
        """BVHを再帰的に構築"""
        if not triangle_indices:
            return None
        
        self.stats['num_nodes'] += 1
        self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], depth)
        
        # バウンディングボックスを計算
        bounding_box = self._calculate_bounding_box(triangle_indices)
        
        # リーフノードの条件
        if len(triangle_indices) <= self.max_triangles_per_leaf or depth >= self.max_depth:
            return BVHNode(
                bounding_box=bounding_box,
                triangle_indices=triangle_indices
            )
        
        # 分割軸と分割点を決定
        split_axis, split_value = self._choose_split(triangle_indices, bounding_box)
        
        # 三角形を分割
        left_indices, right_indices = self._split_triangles(
            triangle_indices, split_axis, split_value
        )
        
        # 分割に失敗した場合はリーフノードにする
        if not left_indices or not right_indices:
            return BVHNode(
                bounding_box=bounding_box,
                triangle_indices=triangle_indices
            )
        
        # 子ノードを再帰的に構築
        left_child = self._build_bvh_recursive(left_indices, depth + 1)
        right_child = self._build_bvh_recursive(right_indices, depth + 1)
        
        return BVHNode(
            bounding_box=bounding_box,
            left_child=left_child,
            right_child=right_child
        )
    
    def _calculate_bounding_box(self, triangle_indices: List[int]) -> BoundingBox:
        """三角形群のバウンディングボックスを計算"""
        if not triangle_indices:
            return BoundingBox(np.zeros(3), np.zeros(3))
        
        all_vertices = []
        for tri_idx in triangle_indices:
            triangle = self.mesh.triangles[tri_idx]
            all_vertices.extend(self.mesh.vertices[triangle])
        
        return BoundingBox.from_points(np.array(all_vertices))
    
    def _choose_split(self, triangle_indices: List[int], bounding_box: BoundingBox) -> Tuple[int, float]:
        """分割軸と分割点を選択"""
        # 最も長い軸を選択
        size = bounding_box.size
        split_axis = np.argmax(size)
        
        # 分割点を決定
        if self.split_method == "median":
            # 中央値分割
            centers = [self.triangle_centers[i][split_axis] for i in triangle_indices]
            split_value = np.median(centers)
        elif self.split_method == "mean":
            # 平均分割
            centers = [self.triangle_centers[i][split_axis] for i in triangle_indices]
            split_value = np.mean(centers)
        else:  # "sah" - Surface Area Heuristic（簡略版）
            split_value = bounding_box.center[split_axis]
        
        return split_axis, split_value
    
    def _split_triangles(self, triangle_indices: List[int], split_axis: int, split_value: float) -> Tuple[List[int], List[int]]:
        """三角形を分割"""
        left_indices = []
        right_indices = []
        
        for tri_idx in triangle_indices:
            center = self.triangle_centers[tri_idx]
            
            if center[split_axis] <= split_value:
                left_indices.append(tri_idx)
            else:
                right_indices.append(tri_idx)
        
        return left_indices, right_indices
    
    # ------------------------------
    #  BVH 検索 (早期打ち切り付き)
    # ------------------------------
    def _query_bvh_point(self, point: np.ndarray, radius: float) -> List[int]:
        """BVHによる点検索 (max_nodes_per_query で早期停止)"""
        if self.root_node is None:
            return []

        result: List[int] = []
        max_nodes = getattr(self, "max_nodes_per_query", None)
        # nodes_visited をリストで渡し、参照でカウント
        counter = [0]  # list をミュータブル参照として使用
        self._query_bvh_point_recursive(self.root_node, point, radius, result, max_nodes, counter)
        return result
    
    def _query_bvh_point_recursive(self, node: BVHNode, point: np.ndarray, radius: float, result: List[int], max_nodes: Optional[int], counter: List[int]):
        """再帰的BVH検索"""
        # 早期打ち切り
        counter[0] += 1
        if max_nodes is not None and counter[0] > max_nodes:
            return
        
        # バウンディングボックスと球の交差チェック
        if not node.bounding_box.intersects_sphere(point, radius):
            return
        
        if node.is_leaf:
            # リーフノード: 実際の距離チェック
            for tri_idx in node.triangle_indices:
                triangle = self.mesh.triangles[tri_idx]
                triangle_vertices = self.mesh.vertices[triangle]
                
                # 三角形の最近接点への距離をチェック
                if self._point_triangle_distance(point, triangle_vertices) <= radius:
                    result.append(tri_idx)
        else:
            # 内部ノード: 子ノードを再帰的に探索
            # max_nodes制限を対称的にチェックして探索
            if node.left_child and counter[0] < max_nodes:
                self._query_bvh_point_recursive(node.left_child, point, radius, result, max_nodes, counter)
            if node.right_child and counter[0] < max_nodes:
                self._query_bvh_point_recursive(node.right_child, point, radius, result, max_nodes, counter)
    
    def _query_bvh_ray(self, origin: np.ndarray, direction: np.ndarray, max_distance: float) -> List[int]:
        """BVHを使ってレイ検索"""
        if self.root_node is None:
            return []
        
        result = []
        self._query_bvh_ray_recursive(self.root_node, origin, direction, max_distance, result)
        return result
    
    def _query_bvh_ray_recursive(self, node: BVHNode, origin: np.ndarray, direction: np.ndarray, max_distance: float, result: List[int]):
        """BVHレイ検索の再帰処理"""
        # バウンディングボックスとレイの交差チェック
        if not self._ray_box_intersect(origin, direction, node.bounding_box, max_distance):
            return
        
        if node.is_leaf:
            # リーフノード: 三角形リストに追加
            result.extend(node.triangle_indices)
        else:
            # 内部ノード: 子ノードを再帰的に探索
            if node.left_child:
                self._query_bvh_ray_recursive(node.left_child, origin, direction, max_distance, result)
            if node.right_child:
                self._query_bvh_ray_recursive(node.right_child, origin, direction, max_distance, result)
    
    def _query_kdtree_point(self, point: np.ndarray, radius: float) -> List[int]:
        """KD-Treeを使って点の近傍検索"""
        if self.kdtree is None:
            return []
        
        # KD-Treeで近傍検索
        indices = self.kdtree.query_ball_point(point, radius)
        return indices
    
    def _point_triangle_distance(self, point: np.ndarray, triangle_vertices: np.ndarray) -> float:
        """点と三角形の最短距離を計算"""
        # 簡略化: 三角形の重心への距離を使用
        center = np.mean(triangle_vertices, axis=0)
        return np.linalg.norm(point - center)
    
    def _ray_box_intersect(self, origin: np.ndarray, direction: np.ndarray, bbox: BoundingBox, max_distance: float) -> bool:
        """レイとバウンディングボックスの交差判定"""
        # スラブ法を使用
        inv_dir = np.where(np.abs(direction) > 1e-8, 1.0 / direction, np.inf)
        
        t1 = (bbox.min_point - origin) * inv_dir
        t2 = (bbox.max_point - origin) * inv_dir
        
        t_min = np.maximum.reduce(np.minimum(t1, t2))
        t_max = np.minimum.reduce(np.maximum(t1, t2))
        
        return t_max >= 0 and t_min <= t_max and t_min <= max_distance
    
    def _update_query_stats(self, query_time: float):
        """クエリ統計を更新"""
        self.stats['total_queries'] += 1
        self.stats['total_query_time_ms'] += query_time
        self.stats['average_query_time_ms'] = (
            self.stats['total_query_time_ms'] / self.stats['total_queries']
        )
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats['total_queries'] = 0
        self.stats['total_query_time_ms'] = 0.0
        self.stats['average_query_time_ms'] = 0.0


# 便利関数

def build_bvh_index(
    mesh: TriangleMesh,
    max_triangles_per_leaf: int = 10,
    max_depth: int = 20
) -> SpatialIndex:
    """
    BVHインデックスを構築（簡単なインターフェース）
    
    Args:
        mesh: 入力メッシュ
        max_triangles_per_leaf: リーフノードあたりの最大三角形数
        max_depth: 最大深度
        
    Returns:
        BVH空間インデックス
    """
    return SpatialIndex(
        mesh=mesh,
        index_type=IndexType.BVH,
        max_triangles_per_leaf=max_triangles_per_leaf,
        max_depth=max_depth
    )


def query_nearest_triangles(
    index: SpatialIndex,
    point: np.ndarray,
    radius: float = 0.1
) -> List[int]:
    """
    最近傍三角形を検索（簡単なインターフェース）
    
    Args:
        index: 空間インデックス
        point: 検索点
        radius: 検索半径
        
    Returns:
        近傍三角形のインデックスリスト
    """
    return index.query_point(point, radius)


def query_point_in_triangles(
    index: SpatialIndex,
    point: np.ndarray,
    tolerance: float = 1e-6
) -> List[int]:
    """
    点を含む三角形を検索（簡単なインターフェース）
    
    Args:
        index: 空間インデックス
        point: 検索点
        tolerance: 許容誤差
        
    Returns:
        点を含む三角形のインデックスリスト
    """
    return index.query_point(point, tolerance) 