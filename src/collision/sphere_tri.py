#!/usr/bin/env python3
"""
衝突検出 - 球-三角形判定

手を球体としてモデル化し、地形メッシュの三角形との精密な衝突判定と
接触点計算を行う機能を提供します。高精度かつ高速な実装を目指します。
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from enum import Enum
import numpy as np

# 他フェーズとの連携
from ..mesh.delaunay import TriangleMesh
from ..mesh.attributes import MeshAttributes
from ..data_types import CollisionType, ContactPoint, CollisionInfo, SearchResult
from ..constants import (
    COLLISION_TOLERANCE,
    COLLISION_DETECTION_PADDING,
    MAX_CONTACTS_PER_SPHERE,
    NUMERICAL_TOLERANCE,
)


class SphereTriangleCollision:
    """球-三角形衝突判定クラス"""
    
    def __init__(
        self,
        mesh: TriangleMesh,
        mesh_attributes: Optional[MeshAttributes] = None,
        collision_tolerance: float = COLLISION_TOLERANCE,      # 衝突判定の許容誤差
        enable_face_culling: bool = False,      # 裏面カリング
        max_contacts_per_sphere: int = MAX_CONTACTS_PER_SPHERE,      # 球あたりの最大接触点数
        detection_padding: float = COLLISION_DETECTION_PADDING,      # 衝突判定のパディング
    ):
        """
        初期化
        
        Args:
            mesh: 地形メッシュ
            mesh_attributes: メッシュ属性（法線など）
            collision_tolerance: 衝突判定の許容誤差
            enable_face_culling: 裏面カリングを有効にするか
            max_contacts_per_sphere: 球あたりの最大接触点数
        """
        self.mesh = mesh
        self.mesh_attributes = mesh_attributes
        self.collision_tolerance = collision_tolerance
        self.enable_face_culling = enable_face_culling
        self.max_contacts_per_sphere = max_contacts_per_sphere
        self.detection_padding = detection_padding
        
        # パフォーマンス統計
        self.stats = {
            'total_tests': 0,
            'total_test_time_ms': 0.0,
            'average_test_time_ms': 0.0,
            'collisions_detected': 0,
            'collision_rate': 0.0,
            'last_test_time_ms': 0.0,
            'last_num_contacts': 0,
            'vertex_collisions': 0,
            'edge_collisions': 0,
            'face_collisions': 0
        }
    
    def test_sphere_collision(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        search_result: SearchResult
    ) -> CollisionInfo:
        """
        球と近傍三角形の衝突テスト
        
        Args:
            sphere_center: 球の中心座標
            sphere_radius: 球の半径
            search_result: 空間検索結果
            
        Returns:
            衝突情報
        """
        start_time = time.perf_counter()
        
        contact_points = []
        
        # 各三角形との衝突判定
        for i, triangle_idx in enumerate(search_result.triangle_indices):
            # 距離による早期除外
            if search_result.distances[i] > sphere_radius * 2:
                continue
                
            contact_point = self._test_sphere_triangle(
                sphere_center, sphere_radius, triangle_idx
            )
            
            if contact_point is not None:
                contact_points.append(contact_point)
                
                # 最大接触点数制限
                if len(contact_points) >= self.max_contacts_per_sphere:
                    break
        
        # 結果をまとめる
        collision_info = self._create_collision_info(
            contact_points, 
            sphere_center,
            (time.perf_counter() - start_time) * 1000
        )
        
        # 統計更新
        self._update_stats(collision_info)
        
        return collision_info
    
    def batch_test_spheres(
        self,
        sphere_centers: List[np.ndarray],
        sphere_radii: List[float],
        search_results: List[SearchResult]
    ) -> List[CollisionInfo]:
        """
        複数球の一括衝突テスト
        
        Args:
            sphere_centers: 球の中心座標リスト
            sphere_radii: 球の半径リスト
            search_results: 空間検索結果リスト
            
        Returns:
            衝突情報リスト
        """
        results = []
        
        for center, radius, search_result in zip(sphere_centers, sphere_radii, search_results):
            collision_info = self.test_sphere_collision(center, radius, search_result)
            results.append(collision_info)
        
        return results
    
    def _test_sphere_triangle(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        triangle_idx: int
    ) -> Optional[ContactPoint]:
        """単一三角形との衝突テスト"""
        triangle = self.mesh.triangles[triangle_idx]
        vertices = self.mesh.vertices[triangle]
        
        # 最近接点を計算
        closest_point, distance, collision_type, barycentric = self._closest_point_on_triangle(
            sphere_center, vertices
        )
        
        # 衝突判定 (パディングを考慮)
        effective_radius = sphere_radius + self.detection_padding
        penetration_depth = effective_radius - distance
        if penetration_depth <= self.collision_tolerance:
            return None
        
        # 法線ベクトル計算
        if self.mesh_attributes and self.mesh_attributes.triangle_normals is not None:
            # 事前計算された法線を使用
            triangle_normal = self.mesh_attributes.triangle_normals[triangle_idx]
        else:
            # その場で計算
            triangle_normal = self._calculate_triangle_normal(vertices)
        
        # 裏面カリング
        if self.enable_face_culling:
            to_sphere = sphere_center - closest_point
            if np.dot(to_sphere, triangle_normal) < 0:
                return None
        
        # 接触法線を計算
        contact_direction = sphere_center - closest_point
        contact_normal = contact_direction / (distance + 1e-8)
        
        return ContactPoint(
            position=closest_point,
            normal=contact_normal,
            depth=penetration_depth,
            triangle_index=triangle_idx,
            barycentric=barycentric,
            collision_type=collision_type
        )
    
    def _closest_point_on_triangle(
        self,
        point: np.ndarray,
        triangle_vertices: np.ndarray
    ) -> Tuple[np.ndarray, float, CollisionType, np.ndarray]:
        """
        点から三角形への最近接点を計算
        
        Returns:
            (最近接点, 距離, 衝突タイプ, 重心座標)
        """
        v0, v1, v2 = triangle_vertices
        
        # 三角形の辺とベクトル
        edge0 = v1 - v0  # v0 -> v1
        edge1 = v2 - v1  # v1 -> v2  
        edge2 = v0 - v2  # v2 -> v0
        
        # 点から各頂点へのベクトル
        to_v0 = point - v0
        to_v1 = point - v1
        to_v2 = point - v2
        
        # 重心座標を計算
        d00 = np.dot(edge0, edge0)
        d01 = np.dot(edge0, -edge2)  # edge0 と (v2->v0) の内積 = edge0 と (v0->v2) の内積
        d11 = np.dot(-edge2, -edge2)  # (v2->v0) と (v2->v0) の内積
        d20 = np.dot(to_v0, edge0)
        d21 = np.dot(to_v0, -edge2)
        
        denom = d00 * d11 - d01 * d01
        
        if abs(denom) < 1e-8:
            # 退化三角形の場合、最近接頂点を返す
            distances = [
                np.linalg.norm(to_v0),
                np.linalg.norm(to_v1), 
                np.linalg.norm(to_v2)
            ]
            min_idx = np.argmin(distances)
            closest_vertex = [v0, v1, v2][min_idx]
            barycentric = np.zeros(3)
            barycentric[min_idx] = 1.0
            return closest_vertex, distances[min_idx], CollisionType.VERTEX_COLLISION, barycentric
        
        # 重心座標計算
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        barycentric = np.array([u, v, w])
        
        # 三角形内部の場合
        if u >= 0 and v >= 0 and w >= 0:
            closest_point = u * v0 + v * v1 + w * v2
            distance = np.linalg.norm(point - closest_point)
            return closest_point, distance, CollisionType.FACE_COLLISION, barycentric
        
        # 三角形外部の場合、エッジや頂点への投影を計算
        return self._closest_point_on_triangle_boundary(point, triangle_vertices)
    
    def _closest_point_on_triangle_boundary(
        self,
        point: np.ndarray,
        triangle_vertices: np.ndarray
    ) -> Tuple[np.ndarray, float, CollisionType, np.ndarray]:
        """三角形境界への最近接点を計算"""
        v0, v1, v2 = triangle_vertices
        edges = [(v0, v1, 0, 1), (v1, v2, 1, 2), (v2, v0, 2, 0)]
        
        min_distance = float('inf')
        closest_point = None
        collision_type = CollisionType.VERTEX_COLLISION
        best_barycentric = np.array([1.0, 0.0, 0.0])
        
        # 各エッジをチェック
        for start, end, start_idx, end_idx in edges:
            edge_vec = end - start
            edge_length_sq = np.dot(edge_vec, edge_vec)
            
            if edge_length_sq < 1e-8:
                # 退化エッジ（点）
                to_start = point - start
                distance = np.linalg.norm(to_start)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = start
                    collision_type = CollisionType.VERTEX_COLLISION
                    barycentric = np.zeros(3)
                    barycentric[start_idx] = 1.0
                    best_barycentric = barycentric
                continue
            
            # エッジ上の投影点を計算
            to_start = point - start
            t = np.dot(to_start, edge_vec) / edge_length_sq
            t = max(0.0, min(1.0, t))  # [0,1]にクランプ
            
            projected_point = start + t * edge_vec
            distance = np.linalg.norm(point - projected_point)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = projected_point
                
                # 重心座標を計算
                barycentric = np.zeros(3)
                barycentric[start_idx] = 1.0 - t
                barycentric[end_idx] = t
                best_barycentric = barycentric
                
                # 衝突タイプを決定
                if t == 0.0 or t == 1.0:
                    collision_type = CollisionType.VERTEX_COLLISION
                else:
                    collision_type = CollisionType.EDGE_COLLISION
        
        return closest_point, min_distance, collision_type, best_barycentric
    
    def _calculate_triangle_normal(self, vertices: np.ndarray) -> np.ndarray:
        """三角形の法線を計算"""
        v0, v1, v2 = vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0])  # デフォルト法線
        
        return normal / norm
    
    def _create_collision_info(
        self,
        contact_points: List[ContactPoint],
        sphere_center: np.ndarray,
        computation_time_ms: float
    ) -> CollisionInfo:
        """衝突情報をまとめる"""
        has_collision = len(contact_points) > 0
        
        if not has_collision:
            return CollisionInfo(
                has_collision=False,
                contact_points=[],
                closest_point=None,
                total_penetration_depth=0.0,
                collision_normal=np.array([0.0, 0.0, 1.0]),
                collision_time_ms=computation_time_ms
            )
        
        # 最も侵入深度の大きい接触点を選択
        closest_point = max(contact_points, key=lambda cp: cp.depth)
        
        # 総侵入深度
        total_penetration = sum(cp.depth for cp in contact_points)
        
        # 平均衝突法線を計算
        if contact_points:
            avg_normal = np.mean([cp.normal for cp in contact_points], axis=0)
            norm = np.linalg.norm(avg_normal)
            collision_normal = avg_normal / (norm + 1e-8)
        else:
            collision_normal = np.array([0.0, 0.0, 1.0])
        
        return CollisionInfo(
            has_collision=True,
            contact_points=contact_points,
            closest_point=closest_point,
            total_penetration_depth=total_penetration,
            collision_normal=collision_normal,
            collision_time_ms=computation_time_ms
        )
    
    def _update_stats(self, collision_info: CollisionInfo):
        """統計を更新"""
        self.stats['total_tests'] += 1
        self.stats['total_test_time_ms'] += collision_info.collision_time_ms
        self.stats['average_test_time_ms'] = (
            self.stats['total_test_time_ms'] / self.stats['total_tests']
        )
        self.stats['last_test_time_ms'] = collision_info.collision_time_ms
        self.stats['last_num_contacts'] = collision_info.num_contacts
        
        if collision_info.has_collision:
            self.stats['collisions_detected'] += 1
            
            # 衝突タイプ別統計
            for contact in collision_info.contact_points:
                if contact.collision_type == CollisionType.VERTEX_COLLISION:
                    self.stats['vertex_collisions'] += 1
                elif contact.collision_type == CollisionType.EDGE_COLLISION:
                    self.stats['edge_collisions'] += 1
                elif contact.collision_type == CollisionType.FACE_COLLISION:
                    self.stats['face_collisions'] += 1
        
        # 衝突率更新
        self.stats['collision_rate'] = self.stats['collisions_detected'] / self.stats['total_tests']
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_tests': 0,
            'total_test_time_ms': 0.0,
            'average_test_time_ms': 0.0,
            'collisions_detected': 0,
            'collision_rate': 0.0,
            'last_test_time_ms': 0.0,
            'last_num_contacts': 0,
            'vertex_collisions': 0,
            'edge_collisions': 0,
            'face_collisions': 0
        }


# 便利関数

def check_sphere_triangle(
    sphere_center: np.ndarray,
    sphere_radius: float,
    triangle_vertices: np.ndarray
) -> Optional[ContactPoint]:
    """
    球-三角形衝突テスト（簡単なインターフェース）
    
    Args:
        sphere_center: 球の中心
        sphere_radius: 球の半径
        triangle_vertices: 三角形の頂点 (3, 3)
        
    Returns:
        接触点（衝突がない場合はNone）
    """
    # 簡易メッシュを作成
    vertices = triangle_vertices
    triangles = np.array([[0, 1, 2]])
    
    from ..mesh.delaunay import TriangleMesh
    mesh = TriangleMesh(vertices=vertices, triangles=triangles)
    
    collision_tester = SphereTriangleCollision(mesh)
    
    # ダミー検索結果を作成
    from .search import SearchResult
    search_result = SearchResult([0], [0.0], 0.0, sphere_center, sphere_radius, 1)
    
    collision_info = collision_tester.test_sphere_collision(
        sphere_center, sphere_radius, search_result
    )
    
    return collision_info.closest_point


def calculate_contact_point(
    sphere_center: np.ndarray,
    sphere_radius: float,
    triangle_vertices: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    接触点と侵入深度を計算（簡単なインターフェース）
    
    Args:
        sphere_center: 球の中心
        sphere_radius: 球の半径
        triangle_vertices: 三角形の頂点
        
    Returns:
        (接触点, 侵入深度)
    """
    contact_point = check_sphere_triangle(sphere_center, sphere_radius, triangle_vertices)
    
    if contact_point is None:
        return np.zeros(3), 0.0
    
    return contact_point.position, contact_point.depth


def point_triangle_distance(point: np.ndarray, triangle_vertices: np.ndarray) -> float:
    """
    点と三角形の最短距離を計算（最適化版への移行）
    
    Args:
        point: 3D点の座標
        triangle_vertices: 三角形の頂点座標 (3, 3)
        
    Returns:
        最短距離
    """
    # 最適化された距離計算を使用
    from .distance import get_distance_calculator
    calculator = get_distance_calculator()
    return calculator.calculate_point_triangle_distance(point, triangle_vertices)


def batch_collision_test(
    sphere_centers: List[np.ndarray],
    sphere_radii: List[float],
    mesh: TriangleMesh,
    search_results: List[SearchResult]
) -> List[CollisionInfo]:
    """
    一括衝突テスト（簡単なインターフェース）
    
    Args:
        sphere_centers: 球の中心リスト
        sphere_radii: 球の半径リスト
        mesh: メッシュ
        search_results: 検索結果リスト
        
    Returns:
        衝突情報リスト
    """
    collision_tester = SphereTriangleCollision(mesh)
    return collision_tester.batch_test_spheres(sphere_centers, sphere_radii, search_results) 