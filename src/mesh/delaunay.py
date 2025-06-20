#!/usr/bin/env python3
"""
Delaunay三角形分割

ハイトマップや2D点から品質の良い三角形メッシュを生成する機能を提供します。
scipy.spatial.Delaunayを使用してロバストな三角形分割を行います。
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import cv2

from .projection import HeightMap


@dataclass
class TriangleMesh:
    """三角形メッシュデータ構造"""
    vertices: np.ndarray       # 頂点座標 (N, 3) - (x, y, z)
    triangles: np.ndarray      # 三角形インデックス (M, 3)
    vertex_colors: Optional[np.ndarray] = None  # 頂点色 (N, 3)
    triangle_normals: Optional[np.ndarray] = None  # 三角形法線 (M, 3)
    vertex_normals: Optional[np.ndarray] = None   # 頂点法線 (N, 3)
    
    @property
    def num_vertices(self) -> int:
        """頂点数を取得"""
        return len(self.vertices)
    
    @property
    def num_triangles(self) -> int:
        """三角形数を取得"""
        return len(self.triangles)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """バウンディングボックスを取得"""
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds
    
    def get_triangle_centers(self) -> np.ndarray:
        """三角形の重心を計算"""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        return (v0 + v1 + v2) / 3.0
    
    def get_triangle_areas(self) -> np.ndarray:
        """三角形の面積を計算"""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        
        # ベクトル外積で面積計算
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        
        # 3D外積の長さは2倍の面積
        if cross.ndim == 1:
            areas = np.linalg.norm(cross) / 2.0
        else:
            areas = np.linalg.norm(cross, axis=1) / 2.0
        
        return areas


class DelaunayTriangulator:
    """Delaunay三角形分割クラス"""
    
    def __init__(
        self,
        max_edge_length: float = 0.1,     # 最大エッジ長
        min_triangle_area: float = 1e-6,  # 最小三角形面積
        adaptive_sampling: bool = True,    # 適応的サンプリング
        boundary_points: bool = True,      # 境界点追加
        quality_threshold: float = 0.5     # 品質閾値（0-1）
    ):
        """
        初期化
        
        Args:
            max_edge_length: 最大エッジ長（長すぎる三角形を除去）
            min_triangle_area: 最小三角形面積（小さすぎる三角形を除去）
            adaptive_sampling: 適応的サンプリングを行うか
            boundary_points: 境界点を追加するか
            quality_threshold: 三角形品質の閾値
        """
        self.max_edge_length = max_edge_length
        self.min_triangle_area = min_triangle_area
        self.adaptive_sampling = adaptive_sampling
        self.boundary_points = boundary_points
        self.quality_threshold = quality_threshold
        
        # パフォーマンス統計
        self.stats = {
            'total_triangulations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_points': 0,
            'last_num_triangles': 0,
            'last_quality_score': 0.0
        }
    
    def triangulate_heightmap(self, heightmap: HeightMap) -> TriangleMesh:
        """
        ハイトマップから三角形メッシュを生成
        
        Args:
            heightmap: 入力ハイトマップ
            
        Returns:
            三角形メッシュ
        """
        start_time = time.perf_counter()
        
        # 有効な点を抽出
        valid_points = self._extract_valid_points(heightmap)
        
        if len(valid_points) < 3:
            raise ValueError("Not enough valid points for triangulation")
        
        # 適応的サンプリング
        if self.adaptive_sampling:
            sampled_points = self._adaptive_sampling(valid_points, heightmap)
        else:
            sampled_points = valid_points
        
        # 境界点追加
        if self.boundary_points:
            sampled_points = self._add_boundary_points(sampled_points, heightmap)
        
        # Delaunay三角形分割
        mesh = self._perform_delaunay(sampled_points)
        
        # 品質フィルタリング
        mesh = self._filter_low_quality_triangles(mesh)
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        quality_score = self._calculate_mesh_quality(mesh)
        self._update_stats(elapsed_ms, len(sampled_points), mesh.num_triangles, quality_score)
        
        return mesh
    
    def triangulate_points(self, points: np.ndarray) -> TriangleMesh:
        """
        2D/3D点群から三角形メッシュを生成
        
        Args:
            points: 点群データ (N, 2) または (N, 3)
            
        Returns:
            三角形メッシュ
        """
        start_time = time.perf_counter()
        
        if points.shape[1] == 2:
            # 2D点の場合、Z=0を追加
            points_3d = np.column_stack([points, np.zeros(len(points))])
        else:
            points_3d = points
        
        # Delaunay三角形分割
        mesh = self._perform_delaunay(points_3d)
        
        # 品質フィルタリング
        mesh = self._filter_low_quality_triangles(mesh)
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        quality_score = self._calculate_mesh_quality(mesh)
        self._update_stats(elapsed_ms, len(points_3d), mesh.num_triangles, quality_score)
        
        return mesh
    
    def _extract_valid_points(self, heightmap: HeightMap) -> np.ndarray:
        """ハイトマップから有効な3D点を抽出"""
        height, width = heightmap.shape
        valid_mask = heightmap.valid_mask
        
        # グリッド座標生成
        y_coords, x_coords = np.meshgrid(
            np.arange(height), np.arange(width), indexing='ij'
        )
        
        # 有効ピクセルのインデックス
        valid_indices = np.where(valid_mask)
        
        # 世界座標に変換
        points_3d = []
        for row, col in zip(valid_indices[0], valid_indices[1]):
            world_x, world_y = heightmap.get_world_coordinates(row, col)
            world_z = heightmap.heights[row, col]
            points_3d.append([world_x, world_y, world_z])
        
        return np.array(points_3d)
    
    def _adaptive_sampling(self, points: np.ndarray, heightmap: HeightMap) -> np.ndarray:
        """適応的サンプリング（密度に応じてサンプリング率を調整）"""
        if len(points) < 100:
            return points
        
        # 高度の変化が大きい領域はより密にサンプリング
        heights = points[:, 2]
        gradient_magnitude = np.abs(np.gradient(heights))
        
        # 勾配に基づく重要度計算
        importance = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # 重要度の高い点を優先的に選択
        num_samples = min(len(points), int(len(points) * 0.7))
        probabilities = importance + 0.1  # 最低確率を保証
        probabilities /= np.sum(probabilities)
        
        try:
            sampled_indices = np.random.choice(
                len(points), 
                size=num_samples, 
                replace=False, 
                p=probabilities
            )
            return points[sampled_indices]
        except:
            # 確率計算に失敗した場合は均等サンプリング
            step = max(1, len(points) // num_samples)
            return points[::step]
    
    def _add_boundary_points(self, points: np.ndarray, heightmap: HeightMap) -> np.ndarray:
        """境界点を追加（凸包を改善）"""
        if len(points) < 4:
            return points
        
        # XY平面での2D凸包を計算
        xy_points = points[:, :2]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy_points)
            boundary_indices = hull.vertices
            
            # 境界点間に追加点を挿入
            boundary_points = points[boundary_indices]
            additional_points = []
            
            for i in range(len(boundary_points)):
                p1 = boundary_points[i]
                p2 = boundary_points[(i + 1) % len(boundary_points)]
                
                # 境界エッジが長い場合、中間点を追加
                edge_length = np.linalg.norm(p2[:2] - p1[:2])
                if edge_length > self.max_edge_length:
                    mid_point = (p1 + p2) / 2
                    additional_points.append(mid_point)
            
            if additional_points:
                return np.vstack([points, np.array(additional_points)])
            
        except:
            pass  # ConvexHullに失敗した場合はそのまま返す
        
        return points
    
    def _perform_delaunay(self, points: np.ndarray) -> TriangleMesh:
        """Delaunay三角形分割を実行"""
        if len(points) < 3:
            raise ValueError("At least 3 points required for triangulation")
        
        # 2D投影でDelaunay分割
        xy_points = points[:, :2]
        
        try:
            delaunay = Delaunay(xy_points)
            triangles = delaunay.simplices
            
            # 有効な三角形のみを保持
            valid_triangles = []
            for tri in triangles:
                if self._is_valid_triangle(points[tri]):
                    valid_triangles.append(tri)
            
            if not valid_triangles:
                # フォールバック: 全ての三角形を使用
                print(f"Warning: No triangles passed quality filter, using all {len(triangles)} triangles")
                valid_triangles = triangles.tolist()
            
            triangles = np.array(valid_triangles)
            
            return TriangleMesh(
                vertices=points,
                triangles=triangles
            )
            
        except Exception as e:
            raise ValueError(f"Delaunay triangulation failed: {e}")
    
    def _is_valid_triangle(self, triangle_vertices: np.ndarray) -> bool:
        """三角形の有効性をチェック"""
        if len(triangle_vertices) != 3:
            return False
        
        # 面積チェック
        v0, v1, v2 = triangle_vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1[:2], edge2[:2])  # 2D外積
        area = abs(cross) / 2.0
        
        # 非常に小さい三角形のみ除去
        if area < 1e-12:
            return False
        
        # エッジ長チェック（より緩い条件）
        edge_lengths = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2)
        ]
        
        # 非常に長いエッジのみ除去
        max_allowed_length = self.max_edge_length * 2.0  # より緩い条件
        if any(length > max_allowed_length for length in edge_lengths):
            return False
        
        # 縦横比チェック（より緩い条件）
        min_edge = min(edge_lengths)
        max_edge = max(edge_lengths)
        
        # ゼロ除算を避ける
        if max_edge < 1e-12:
            return False
            
        aspect_ratio = min_edge / max_edge
        
        # 極端に細い三角形のみ除去
        if aspect_ratio < 0.01:  # より緩い条件
            return False
        
        return True
    
    def _filter_low_quality_triangles(self, mesh: TriangleMesh) -> TriangleMesh:
        """低品質な三角形を除去"""
        if mesh.num_triangles == 0:
            return mesh
        
        # 三角形品質を計算
        qualities = self._calculate_triangle_qualities(mesh)
        
        # 閾値以上の品質の三角形のみを保持
        good_triangles_mask = qualities >= self.quality_threshold
        good_triangles = mesh.triangles[good_triangles_mask]
        
        if len(good_triangles) == 0:
            # 全て除去された場合は元のメッシュを返す
            return mesh
        
        # 使用される頂点のみを保持
        used_vertices = np.unique(good_triangles.flatten())
        new_vertices = mesh.vertices[used_vertices]
        
        # 三角形インデックスを再マッピング
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        new_triangles = np.array([
            [vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]]
            for tri in good_triangles
        ])
        
        return TriangleMesh(
            vertices=new_vertices,
            triangles=new_triangles
        )
    
    def _calculate_triangle_qualities(self, mesh: TriangleMesh) -> np.ndarray:
        """三角形の品質を計算（0-1、1が最高品質）"""
        qualities = []
        
        for triangle in mesh.triangles:
            v0, v1, v2 = mesh.vertices[triangle]
            
            # エッジ長
            edge_lengths = [
                np.linalg.norm(v1 - v0),
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v0 - v2)
            ]
            
            # 面積
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1[:2], edge2[:2])
            area = abs(cross) / 2.0
            
            # 品質指標（面積 / 最長エッジ^2）
            max_edge = max(edge_lengths)
            quality = (4 * np.sqrt(3) * area) / (sum(e*e for e in edge_lengths))
            quality = max(0.0, min(1.0, quality))  # 0-1にクランプ
            
            qualities.append(quality)
        
        return np.array(qualities)
    
    def _calculate_mesh_quality(self, mesh: TriangleMesh) -> float:
        """メッシュ全体の品質を計算"""
        if mesh.num_triangles == 0:
            return 0.0
        
        qualities = self._calculate_triangle_qualities(mesh)
        return np.mean(qualities)
    
    def _update_stats(self, elapsed_ms: float, num_points: int, num_triangles: int, quality: float):
        """パフォーマンス統計更新"""
        self.stats['total_triangulations'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_triangulations']
        self.stats['last_num_points'] = num_points
        self.stats['last_num_triangles'] = num_triangles
        self.stats['last_quality_score'] = quality
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_triangulations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_points': 0,
            'last_num_triangles': 0,
            'last_quality_score': 0.0
        }


# 便利関数

def create_mesh_from_heightmap(
    heightmap: HeightMap,
    max_edge_length: float = 0.05,
    quality_threshold: float = 0.5
) -> TriangleMesh:
    """
    ハイトマップから三角形メッシュを作成（簡単なインターフェース）
    
    Args:
        heightmap: 入力ハイトマップ
        max_edge_length: 最大エッジ長
        quality_threshold: 品質閾値
        
    Returns:
        三角形メッシュ
    """
    triangulator = DelaunayTriangulator(
        max_edge_length=max_edge_length,
        quality_threshold=quality_threshold
    )
    return triangulator.triangulate_heightmap(heightmap)


def triangulate_points(
    points: np.ndarray,
    max_edge_length: float = 0.1
) -> TriangleMesh:
    """
    点群からDelaunay三角形分割を実行（簡単なインターフェース）
    
    Args:
        points: 点群データ (N, 2) または (N, 3)
        max_edge_length: 最大エッジ長
        
    Returns:
        三角形メッシュ
    """
    triangulator = DelaunayTriangulator(max_edge_length=max_edge_length)
    return triangulator.triangulate_points(points) 