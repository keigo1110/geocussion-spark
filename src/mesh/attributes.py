#!/usr/bin/env python3
"""
メッシュ属性計算

三角形メッシュの法線、曲率、勾配などの幾何学的属性を計算する機能を提供します。
衝突検出やレンダリングで使用される重要な情報を生成します。
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse

from .delaunay import TriangleMesh


@dataclass
class MeshAttributes:
    """メッシュ属性データ構造"""
    vertex_normals: np.ndarray           # 頂点法線 (N, 3)
    triangle_normals: np.ndarray         # 三角形法線 (M, 3)
    vertex_curvatures: np.ndarray        # 頂点曲率 (N,) - 平均曲率
    gaussian_curvatures: np.ndarray      # ガウス曲率 (N,)
    mean_curvatures: np.ndarray          # 平均曲率 (N,)
    gradients: np.ndarray                # 勾配ベクトル (N, 3)
    gradient_magnitudes: np.ndarray      # 勾配大きさ (N,)
    triangle_areas: np.ndarray           # 三角形面積 (M,)
    vertex_areas: np.ndarray             # 頂点面積 (N,)
    edge_lengths: Optional[np.ndarray] = None  # エッジ長（必要に応じて）
    
    @property
    def num_vertices(self) -> int:
        """頂点数を取得"""
        return len(self.vertex_normals)
    
    @property
    def num_triangles(self) -> int:
        """三角形数を取得"""
        return len(self.triangle_normals)
    
    def get_surface_roughness(self) -> float:
        """表面粗さを計算"""
        return np.std(self.gradient_magnitudes)
    
    def get_curvature_statistics(self) -> dict:
        """曲率統計を取得"""
        return {
            'mean_curvature_avg': np.mean(self.mean_curvatures),
            'mean_curvature_std': np.std(self.mean_curvatures),
            'gaussian_curvature_avg': np.mean(self.gaussian_curvatures),
            'gaussian_curvature_std': np.std(self.gaussian_curvatures),
            'max_curvature': np.max(self.vertex_curvatures),
            'min_curvature': np.min(self.vertex_curvatures)
        }


class AttributeCalculator:
    """メッシュ属性計算クラス"""
    
    def __init__(
        self,
        smooth_normals: bool = True,        # 法線を平滑化するか
        curvature_radius: float = 0.05,     # 曲率計算半径
        gradient_method: str = "finite_diff", # 勾配計算手法
        normalize_attributes: bool = True    # 属性を正規化するか
    ):
        """
        初期化
        
        Args:
            smooth_normals: 法線平滑化を行うか
            curvature_radius: 曲率計算の近傍半径
            gradient_method: 勾配計算手法 ("finite_diff" or "laplacian")
            normalize_attributes: 属性を正規化するか
        """
        self.smooth_normals = smooth_normals
        self.curvature_radius = curvature_radius
        self.gradient_method = gradient_method
        self.normalize_attributes = normalize_attributes
        
        # パフォーマンス統計
        self.stats = {
            'total_calculations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_vertices': 0,
            'last_num_triangles': 0,
            'normals_time_ms': 0.0,
            'curvature_time_ms': 0.0,
            'gradient_time_ms': 0.0
        }
    
    def compute_attributes(self, mesh: TriangleMesh) -> MeshAttributes:
        """
        メッシュの全属性を計算
        
        Args:
            mesh: 入力メッシュ
            
        Returns:
            計算されたメッシュ属性
        """
        start_time = time.perf_counter()
        
        if mesh.num_vertices == 0 or mesh.num_triangles == 0:
            raise ValueError("Empty mesh")
        
        # 法線計算
        normals_start = time.perf_counter()
        vertex_normals = self.calculate_vertex_normals(mesh)
        triangle_normals = self.calculate_triangle_normals(mesh)
        normals_time = (time.perf_counter() - normals_start) * 1000
        
        # 曲率計算
        curvature_start = time.perf_counter()
        vertex_curvatures, gaussian_curvatures, mean_curvatures = self.calculate_curvatures(mesh)
        curvature_time = (time.perf_counter() - curvature_start) * 1000
        
        # 勾配計算
        gradient_start = time.perf_counter()
        gradients, gradient_magnitudes = self.calculate_gradients(mesh)
        gradient_time = (time.perf_counter() - gradient_start) * 1000
        
        # 面積計算
        triangle_areas = mesh.get_triangle_areas()
        vertex_areas = self._calculate_vertex_areas(mesh, triangle_areas)
        
        # パフォーマンス統計更新
        total_time = (time.perf_counter() - start_time) * 1000
        self._update_stats(total_time, mesh.num_vertices, mesh.num_triangles, 
                          normals_time, curvature_time, gradient_time)
        
        return MeshAttributes(
            vertex_normals=vertex_normals,
            triangle_normals=triangle_normals,
            vertex_curvatures=vertex_curvatures,
            gaussian_curvatures=gaussian_curvatures,
            mean_curvatures=mean_curvatures,
            gradients=gradients,
            gradient_magnitudes=gradient_magnitudes,
            triangle_areas=triangle_areas,
            vertex_areas=vertex_areas
        )
    
    def calculate_vertex_normals(self, mesh: TriangleMesh) -> np.ndarray:
        """頂点法線を計算"""
        if mesh.vertex_normals is not None and not self.smooth_normals:
            return mesh.vertex_normals
        
        vertex_normals = np.zeros((mesh.num_vertices, 3))
        
        # 三角形法線を計算
        triangle_normals = self.calculate_triangle_normals(mesh)
        triangle_areas = mesh.get_triangle_areas()
        
        # 各三角形の寄与を頂点に加算（面積重み付き）
        for i, triangle in enumerate(mesh.triangles):
            normal = triangle_normals[i]
            area = triangle_areas[i]
            
            for vertex_idx in triangle:
                vertex_normals[vertex_idx] += normal * area
        
        # 正規化
        norms = np.linalg.norm(vertex_normals, axis=1)
        valid_mask = norms > 1e-8
        vertex_normals[valid_mask] /= norms[valid_mask, np.newaxis]
        
        # 無効な法線はZ軸方向に設定
        vertex_normals[~valid_mask] = [0, 0, 1]
        
        # 平滑化
        if self.smooth_normals:
            vertex_normals = self._smooth_vertex_normals(mesh, vertex_normals)
        
        return vertex_normals
    
    def calculate_triangle_normals(self, mesh: TriangleMesh) -> np.ndarray:
        """三角形法線を計算"""
        if mesh.triangle_normals is not None:
            return mesh.triangle_normals
        
        triangle_normals = np.zeros((mesh.num_triangles, 3))
        
        for i, triangle in enumerate(mesh.triangles):
            v0, v1, v2 = mesh.vertices[triangle]
            
            # エッジベクトル
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # 外積で法線計算
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            
            if norm > 1e-8:
                triangle_normals[i] = normal / norm
            else:
                triangle_normals[i] = [0, 0, 1]  # デフォルト法線
        
        return triangle_normals
    
    def calculate_curvatures(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """曲率を計算（主曲率、ガウス曲率、平均曲率）"""
        vertex_normals = self.calculate_vertex_normals(mesh)
        
        # 隣接リストを構築
        adjacency = self._build_vertex_adjacency(mesh)
        
        # 各頂点の曲率を計算
        mean_curvatures = np.zeros(mesh.num_vertices)
        gaussian_curvatures = np.zeros(mesh.num_vertices)
        
        for vertex_idx in range(mesh.num_vertices):
            neighbors = adjacency[vertex_idx]
            
            if len(neighbors) < 3:
                continue
            
            # 平均曲率を計算（離散ラプラシアン）
            mean_curvature = self._calculate_mean_curvature(
                mesh, vertex_idx, neighbors, vertex_normals
            )
            
            # ガウス曲率を計算（角度欠損法）
            gaussian_curvature = self._calculate_gaussian_curvature(
                mesh, vertex_idx, neighbors
            )
            
            mean_curvatures[vertex_idx] = mean_curvature
            gaussian_curvatures[vertex_idx] = gaussian_curvature
        
        # 主曲率を計算（平均曲率とガウス曲率から）
        discriminant = mean_curvatures**2 - gaussian_curvatures
        discriminant = np.maximum(discriminant, 0)  # 負の値をクランプ
        
        k1 = mean_curvatures + np.sqrt(discriminant)  # 最大主曲率
        k2 = mean_curvatures - np.sqrt(discriminant)  # 最小主曲率
        
        # 頂点曲率として平均曲率の絶対値を使用
        vertex_curvatures = np.abs(mean_curvatures)
        
        # 正規化（必要に応じて）
        if self.normalize_attributes:
            if np.max(vertex_curvatures) > 0:
                vertex_curvatures /= np.max(vertex_curvatures)
            if np.max(np.abs(gaussian_curvatures)) > 0:
                gaussian_curvatures /= np.max(np.abs(gaussian_curvatures))
            if np.max(np.abs(mean_curvatures)) > 0:
                mean_curvatures /= np.max(np.abs(mean_curvatures))
        
        return vertex_curvatures, gaussian_curvatures, mean_curvatures
    
    def calculate_gradients(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """勾配を計算"""
        if self.gradient_method == "finite_diff":
            return self._calculate_gradients_finite_diff(mesh)
        else:
            return self._calculate_gradients_laplacian(mesh)
    
    def _smooth_vertex_normals(self, mesh: TriangleMesh, normals: np.ndarray) -> np.ndarray:
        """頂点法線を平滑化"""
        adjacency = self._build_vertex_adjacency(mesh)
        smoothed_normals = normals.copy()
        
        for vertex_idx in range(mesh.num_vertices):
            neighbors = adjacency[vertex_idx]
            
            if len(neighbors) == 0:
                continue
            
            # 近傍法線の平均
            neighbor_normals = normals[neighbors]
            avg_normal = np.mean(neighbor_normals, axis=0)
            
            # 現在の法線と平均の重み付き合成
            smoothed_normals[vertex_idx] = 0.7 * normals[vertex_idx] + 0.3 * avg_normal
        
        # 再正規化
        norms = np.linalg.norm(smoothed_normals, axis=1)
        valid_mask = norms > 1e-8
        smoothed_normals[valid_mask] /= norms[valid_mask, np.newaxis]
        
        return smoothed_normals
    
    def _build_vertex_adjacency(self, mesh: TriangleMesh) -> List[List[int]]:
        """頂点隣接リストを構築"""
        adjacency = [set() for _ in range(mesh.num_vertices)]
        
        for triangle in mesh.triangles:
            v0, v1, v2 = triangle
            adjacency[v0].update([v1, v2])
            adjacency[v1].update([v0, v2])
            adjacency[v2].update([v0, v1])
        
        return [list(neighbors) for neighbors in adjacency]
    
    def _calculate_mean_curvature(
        self, 
        mesh: TriangleMesh, 
        vertex_idx: int, 
        neighbors: List[int], 
        vertex_normals: np.ndarray
    ) -> float:
        """平均曲率を計算"""
        vertex_pos = mesh.vertices[vertex_idx]
        vertex_normal = vertex_normals[vertex_idx]
        
        laplacian = np.zeros(3)
        total_weight = 0.0
        
        for neighbor_idx in neighbors:
            neighbor_pos = mesh.vertices[neighbor_idx]
            edge_vector = neighbor_pos - vertex_pos
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 1e-8:
                weight = 1.0 / edge_length  # 距離の逆数重み
                laplacian += weight * edge_vector
                total_weight += weight
        
        if total_weight > 1e-8:
            laplacian /= total_weight
            # 法線方向成分を取得
            mean_curvature = np.dot(laplacian, vertex_normal)
            return mean_curvature
        
        return 0.0
    
    def _calculate_gaussian_curvature(
        self, 
        mesh: TriangleMesh, 
        vertex_idx: int, 
        neighbors: List[int]
    ) -> float:
        """ガウス曲率を計算（角度欠損法）"""
        vertex_pos = mesh.vertices[vertex_idx]
        
        # 隣接する三角形の角度を計算
        angle_sum = 0.0
        
        for i in range(len(neighbors)):
            v1_idx = neighbors[i]
            v2_idx = neighbors[(i + 1) % len(neighbors)]
            
            v1_pos = mesh.vertices[v1_idx]
            v2_pos = mesh.vertices[v2_idx]
            
            # 頂点からのベクトル
            vec1 = v1_pos - vertex_pos
            vec2 = v2_pos - vertex_pos
            
            # 正規化
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 1e-8 and norm2 > 1e-8:
                vec1 /= norm1
                vec2 /= norm2
                
                # 角度計算
                cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_sum += angle
        
        # 角度欠損からガウス曲率を計算
        if len(neighbors) > 0:
            gaussian_curvature = (2 * np.pi - angle_sum) / self._calculate_vertex_voronoi_area(mesh, vertex_idx)
            return gaussian_curvature
        
        return 0.0
    
    def _calculate_vertex_voronoi_area(self, mesh: TriangleMesh, vertex_idx: int) -> float:
        """頂点のVoronoi面積を計算"""
        # 簡略化として、隣接三角形面積の1/3の合計を使用
        total_area = 0.0
        
        for triangle in mesh.triangles:
            if vertex_idx in triangle:
                v0, v1, v2 = mesh.vertices[triangle]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                total_area += area / 3.0
        
        return max(total_area, 1e-8)  # 最小面積を保証
    
    def _calculate_gradients_finite_diff(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """有限差分法で勾配を計算"""
        gradients = np.zeros((mesh.num_vertices, 3))
        adjacency = self._build_vertex_adjacency(mesh)
        
        # 高度（Z座標）の勾配を計算
        for vertex_idx in range(mesh.num_vertices):
            neighbors = adjacency[vertex_idx]
            
            if len(neighbors) < 2:
                continue
            
            vertex_pos = mesh.vertices[vertex_idx]
            gradient = np.zeros(3)
            
            for neighbor_idx in neighbors:
                neighbor_pos = mesh.vertices[neighbor_idx]
                diff_pos = neighbor_pos - vertex_pos
                
                # XY平面での距離
                xy_distance = np.linalg.norm(diff_pos[:2])
                if xy_distance > 1e-8:
                    # Z方向の勾配
                    z_gradient = diff_pos[2] / xy_distance
                    
                    # XY方向の単位ベクトル
                    xy_direction = diff_pos[:2] / xy_distance
                    
                    # 勾配ベクトルに寄与
                    gradient[:2] += z_gradient * xy_direction
            
            if len(neighbors) > 0:
                gradients[vertex_idx] = gradient / len(neighbors)
        
        # 勾配の大きさ
        gradient_magnitudes = np.linalg.norm(gradients, axis=1)
        
        # 正規化（必要に応じて）
        if self.normalize_attributes and np.max(gradient_magnitudes) > 0:
            gradients /= np.max(gradient_magnitudes)
            gradient_magnitudes /= np.max(gradient_magnitudes)
        
        return gradients, gradient_magnitudes
    
    def _calculate_gradients_laplacian(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """ラプラシアン法で勾配を計算"""
        # より精密な勾配計算（実装省略、有限差分法を使用）
        return self._calculate_gradients_finite_diff(mesh)
    
    def _calculate_vertex_areas(self, mesh: TriangleMesh, triangle_areas: np.ndarray) -> np.ndarray:
        """頂点面積を計算"""
        vertex_areas = np.zeros(mesh.num_vertices)
        
        for i, triangle in enumerate(mesh.triangles):
            area_contribution = triangle_areas[i] / 3.0
            for vertex_idx in triangle:
                vertex_areas[vertex_idx] += area_contribution
        
        return vertex_areas
    
    def _update_stats(
        self, 
        total_time: float, 
        num_vertices: int, 
        num_triangles: int,
        normals_time: float, 
        curvature_time: float, 
        gradient_time: float
    ):
        """パフォーマンス統計更新"""
        self.stats['total_calculations'] += 1
        self.stats['total_time_ms'] += total_time
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_calculations']
        self.stats['last_num_vertices'] = num_vertices
        self.stats['last_num_triangles'] = num_triangles
        self.stats['normals_time_ms'] = normals_time
        self.stats['curvature_time_ms'] = curvature_time
        self.stats['gradient_time_ms'] = gradient_time
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_calculations': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_vertices': 0,
            'last_num_triangles': 0,
            'normals_time_ms': 0.0,
            'curvature_time_ms': 0.0,
            'gradient_time_ms': 0.0
        }


# 便利関数

def calculate_vertex_normals(mesh: TriangleMesh, smooth: bool = True) -> np.ndarray:
    """頂点法線を計算（簡単なインターフェース）"""
    calculator = AttributeCalculator(smooth_normals=smooth)
    return calculator.calculate_vertex_normals(mesh)


def calculate_face_normals(mesh: TriangleMesh) -> np.ndarray:
    """面法線を計算（簡単なインターフェース）"""
    calculator = AttributeCalculator()
    return calculator.calculate_triangle_normals(mesh)


def calculate_curvature(mesh: TriangleMesh) -> np.ndarray:
    """曲率を計算（簡単なインターフェース）"""
    calculator = AttributeCalculator()
    vertex_curvatures, _, _ = calculator.calculate_curvatures(mesh)
    return vertex_curvatures


def calculate_gradient(mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    """勾配を計算（簡単なインターフェース）"""
    calculator = AttributeCalculator()
    return calculator.calculate_gradients(mesh)


def compute_mesh_attributes(mesh: TriangleMesh) -> MeshAttributes:
    """全メッシュ属性を計算（簡単なインターフェース）"""
    calculator = AttributeCalculator()
    return calculator.compute_attributes(mesh) 