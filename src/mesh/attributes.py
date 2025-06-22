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
        """頂点法線を計算（ベクトル化対応）"""
        if mesh.vertex_normals is not None and not self.smooth_normals:
            return mesh.vertex_normals

        # 三角形法線と面積をベクトル化して計算
        triangle_normals = self.calculate_triangle_normals(mesh)
        triangle_areas = mesh.get_triangle_areas()

        vertex_normals = np.zeros((mesh.num_vertices, 3))
        weighted_normals = triangle_normals * triangle_areas[:, np.newaxis]

        # 各頂点に、関連する三角形の重み付き法線を加算
        # np.add.atはインデックスが重複する場合に値を累積する
        np.add.at(vertex_normals, mesh.triangles[:, 0], weighted_normals)
        np.add.at(vertex_normals, mesh.triangles[:, 1], weighted_normals)
        np.add.at(vertex_normals, mesh.triangles[:, 2], weighted_normals)

        # 法線を正規化
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        valid_mask = (norms > 1e-8).flatten()
        
        normalized_normals = np.zeros_like(vertex_normals)
        normalized_normals[valid_mask] = vertex_normals[valid_mask] / norms[valid_mask]

        # 無効な法線はZ軸方向に設定
        normalized_normals[~valid_mask] = [0, 0, 1]
        
        # 平滑化
        if self.smooth_normals:
            normalized_normals = self._smooth_vertex_normals(mesh, normalized_normals)
        
        return normalized_normals
    
    def calculate_triangle_normals(self, mesh: TriangleMesh) -> np.ndarray:
        """三角形法線を計算（ベクトル化対応）"""
        if mesh.triangle_normals is not None:
            return mesh.triangle_normals
        
        # 三角形の各頂点の座標を取得
        v0 = mesh.vertices[mesh.triangles[:, 0]]
        v1 = mesh.vertices[mesh.triangles[:, 1]]
        v2 = mesh.vertices[mesh.triangles[:, 2]]
        
        # エッジベクトルを計算
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 外積で法線を一括計算
        normals = np.cross(edge1, edge2)
        
        # 法線を正規化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid_mask = (norms > 1e-8).flatten()
        
        normalized_normals = np.zeros_like(normals)
        normalized_normals[valid_mask] = normals[valid_mask] / norms[valid_mask]
        
        # 縮退三角形など、ノルムがゼロの場合はデフォルト法線（Z軸）を設定
        normalized_normals[~valid_mask] = [0, 0, 1]
        
        return normalized_normals
    
    def calculate_curvatures(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """曲率を計算（ベクトル化対応）"""
        num_vertices = mesh.num_vertices
        vertices = mesh.vertices
        triangles = mesh.triangles

        # ガウス曲率の計算 (ベクトル化)
        # 三角形の3つの辺の長さを計算
        v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        a = np.linalg.norm(v1 - v2, axis=1)
        b = np.linalg.norm(v0 - v2, axis=1)
        c = np.linalg.norm(v0 - v1, axis=1)

        # 各頂点の角度を余弦定理で計算
        # ゼロ除算を回避
        eps = 1e-8
        cos_alpha0 = (b**2 + c**2 - a**2) / (2 * b * c + eps)
        cos_alpha1 = (a**2 + c**2 - b**2) / (2 * a * c + eps)
        cos_alpha2 = (a**2 + b**2 - c**2) / (2 * a * b + eps)
        
        alpha0 = np.arccos(np.clip(cos_alpha0, -1, 1))
        alpha1 = np.arccos(np.clip(cos_alpha1, -1, 1))
        alpha2 = np.arccos(np.clip(cos_alpha2, -1, 1))

        # 各頂点に角度を集約
        gaussian_curvatures = np.full(num_vertices, 2 * np.pi)
        np.add.at(gaussian_curvatures, triangles[:, 0], -alpha0)
        np.add.at(gaussian_curvatures, triangles[:, 1], -alpha1)
        np.add.at(gaussian_curvatures, triangles[:, 2], -alpha2)

        # 平均曲率の計算 (ベクトル化)
        # Cotangent Laplacian Operator (離散ラプラシアン)
        # See: https://cse.msu.edu/~yiying/cse848/papers/discrete_laplace.pdf
        cot_alpha0 = 1.0 / (np.tan(alpha0) + eps)
        cot_alpha1 = 1.0 / (np.tan(alpha1) + eps)
        cot_alpha2 = 1.0 / (np.tan(alpha2) + eps)
        
        # 疎行列の要素を構築
        rows = np.concatenate([triangles[:,0], triangles[:,1], triangles[:,1], triangles[:,2], triangles[:,2], triangles[:,0]])
        cols = np.concatenate([triangles[:,1], triangles[:,0], triangles[:,2], triangles[:,1], triangles[:,0], triangles[:,2]])
        weights = 0.5 * np.concatenate([cot_alpha2, cot_alpha2, cot_alpha0, cot_alpha0, cot_alpha1, cot_alpha1])

        # 対角成分
        diag_rows = np.arange(num_vertices)
        diag_weights = np.zeros(num_vertices)
        np.add.at(diag_weights, rows, -weights)

        # 疎行列 (Cotangent matrix) を構築
        L = sparse.coo_matrix((np.concatenate([weights, diag_weights]), (np.concatenate([rows, diag_rows]), np.concatenate([cols, diag_rows]))),
                              shape=(num_vertices, num_vertices)).tocsr()
        
        # ラプラシアンベクトル
        laplacian_vectors = L.dot(vertices)
        
        # 平均曲率法線ベクトルから平均曲率を計算
        mean_curvatures_normal = np.linalg.norm(laplacian_vectors, axis=1)
        
        # 頂点法線
        vertex_normals = self.calculate_vertex_normals(mesh)
        
        # ラプラシアンが法線と同じ方向を向いているかチェック
        orientation_sign = np.sign(np.sum(laplacian_vectors * vertex_normals, axis=1))
        mean_curvatures = mean_curvatures_normal * orientation_sign

        # 頂点面積で正規化
        triangle_areas = mesh.get_triangle_areas()
        vertex_areas = self._calculate_vertex_areas(mesh, triangle_areas)
        mean_curvatures /= (2 * vertex_areas + eps)
        gaussian_curvatures /= (vertex_areas + eps)
        
        # 主曲率を計算
        discriminant = np.maximum(mean_curvatures**2 - gaussian_curvatures, 0)
        k1 = mean_curvatures + np.sqrt(discriminant)
        k2 = mean_curvatures - np.sqrt(discriminant)
        
        vertex_curvatures = np.abs(mean_curvatures) # 主曲率の平均として定義

        if self.normalize_attributes:
            max_abs_curv = np.max(np.abs(vertex_curvatures))
            if max_abs_curv > eps:
                vertex_curvatures /= max_abs_curv
        
        return vertex_curvatures, gaussian_curvatures, mean_curvatures
    
    def calculate_gradients(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """勾配を計算（ベクトル化対応）"""
        num_vertices = mesh.num_vertices
        vertices = mesh.vertices
        
        # 隣接行列を構築
        adjacency = self._build_vertex_adjacency_sparse(mesh)
        num_neighbors = np.array(adjacency.sum(axis=1)).flatten()

        # 隣接頂点との座標の差の合計を計算
        sum_diffs = adjacency.dot(vertices) - (sparse.diags(num_neighbors) @ vertices)
        
        # 平均差分を計算
        # ゼロ除算を回避
        valid_mask = num_neighbors > 0
        gradients = np.zeros_like(vertices)
        gradients[valid_mask] = sum_diffs[valid_mask] / num_neighbors[valid_mask, np.newaxis]
        
        # Z成分（高さ）の勾配を主に使うが、XY成分も保持
        gradient_magnitudes = np.linalg.norm(gradients, axis=1)

        if self.normalize_attributes:
            max_grad = np.max(gradient_magnitudes)
            if max_grad > 1e-8:
                gradient_magnitudes /= max_grad
                
        return gradients, gradient_magnitudes
    
    def _smooth_vertex_normals(self, mesh: TriangleMesh, normals: np.ndarray) -> np.ndarray:
        """頂点法線を平滑化（ベクトル化）"""
        adjacency = self._build_vertex_adjacency_sparse(mesh)
        
        # 隣接頂点の法線の平均を取る
        num_neighbors = np.array(adjacency.sum(axis=1)).flatten()
        sum_neighbor_normals = adjacency.dot(normals)
        
        # ゼロ除算を回避
        valid_mask = num_neighbors > 0
        avg_neighbor_normals = np.zeros_like(normals)
        avg_neighbor_normals[valid_mask] = sum_neighbor_normals[valid_mask] / num_neighbors[valid_mask, np.newaxis]
        
        # 現在の法線と平均の重み付き合成
        smoothed_normals = 0.7 * normals + 0.3 * avg_neighbor_normals
        
        # 再正規化
        norms = np.linalg.norm(smoothed_normals, axis=1, keepdims=True)
        valid_mask_norm = (norms > 1e-8).flatten()
        smoothed_normals[valid_mask_norm] /= norms[valid_mask_norm]
        
        return smoothed_normals
    
    def _build_vertex_adjacency_sparse(self, mesh: TriangleMesh) -> sparse.csr_matrix:
        """頂点の隣接関係を疎行列で構築"""
        num_vertices = mesh.num_vertices
        rows = mesh.triangles[:, [0, 0, 1, 1, 2, 2]].flatten()
        cols = mesh.triangles[:, [1, 2, 0, 2, 0, 1]].flatten()
        data = np.ones(len(rows), dtype=bool)
        
        adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices))
        return adjacency.tocsr()
    
    def _build_vertex_adjacency(self, mesh: TriangleMesh) -> List[List[int]]:
        """頂点の隣接関係をリストで構築（旧実装）"""
        adjacency = [[] for _ in range(mesh.num_vertices)]
        for triangle in mesh.triangles:
            adjacency[triangle[0]].extend([triangle[1], triangle[2]])
            adjacency[triangle[1]].extend([triangle[0], triangle[2]])
            adjacency[triangle[2]].extend([triangle[0], triangle[1]])
        
        # 重複を除去
        for i in range(mesh.num_vertices):
            adjacency[i] = list(set(adjacency[i]))
            
        return adjacency
    
    def _calculate_mean_curvature(
        self, 
        mesh: TriangleMesh, 
        vertex_idx: int, 
        neighbors: List[int]
    ) -> float:
        """ガウス曲率を計算（旧実装）"""
        # (このメソッドはベクトル化された `calculate_curvatures` に置き換えられた)
        angles_sum = 0.0
        p0 = mesh.vertices[vertex_idx]
        
        for i in range(len(neighbors)):
            p1 = mesh.vertices[neighbors[i]]
            p2 = mesh.vertices[neighbors[(i + 1) % len(neighbors)]] # リング上の次の隣接点
            
            v1 = p1 - p0
            v2 = p2 - p0
            
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product > 1e-8:
                # 角度を計算
                angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
                angles_sum += angle
        
        return 2 * np.pi - angles_sum
    
    def _calculate_vertex_voronoi_area(self, mesh: TriangleMesh, vertex_idx: int) -> float:
        """ボロノイ領域の面積を計算（旧実装）"""
        # (このメソッドはベクトル化された `_calculate_vertex_areas` に置き換えられた)
        area = 0.0
        p_i = mesh.vertices[vertex_idx]
        
        # 隣接する三角形を見つける
        incident_triangles = np.where(np.any(mesh.triangles == vertex_idx, axis=1))[0]
        
        for tri_idx in incident_triangles:
            v_indices = mesh.triangles[tri_idx]
            
            # ボロノイ領域の計算
            # 鈍角三角形の場合は特別処理
            
            p_j_idx, p_k_idx = [v for v in v_indices if v != vertex_idx]
            p_j, p_k = mesh.vertices[p_j_idx], mesh.vertices[p_k_idx]
            
            edge_ij = p_j - p_i
            edge_ik = p_k - p_i
            edge_jk = p_k - p_j
            
            if np.dot(edge_ij, -edge_jk) < 0 or np.dot(edge_ik, edge_jk) < 0:
                # 鈍角
                area += 0.5 * np.linalg.norm(np.cross(edge_ij, edge_ik)) / 2.0
            else:
                # 鋭角
                cot_j = 1.0 / np.tan(np.arccos(np.dot(-edge_ij, edge_jk) / (np.linalg.norm(edge_ij) * np.linalg.norm(edge_jk))))
                cot_k = 1.0 / np.tan(np.arccos(np.dot(-edge_ik, -edge_jk) / (np.linalg.norm(edge_ik) * np.linalg.norm(edge_jk))))
                area += (np.linalg.norm(edge_ij)**2 * cot_j + np.linalg.norm(edge_ik)**2 * cot_k) / 8.0
                
        return area
    
    def _calculate_gradients_finite_diff(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """有限差分法で勾配を計算（旧実装）"""
        # (このメソッドはベクトル化された `calculate_gradients` に置き換えられた)
        gradients = np.zeros((mesh.num_vertices, 3))
        adjacency = self._build_vertex_adjacency(mesh)
        
        for i in range(mesh.num_vertices):
            p_i = mesh.vertices[i]
            neighbors_i = adjacency[i]
            
            if not neighbors_i:
                continue
                
            neighbor_coords = mesh.vertices[neighbors_i]
            
            # 中心差分法
            grad = np.mean(neighbor_coords - p_i, axis=0)
            gradients[i] = grad
            
        gradient_magnitudes = np.linalg.norm(gradients, axis=1)
        
        if self.normalize_attributes:
            max_grad = np.max(gradient_magnitudes)
            if max_grad > 1e-8:
                gradient_magnitudes /= max_grad
                
        return gradients, gradient_magnitudes

    def _calculate_gradients_laplacian(self, mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """ラプラシアン法で勾配を計算（旧実装）"""
        # (このメソッドはベクトル化された `calculate_curvatures` に置き換えられた)
        mean_curvatures = self.calculate_curvatures(mesh)[2]
        gradients = np.zeros((mesh.num_vertices, 3)) # Not straightforward to get vector gradient
        gradient_magnitudes = np.abs(mean_curvatures)
        return gradients, gradient_magnitudes

    def _calculate_vertex_areas(self, mesh: TriangleMesh, triangle_areas: np.ndarray) -> np.ndarray:
        """各頂点に寄与する面積を計算（ベクトル化）"""
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

    def _calculate_vertex_normals_legacy(self, mesh: TriangleMesh) -> np.ndarray:
        """頂点法線を計算（旧実装）"""
        vertex_normals = np.zeros((mesh.num_vertices, 3))
        triangle_normals = self._calculate_triangle_normals_legacy(mesh)
        triangle_areas = mesh.get_triangle_areas()
        for i, triangle in enumerate(mesh.triangles):
            normal = triangle_normals[i]
            area = triangle_areas[i]
            for vertex_idx in triangle:
                vertex_normals[vertex_idx] += normal * area
        norms = np.linalg.norm(vertex_normals, axis=1)
        valid_mask = norms > 1e-8
        vertex_normals[valid_mask] /= norms[valid_mask, np.newaxis]
        vertex_normals[~valid_mask] = [0, 0, 1]
        if self.smooth_normals:
             vertex_normals = self._smooth_vertex_normals_legacy(mesh, vertex_normals)
        return vertex_normals

    def _calculate_triangle_normals_legacy(self, mesh: TriangleMesh) -> np.ndarray:
        """三角形法線を計算（旧実装）"""
        triangle_normals = np.zeros((mesh.num_triangles, 3))
        for i, triangle in enumerate(mesh.triangles):
            v0, v1, v2 = mesh.vertices[triangle]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                triangle_normals[i] = normal / norm
            else:
                triangle_normals[i] = [0, 0, 1]
        return triangle_normals

    def _smooth_vertex_normals_legacy(self, mesh: TriangleMesh, normals: np.ndarray) -> np.ndarray:
        """頂点法線を平滑化（旧実装）"""
        adjacency = self._build_vertex_adjacency(mesh)
        smoothed_normals = normals.copy()
        for vertex_idx in range(mesh.num_vertices):
            neighbors = adjacency[vertex_idx]
            if not neighbors:
                continue
            neighbor_normals = normals[neighbors]
            avg_normal = np.mean(neighbor_normals, axis=0)
            smoothed_normals[vertex_idx] = 0.7 * normals[vertex_idx] + 0.3 * avg_normal
        norms = np.linalg.norm(smoothed_normals, axis=1, keepdims=True)
        valid_mask = (norms > 1e-8).flatten()
        smoothed_normals[valid_mask] /= norms[valid_mask]
        return smoothed_normals


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