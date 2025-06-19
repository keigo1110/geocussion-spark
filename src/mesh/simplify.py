#!/usr/bin/env python3
"""
メッシュ簡略化

Open3Dを使用してメッシュの複雑さを削減し、リアルタイム処理に適した
シンプルなメッシュを生成する機能を提供します。
"""

import time
from enum import Enum
from typing import Tuple, Optional
import numpy as np
import open3d as o3d

from .delaunay import TriangleMesh


class SimplificationMethod(Enum):
    """簡略化手法の列挙"""
    QUADRIC_ERROR = "quadric"           # Quadric Error Metrics
    VERTEX_CLUSTERING = "clustering"    # 頂点クラスタリング  
    EDGE_COLLAPSE = "edge_collapse"     # エッジ崩壊
    UNIFORM_DECIMATION = "uniform"      # 均等間引き


class MeshSimplifier:
    """メッシュ簡略化クラス"""
    
    def __init__(
        self,
        method: SimplificationMethod = SimplificationMethod.QUADRIC_ERROR,
        target_reduction: float = 0.5,        # 目標削減率 (0.0-1.0)
        preserve_boundary: bool = True,        # 境界保持
        preserve_topology: bool = True,        # トポロジー保持
        aggressive: bool = False,              # 積極的簡略化
        max_iterations: int = 10,              # 最大反復回数
        quality_threshold: float = 0.1         # 品質閾値
    ):
        """
        初期化
        
        Args:
            method: 簡略化手法
            target_reduction: 目標削減率
            preserve_boundary: 境界を保持するか
            preserve_topology: トポロジーを保持するか  
            aggressive: 積極的簡略化を行うか
            max_iterations: 最大反復回数
            quality_threshold: 最低品質閾値
        """
        self.method = method
        self.target_reduction = target_reduction
        self.preserve_boundary = preserve_boundary
        self.preserve_topology = preserve_topology
        self.aggressive = aggressive
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        
        # パフォーマンス統計
        self.stats = {
            'total_simplifications': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_input_triangles': 0,
            'last_output_triangles': 0,
            'last_reduction_ratio': 0.0
        }
    
    def simplify_mesh(self, mesh: TriangleMesh) -> TriangleMesh:
        """
        メッシュを簡略化
        
        Args:
            mesh: 入力メッシュ
            
        Returns:
            簡略化されたメッシュ
        """
        start_time = time.perf_counter()
        
        if mesh.num_triangles == 0:
            return mesh
        
        # Open3Dメッシュに変換
        o3d_mesh = self._to_open3d_mesh(mesh)
        
        # 前処理
        o3d_mesh = self._preprocess_mesh(o3d_mesh)
        
        # 簡略化実行
        simplified_o3d = self._perform_simplification(o3d_mesh)
        
        # 後処理
        simplified_o3d = self._postprocess_mesh(simplified_o3d)
        
        # TriangleMeshに変換
        simplified_mesh = self._from_open3d_mesh(simplified_o3d)
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        reduction_ratio = 1.0 - (simplified_mesh.num_triangles / mesh.num_triangles)
        self._update_stats(elapsed_ms, mesh.num_triangles, simplified_mesh.num_triangles, reduction_ratio)
        
        return simplified_mesh
    
    def simplify_to_target_count(self, mesh: TriangleMesh, target_triangles: int) -> TriangleMesh:
        """
        指定した三角形数になるまで簡略化
        
        Args:
            mesh: 入力メッシュ
            target_triangles: 目標三角形数
            
        Returns:
            簡略化されたメッシュ
        """
        if target_triangles >= mesh.num_triangles:
            return mesh
        
        # 削減率を計算
        reduction_ratio = 1.0 - (target_triangles / mesh.num_triangles)
        
        # 一時的に削減率を変更
        original_reduction = self.target_reduction
        self.target_reduction = reduction_ratio
        
        try:
            simplified_mesh = self.simplify_mesh(mesh)
        finally:
            # 元の設定に戻す
            self.target_reduction = original_reduction
        
        return simplified_mesh
    
    def adaptive_simplify(self, mesh: TriangleMesh, max_triangles: int = 1000) -> TriangleMesh:
        """
        適応的簡略化（品質を保ちながら段階的に簡略化）
        
        Args:
            mesh: 入力メッシュ
            max_triangles: 最大三角形数
            
        Returns:
            簡略化されたメッシュ
        """
        if mesh.num_triangles <= max_triangles:
            return mesh
        
        current_mesh = mesh
        step_reduction = 0.2  # 一度に20%削減
        
        for iteration in range(self.max_iterations):
            if current_mesh.num_triangles <= max_triangles:
                break
            
            # 段階的に簡略化
            target_triangles = max(
                max_triangles,
                int(current_mesh.num_triangles * (1 - step_reduction))
            )
            
            simplified = self.simplify_to_target_count(current_mesh, target_triangles)
            
            # 品質チェック
            if self._check_mesh_quality(simplified):
                current_mesh = simplified
            else:
                # 品質が低下した場合は削減率を下げる
                step_reduction *= 0.5
                if step_reduction < 0.05:  # 最小削減率
                    break
        
        return current_mesh
    
    def _to_open3d_mesh(self, mesh: TriangleMesh) -> o3d.geometry.TriangleMesh:
        """TriangleMeshをOpen3Dメッシュに変換"""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
        
        # 色情報があれば追加
        if mesh.vertex_colors is not None:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors)
        
        return o3d_mesh
    
    def _from_open3d_mesh(self, o3d_mesh: o3d.geometry.TriangleMesh) -> TriangleMesh:
        """Open3DメッシュをTriangleMeshに変換"""
        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)
        
        # 色情報があれば取得
        vertex_colors = None
        if o3d_mesh.has_vertex_colors():
            vertex_colors = np.asarray(o3d_mesh.vertex_colors)
        
        # 法線情報があれば取得
        vertex_normals = None
        triangle_normals = None
        if o3d_mesh.has_vertex_normals():
            vertex_normals = np.asarray(o3d_mesh.vertex_normals)
        if o3d_mesh.has_triangle_normals():
            triangle_normals = np.asarray(o3d_mesh.triangle_normals)
        
        return TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            vertex_colors=vertex_colors,
            vertex_normals=vertex_normals,
            triangle_normals=triangle_normals
        )
    
    def _preprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """メッシュの前処理"""
        # 重複頂点の除去
        mesh.remove_duplicated_vertices()
        
        # 重複三角形の除去
        mesh.remove_duplicated_triangles()
        
        # 孤立した頂点の除去
        mesh.remove_unreferenced_vertices()
        
        # 退化した三角形の除去
        mesh.remove_degenerate_triangles()
        
        # 法線計算（必要に応じて）
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        return mesh
    
    def _perform_simplification(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """簡略化を実行"""
        target_triangles = int(len(mesh.triangles) * (1 - self.target_reduction))
        target_triangles = max(1, target_triangles)  # 最低1つの三角形は保持
        
        try:
            if self.method == SimplificationMethod.QUADRIC_ERROR:
                return mesh.simplify_quadric_decimation(target_triangles)
            
            elif self.method == SimplificationMethod.VERTEX_CLUSTERING:
                # 頂点クラスタリングによる簡略化
                voxel_size = self._calculate_voxel_size(mesh)
                return mesh.simplify_vertex_clustering(voxel_size)
            
            elif self.method == SimplificationMethod.EDGE_COLLAPSE:
                # より積極的なQuadric Error使用
                return mesh.simplify_quadric_decimation(
                    target_triangles,
                    maximum_error=0.1,
                    boundary_weight=1.0 if self.preserve_boundary else 0.1
                )
            
            elif self.method == SimplificationMethod.UNIFORM_DECIMATION:
                # 均等間引き（Open3Dでは直接サポートされていないため、
                # vertex clusteringで代用）
                bbox = mesh.get_axis_aligned_bounding_box()
                bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
                avg_size = np.mean(bbox_size)
                voxel_size = avg_size / (target_triangles ** 0.5)
                return mesh.simplify_vertex_clustering(voxel_size)
            
            else:
                # デフォルトはQuadric Error
                return mesh.simplify_quadric_decimation(target_triangles)
                
        except Exception as e:
            print(f"Simplification failed, returning original mesh: {e}")
            return mesh
    
    def _calculate_voxel_size(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """適切なボクセルサイズを計算"""
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
        max_dim = np.max(bbox_size)
        
        # 目標削減率に基づいてボクセルサイズを決定
        target_voxels = int(len(mesh.vertices) * (1 - self.target_reduction))
        voxel_size = max_dim / (target_voxels ** (1/3))
        
        return max(voxel_size, max_dim * 0.001)  # 最小サイズを保証
    
    def _postprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """メッシュの後処理"""
        # 孤立した部分の除去
        if len(mesh.triangles) > 0:
            # 最大連結成分のみを保持
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            if len(cluster_n_triangles) > 0:
                largest_cluster_idx = np.argmax(cluster_n_triangles)
                triangles_to_remove = [
                    i for i, cluster_id in enumerate(triangle_clusters) 
                    if cluster_id != largest_cluster_idx
                ]
                mesh.remove_triangles_by_index(triangles_to_remove)
                mesh.remove_unreferenced_vertices()
        
        # 法線の再計算
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        return mesh
    
    def _check_mesh_quality(self, mesh: TriangleMesh) -> bool:
        """メッシュ品質をチェック"""
        if mesh.num_triangles == 0:
            return False
        
        # 基本的な品質チェック
        try:
            areas = mesh.get_triangle_areas()
            
            # 極小三角形の割合をチェック
            min_area = np.median(areas) * 0.01  # 中央値の1%
            small_triangles = np.sum(areas < min_area)
            small_ratio = small_triangles / len(areas)
            
            return small_ratio < 0.1  # 10%以下なら品質OK
            
        except:
            return False
    
    def _update_stats(self, elapsed_ms: float, input_triangles: int, output_triangles: int, reduction_ratio: float):
        """パフォーマンス統計更新"""
        self.stats['total_simplifications'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_simplifications']
        self.stats['last_input_triangles'] = input_triangles
        self.stats['last_output_triangles'] = output_triangles
        self.stats['last_reduction_ratio'] = reduction_ratio
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_simplifications': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_input_triangles': 0,
            'last_output_triangles': 0,
            'last_reduction_ratio': 0.0
        }


# 便利関数

def simplify_mesh(
    mesh: TriangleMesh,
    target_reduction: float = 0.5,
    method: SimplificationMethod = SimplificationMethod.QUADRIC_ERROR
) -> TriangleMesh:
    """
    メッシュを簡略化（簡単なインターフェース）
    
    Args:
        mesh: 入力メッシュ
        target_reduction: 目標削減率
        method: 簡略化手法
        
    Returns:
        簡略化されたメッシュ
    """
    simplifier = MeshSimplifier(
        method=method,
        target_reduction=target_reduction
    )
    return simplifier.simplify_mesh(mesh)


def reduce_triangle_count(
    mesh: TriangleMesh,
    max_triangles: int,
    preserve_quality: bool = True
) -> TriangleMesh:
    """
    三角形数を指定数まで削減（簡単なインターフェース）
    
    Args:
        mesh: 入力メッシュ
        max_triangles: 最大三角形数
        preserve_quality: 品質を保持するか
        
    Returns:
        簡略化されたメッシュ
    """
    if mesh.num_triangles <= max_triangles:
        return mesh
    
    simplifier = MeshSimplifier(
        method=SimplificationMethod.QUADRIC_ERROR,
        preserve_boundary=preserve_quality,
        preserve_topology=preserve_quality
    )
    
    if preserve_quality:
        return simplifier.adaptive_simplify(mesh, max_triangles)
    else:
        return simplifier.simplify_to_target_count(mesh, max_triangles) 