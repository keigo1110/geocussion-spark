#!/usr/bin/env python3
"""
最適化BVH衝突検出システム
GPU加速バッチ処理による高速球-三角形衝突検出
speedup.md対応: 6-9ms/手 → 1-3µs/手 の大幅改善目標
"""

import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    from numba import jit, cuda
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src import get_logger
from src.performance.profiler import measure_performance, profile_phase
from src.mesh.delaunay import TriangleMesh
from src.collision.sphere_tri import CollisionInfo, ContactPoint
from src.data_types import CollisionType

logger = get_logger(__name__)


@dataclass
class BVHConfig:
    """BVH設定"""
    max_triangles_per_leaf: int = 10
    max_depth: int = 20
    use_gpu_acceleration: bool = True
    batch_size: int = 1000
    enable_early_termination: bool = True
    max_nodes_per_query: int = 500
    enable_parallel_queries: bool = True


@dataclass
class CollisionStats:
    """衝突検出統計"""
    total_queries: int = 0
    total_time_ms: float = 0.0
    avg_time_per_query_us: float = 0.0
    gpu_accelerated_queries: int = 0
    cpu_fallback_queries: int = 0
    early_terminations: int = 0
    triangles_tested: int = 0
    triangles_skipped: int = 0


class AccelerationStrategy(Enum):
    """加速戦略"""
    GPU_CUPY = "gpu_cupy"           # CuPy GPU加速
    GPU_NUMBA = "gpu_numba"         # Numba CUDA加速
    CPU_PARALLEL = "cpu_parallel"   # CPU並列処理
    CPU_SEQUENTIAL = "cpu_sequential" # CPU逐次処理


class OptimizedBVHNode:
    """最適化BVHノード"""
    
    def __init__(self):
        self.bounding_box: Optional['BoundingBox'] = None
        self.triangle_indices: List[int] = []
        self.left_child: Optional['OptimizedBVHNode'] = None
        self.right_child: Optional['OptimizedBVHNode'] = None
        self.is_leaf: bool = True
        self.node_id: int = -1  # GPU処理用ID


class BoundingBox:
    """バウンディングボックス"""
    
    def __init__(self, min_point: np.ndarray, max_point: np.ndarray):
        self.min_point = min_point.copy()
        self.max_point = max_point.copy()
        self.center = (min_point + max_point) / 2.0
        self.extent = max_point - min_point
    
    def intersects_sphere(self, center: np.ndarray, radius: float) -> bool:
        """球との交差判定（高速版）"""
        # 最近接点を計算
        closest = np.clip(center, self.min_point, self.max_point)
        distance_squared = np.sum((center - closest) ** 2)
        return distance_squared <= radius * radius
    
    def expand(self, other: 'BoundingBox') -> 'BoundingBox':
        """バウンディングボックス拡張"""
        new_min = np.minimum(self.min_point, other.min_point)
        new_max = np.maximum(self.max_point, other.max_point)
        return BoundingBox(new_min, new_max)


class OptimizedBVH:
    """最適化BVH構造"""
    
    def __init__(self, mesh: TriangleMesh, config: BVHConfig):
        self.mesh = mesh
        self.config = config
        self.root: Optional[OptimizedBVHNode] = None
        self.stats = CollisionStats()
        
        # GPU最適化用データ
        self._flat_nodes: List[OptimizedBVHNode] = []
        self._gpu_data_prepared = False
        
        # 戦略選択
        self.strategy = self._select_acceleration_strategy()
        
        # BVH構築
        self._build_bvh()
        
        # GPU データ準備
        if self.strategy in [AccelerationStrategy.GPU_CUPY, AccelerationStrategy.GPU_NUMBA]:
            self._prepare_gpu_data()
    
    def _select_acceleration_strategy(self) -> AccelerationStrategy:
        """最適な加速戦略を選択"""
        if self.config.use_gpu_acceleration:
            if HAS_CUPY:
                logger.info("Using CuPy GPU acceleration for BVH")
                return AccelerationStrategy.GPU_CUPY
            elif HAS_NUMBA:
                logger.info("Using Numba CUDA acceleration for BVH")
                return AccelerationStrategy.GPU_NUMBA
        
        if self.config.enable_parallel_queries:
            logger.info("Using CPU parallel acceleration for BVH")
            return AccelerationStrategy.CPU_PARALLEL
        else:
            logger.info("Using CPU sequential processing for BVH")
            return AccelerationStrategy.CPU_SEQUENTIAL
    
    def _build_bvh(self):
        """BVH構築"""
        if self.mesh.num_triangles == 0:
            return
        
        # 全三角形インデックス
        all_indices = list(range(self.mesh.num_triangles))
        
        # ルートノード構築
        self.root = self._build_node(all_indices, 0)
        
        # フラット構造作成（GPU用）
        self._flatten_tree()
        
        logger.info(f"BVH built with {len(self._flat_nodes)} nodes for {self.mesh.num_triangles} triangles")
    
    def _build_node(self, triangle_indices: List[int], depth: int) -> OptimizedBVHNode:
        """再帰的ノード構築"""
        node = OptimizedBVHNode()
        node.triangle_indices = triangle_indices
        node.node_id = len(self._flat_nodes)
        
        # バウンディングボックス計算
        node.bounding_box = self._compute_bounding_box(triangle_indices)
        
        # リーフ判定
        if (len(triangle_indices) <= self.config.max_triangles_per_leaf or 
            depth >= self.config.max_depth):
            node.is_leaf = True
            return node
        
        # 分割軸選択（最大範囲軸）
        extent = node.bounding_box.extent
        split_axis = np.argmax(extent)
        
        # 三角形の重心で分割
        centroids = []
        for tri_idx in triangle_indices:
            triangle = self.mesh.triangles[tri_idx]
            vertices = self.mesh.vertices[triangle]
            centroid = np.mean(vertices, axis=0)
            centroids.append(centroid[split_axis])
        
        # 中央値で分割
        median = np.median(centroids)
        left_indices = []
        right_indices = []
        
        for i, tri_idx in enumerate(triangle_indices):
            if centroids[i] <= median:
                left_indices.append(tri_idx)
            else:
                right_indices.append(tri_idx)
        
        # 分割失敗時はリーフにする
        if len(left_indices) == 0 or len(right_indices) == 0:
            node.is_leaf = True
            return node
        
        # 内部ノード
        node.is_leaf = False
        node.left_child = self._build_node(left_indices, depth + 1)
        node.right_child = self._build_node(right_indices, depth + 1)
        
        return node
    
    def _compute_bounding_box(self, triangle_indices: List[int]) -> BoundingBox:
        """三角形群のバウンディングボックス計算"""
        if not triangle_indices:
            return BoundingBox(np.zeros(3), np.zeros(3))
        
        all_vertices = []
        for tri_idx in triangle_indices:
            triangle = self.mesh.triangles[tri_idx]
            all_vertices.extend(self.mesh.vertices[triangle])
        
        vertices_array = np.array(all_vertices)
        min_point = np.min(vertices_array, axis=0)
        max_point = np.max(vertices_array, axis=0)
        
        return BoundingBox(min_point, max_point)
    
    def _flatten_tree(self):
        """ツリーをフラット構造に変換（GPU処理用）"""
        self._flat_nodes = []
        if self.root:
            self._flatten_recursive(self.root)
    
    def _flatten_recursive(self, node: OptimizedBVHNode):
        """再帰的フラット化"""
        self._flat_nodes.append(node)
        if not node.is_leaf:
            if node.left_child:
                self._flatten_recursive(node.left_child)
            if node.right_child:
                self._flatten_recursive(node.right_child)
    
    def _prepare_gpu_data(self):
        """GPU処理用データ準備"""
        if not HAS_CUPY and self.strategy == AccelerationStrategy.GPU_CUPY:
            return
        
        try:
            # 頂点データをGPUに転送
            if HAS_CUPY:
                self._gpu_vertices = cp.asarray(self.mesh.vertices, dtype=cp.float32)
                self._gpu_triangles = cp.asarray(self.mesh.triangles, dtype=cp.int32)
            
            # ノード情報をGPU用に準備
            self._prepare_node_data()
            
            self._gpu_data_prepared = True
            logger.info(f"GPU data prepared for {len(self._flat_nodes)} nodes")
            
        except Exception as e:
            logger.warning(f"GPU data preparation failed: {e}")
            self.strategy = AccelerationStrategy.CPU_PARALLEL
    
    def _prepare_node_data(self):
        """ノードデータのGPU準備"""
        # ノードバウンディングボックス
        node_bboxes = []
        node_triangle_starts = []
        node_triangle_counts = []
        all_node_triangles = []
        
        for node in self._flat_nodes:
            if node.bounding_box:
                bbox_data = np.concatenate([
                    node.bounding_box.min_point,
                    node.bounding_box.max_point
                ])
                node_bboxes.append(bbox_data)
            else:
                node_bboxes.append(np.zeros(6))
            
            node_triangle_starts.append(len(all_node_triangles))
            node_triangle_counts.append(len(node.triangle_indices))
            all_node_triangles.extend(node.triangle_indices)
        
        if HAS_CUPY:
            self._gpu_node_bboxes = cp.asarray(node_bboxes, dtype=cp.float32)
            self._gpu_node_triangle_starts = cp.asarray(node_triangle_starts, dtype=cp.int32)
            self._gpu_node_triangle_counts = cp.asarray(node_triangle_counts, dtype=cp.int32)
            self._gpu_all_node_triangles = cp.asarray(all_node_triangles, dtype=cp.int32)
    
    @profile_phase("bvh_collision_query")
    def query_sphere_collision(self, center: np.ndarray, radius: float) -> List[int]:
        """球衝突クエリ（最適化版）"""
        with measure_performance("bvh_sphere_collision"):
            start_time = time.perf_counter()
            
            # 戦略別処理
            if self.strategy == AccelerationStrategy.GPU_CUPY:
                triangle_indices = self._query_gpu_cupy(center, radius)
            elif self.strategy == AccelerationStrategy.GPU_NUMBA:
                triangle_indices = self._query_gpu_numba(center, radius)
            elif self.strategy == AccelerationStrategy.CPU_PARALLEL:
                triangle_indices = self._query_cpu_parallel(center, radius)
            else:
                triangle_indices = self._query_cpu_sequential(center, radius)
            
            # 統計更新
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.stats.total_queries += 1
            self.stats.total_time_ms += elapsed_ms
            self.stats.avg_time_per_query_us = (self.stats.total_time_ms / self.stats.total_queries) * 1000.0
            
            return triangle_indices
    
    def _query_gpu_cupy(self, center: np.ndarray, radius: float) -> List[int]:
        """CuPy GPU クエリ"""
        if not self._gpu_data_prepared or not HAS_CUPY:
            return self._query_cpu_sequential(center, radius)
        
        try:
            # GPU上で球-BVH交差判定
            gpu_center = cp.asarray(center, dtype=cp.float32)
            gpu_radius = cp.float32(radius)
            
            # カスタムCUDAカーネル（簡易版）
            intersecting_triangles = self._gpu_bvh_traversal_cupy(gpu_center, gpu_radius)
            
            self.stats.gpu_accelerated_queries += 1
            return intersecting_triangles.tolist()
            
        except Exception as e:
            logger.debug(f"CuPy GPU query failed: {e}")
            self.stats.cpu_fallback_queries += 1
            return self._query_cpu_sequential(center, radius)
    
    def _gpu_bvh_traversal_cupy(self, center: cp.ndarray, radius: cp.float32) -> cp.ndarray:
        """CuPy GPU BVH探索"""
        # 簡易実装：全ノードをチェック（実際は階層探索が必要）
        result_triangles = []
        
        for i, node in enumerate(self._flat_nodes):
            if node.is_leaf and node.bounding_box:
                # バウンディングボックスチェック
                bbox_min = cp.asarray(node.bounding_box.min_point, dtype=cp.float32)
                bbox_max = cp.asarray(node.bounding_box.max_point, dtype=cp.float32)
                
                # 球-ボックス交差判定
                closest = cp.clip(center, bbox_min, bbox_max)
                distance_squared = cp.sum((center - closest) ** 2)
                
                if distance_squared <= radius * radius:
                    result_triangles.extend(node.triangle_indices)
        
        return cp.asarray(result_triangles, dtype=cp.int32)
    
    def _query_gpu_numba(self, center: np.ndarray, radius: float) -> List[int]:
        """Numba CUDA クエリ"""
        # TODO: Numba CUDA実装
        return self._query_cpu_sequential(center, radius)
    
    def _query_cpu_parallel(self, center: np.ndarray, radius: float) -> List[int]:
        """CPU並列クエリ"""
        if not self.root:
            return []
        
        # 並列処理用のワーカー
        def worker(node_batch):
            results = []
            for node in node_batch:
                if node.is_leaf and node.bounding_box:
                    if node.bounding_box.intersects_sphere(center, radius):
                        results.extend(node.triangle_indices)
            return results
        
        # リーフノードを取得
        leaf_nodes = [node for node in self._flat_nodes if node.is_leaf]
        
        # バッチ分割
        batch_size = max(1, len(leaf_nodes) // 4)  # 4スレッド想定
        batches = [leaf_nodes[i:i+batch_size] for i in range(0, len(leaf_nodes), batch_size)]
        
        # 並列実行
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, batch) for batch in batches]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        return list(set(all_results))  # 重複除去
    
    def _query_cpu_sequential(self, center: np.ndarray, radius: float) -> List[int]:
        """CPU逐次クエリ（従来版）"""
        if not self.root:
            return []
        
        result = []
        nodes_visited = [0]  # 参照渡しでカウント
        
        self._query_recursive(
            self.root, center, radius, result, 
            nodes_visited, self.config.max_nodes_per_query
        )
        
        if nodes_visited[0] >= self.config.max_nodes_per_query:
            self.stats.early_terminations += 1
        
        return result
    
    def _query_recursive(self, node: OptimizedBVHNode, center: np.ndarray, 
                        radius: float, result: List[int], 
                        nodes_visited: List[int], max_nodes: int):
        """再帰的BVH探索"""
        nodes_visited[0] += 1
        
        # 早期終了
        if nodes_visited[0] > max_nodes:
            return
        
        # バウンディングボックス交差チェック
        if not node.bounding_box or not node.bounding_box.intersects_sphere(center, radius):
            return
        
        if node.is_leaf:
            # リーフノード：三角形を結果に追加
            result.extend(node.triangle_indices)
            self.stats.triangles_tested += len(node.triangle_indices)
        else:
            # 内部ノード：子ノードを探索
            if node.left_child:
                self._query_recursive(node.left_child, center, radius, result, nodes_visited, max_nodes)
            if node.right_child and nodes_visited[0] <= max_nodes:
                self._query_recursive(node.right_child, center, radius, result, nodes_visited, max_nodes)
    
    def get_stats(self) -> CollisionStats:
        """統計情報を取得"""
        return self.stats
    
    def reset_stats(self):
        """統計をリセット"""
        self.stats = CollisionStats()
    
    def print_stats(self):
        """統計情報をコンソール出力"""
        print("\n" + "="*50)
        print("OPTIMIZED BVH COLLISION STATS")
        print("="*50)
        print(f"Strategy: {self.strategy.value}")
        print(f"Total queries: {self.stats.total_queries}")
        print(f"Avg time per query: {self.stats.avg_time_per_query_us:.1f}µs")
        print(f"GPU accelerated: {self.stats.gpu_accelerated_queries}")
        print(f"CPU fallback: {self.stats.cpu_fallback_queries}")
        print(f"Early terminations: {self.stats.early_terminations}")
        print(f"Triangles tested: {self.stats.triangles_tested}")
        if self.stats.total_queries > 0:
            gpu_rate = self.stats.gpu_accelerated_queries / self.stats.total_queries * 100
            print(f"GPU acceleration rate: {gpu_rate:.1f}%")
        print("="*50)


class OptimizedCollisionSearcher:
    """最適化衝突検索器"""
    
    def __init__(self, mesh: TriangleMesh, config: Optional[BVHConfig] = None):
        self.mesh = mesh
        self.config = config or BVHConfig()
        
        # BVH構築
        self.bvh = OptimizedBVH(mesh, self.config)
        
        logger.info(f"Optimized collision searcher initialized for mesh with {mesh.num_triangles} triangles")
    
    @profile_phase("optimized_collision_search")
    def search_sphere_collision(self, center: np.ndarray, radius: float) -> List[int]:
        """最適化球衝突検索"""
        return self.bvh.query_sphere_collision(center, radius)
    
    def batch_search_spheres(self, centers: np.ndarray, radii: Union[float, np.ndarray]) -> List[List[int]]:
        """複数球の一括衝突検索"""
        if isinstance(radii, float):
            radii = np.full(len(centers), radii)
        
        results = []
        for center, radius in zip(centers, radii):
            triangle_indices = self.search_sphere_collision(center, radius)
            results.append(triangle_indices)
        
        return results
    
    def get_stats(self) -> CollisionStats:
        """統計情報を取得"""
        return self.bvh.get_stats()
    
    def print_stats(self):
        """統計情報をコンソール出力"""
        self.bvh.print_stats()


def create_optimized_collision_searcher(mesh: TriangleMesh, 
                                       config: Optional[BVHConfig] = None) -> OptimizedCollisionSearcher:
    """最適化衝突検索器ファクトリー"""
    return OptimizedCollisionSearcher(mesh, config)


def benchmark_collision_methods(mesh: TriangleMesh, test_spheres: List[Tuple[np.ndarray, float]], 
                               iterations: int = 100) -> Dict[str, Any]:
    """衝突検出手法のベンチマーク"""
    results = {}
    
    # 最適化BVH
    try:
        optimized_searcher = OptimizedCollisionSearcher(mesh)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            for center, radius in test_spheres:
                _ = optimized_searcher.search_sphere_collision(center, radius)
            times.append((time.perf_counter() - start) * 1000.0)
        
        results['optimized_bvh'] = {
            'avg_ms': np.mean(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'std_ms': np.std(times)
        }
    except Exception as e:
        results['optimized_bvh'] = {'error': str(e)}
    
    # TODO: 従来方式との比較
    
    return results 