#!/usr/bin/env python3
"""
Collision Detection Memory Optimization Module

numpy配列コピーを最小化し、メモリ効率とパフォーマンスを向上させるための最適化機能。
配列プール、インプレース操作、参照渡し最適化を提供します。
"""

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import deque

from ..config import get_config


@dataclass
class ArrayStats:
    """配列使用統計"""
    total_allocations: int = 0
    total_deallocations: int = 0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    reuse_rate: float = 0.0


class ArrayPool:
    """numpy配列プール（メモリ再利用）"""
    
    # NOTE (T-MEM-001): The pool size was reduced dramatically (100 ➜ 4)
    # to avoid retaining large numbers of inactive ndarrays for long-running
    # sessions which led to unbounded RSS growth in production. 4 buffers per
    # (shape, dtype) key has proven sufficient for double-buffering and a
    # small safety margin while keeping the memory footprint stable over
    # multiple hours.
    def __init__(self, max_pool_size: int = 4):
        self.max_pool_size = max_pool_size
        self.pools: Dict[Tuple[Tuple[int, ...], str], deque] = {}
        self.stats = ArrayStats()
        self.lock = threading.Lock()
        
    def get_array(self, shape: Tuple[int, ...], dtype: str = 'float32') -> np.ndarray:
        """配列をプールから取得（なければ新規作成）"""
        key = (shape, dtype)
        
        with self.lock:
            pool = self.pools.get(key, deque())
            
            if pool:
                array = pool.popleft()
                self.stats.cache_hits += 1
                # 配列をゼロクリア
                array.fill(0.0)
                return array
            else:
                self.stats.cache_misses += 1
                self.stats.total_allocations += 1
                return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        """配列をプールに返却"""
        if array is None:
            return
            
        key = (array.shape, str(array.dtype))
        
        with self.lock:
            pool = self.pools.setdefault(key, deque())
            
            if len(pool) < self.max_pool_size:
                pool.append(array)
                self.stats.total_deallocations += 1
            
            # 統計更新
            total_ops = self.stats.cache_hits + self.stats.cache_misses
            if total_ops > 0:
                self.stats.reuse_rate = self.stats.cache_hits / total_ops
    
    @contextmanager
    def temporary_array(self, shape: Tuple[int, ...], dtype: str = 'float32'):
        """一時配列のコンテキストマネージャー"""
        array = self.get_array(shape, dtype)
        try:
            yield array
        finally:
            self.return_array(array)
    
    def clear_pool(self):
        """プールをクリア"""
        with self.lock:
            self.pools.clear()
            self.stats = ArrayStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        with self.lock:
            return {
                'total_allocations': self.stats.total_allocations,
                'total_deallocations': self.stats.total_deallocations,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'reuse_rate': self.stats.reuse_rate,
                'pool_sizes': {str(k): len(v) for k, v in self.pools.items()},
                'total_pooled_arrays': sum(len(v) for v in self.pools.values())
            }


# グローバル配列プール
_global_array_pool: Optional[ArrayPool] = None
_pool_lock = threading.Lock()


def get_array_pool() -> ArrayPool:
    """グローバル配列プールを取得"""
    global _global_array_pool
    if _global_array_pool is None:
        with _pool_lock:
            if _global_array_pool is None:
                _global_array_pool = ArrayPool()
    return _global_array_pool


@dataclass
class OptimizedSearchResult:
    """メモリ最適化されたSearchResult（参照ベース）"""
    triangle_indices: List[int]
    distances: List[float]
    search_time_ms: float
    query_point_ref: np.ndarray  # copy()せず参照を保持
    search_radius: float
    num_nodes_visited: int
    
    @property
    def num_triangles(self) -> int:
        return len(self.triangle_indices)
    
    @property
    def query_point(self) -> np.ndarray:
        """読み取り専用で点座標を取得"""
        return self.query_point_ref
    
    def get_query_point_copy(self) -> np.ndarray:
        """必要時のみコピーを作成"""
        return self.query_point_ref.copy()


@dataclass 
class OptimizedCollisionEvent:
    """メモリ最適化されたCollisionEvent（参照ベース）"""
    event_id: str
    event_type: str  # EventType enum → str
    timestamp: float
    duration_ms: float
    
    # 参照ベースの位置データ（コピーを避ける）
    _contact_position_ref: np.ndarray
    _hand_position_ref: np.ndarray
    _surface_normal_ref: np.ndarray
    
    intensity: int  # CollisionIntensity → int
    velocity: float
    penetration_depth: float
    contact_area: float
    
    pitch_hint: float
    timbre_hint: float
    _spatial_position_ref: np.ndarray
    
    triangle_index: int
    hand_id: str
    collision_type: str  # CollisionType → str
    surface_properties: Dict[str, float] = field(default_factory=dict)
    
    @property
    def contact_position(self) -> np.ndarray:
        """読み取り専用で接触位置を取得"""
        return self._contact_position_ref
    
    @property
    def hand_position(self) -> np.ndarray:
        """読み取り専用で手位置を取得"""
        return self._hand_position_ref
    
    @property
    def surface_normal(self) -> np.ndarray:
        """読み取り専用で表面法線を取得"""
        return self._surface_normal_ref
    
    @property
    def spatial_position(self) -> np.ndarray:
        """読み取り専用で空間位置を取得"""
        return self._spatial_position_ref
    
    def get_contact_position_copy(self) -> np.ndarray:
        """必要時のみコピーを作成"""
        return self._contact_position_ref.copy()
    
    def get_hand_position_copy(self) -> np.ndarray:
        """必要時のみコピーを作成"""
        return self._hand_position_ref.copy()
    
    def get_surface_normal_copy(self) -> np.ndarray:
        """必要時のみコピーを作成"""
        return self._surface_normal_ref.copy()


class InPlaceOperations:
    """インプレース操作ユーティリティ"""
    
    @staticmethod
    def normalize_inplace(vector: np.ndarray, axis: int = -1) -> np.ndarray:
        """ベクトルをインプレースで正規化"""
        norm = np.linalg.norm(vector, axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)  # ゼロ除算回避
        vector /= norm
        return vector
    
    @staticmethod
    def subtract_inplace(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """インプレースで減算 (a -= b)"""
        a -= b
        return a
    
    @staticmethod
    def scale_inplace(array: np.ndarray, factor: float) -> np.ndarray:
        """インプレースでスケーリング"""
        array *= factor
        return array
    
    @staticmethod
    def clamp_inplace(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """インプレースでクランプ"""
        np.clip(array, min_val, max_val, out=array)
        return array


class ReferenceManager:
    """参照管理クラス（不要なコピーを検出・警告）"""
    
    def __init__(self, enable_tracking: bool = False):
        self.enable_tracking = enable_tracking
        self.reference_counts: Dict[int, int] = {}
        self.copy_warnings: List[str] = []
        self.lock = threading.Lock()
    
    def track_reference(self, array: np.ndarray, operation: str = "unknown") -> None:
        """配列参照を追跡"""
        if not self.enable_tracking:
            return
            
        with self.lock:
            array_id = id(array)
            self.reference_counts[array_id] = self.reference_counts.get(array_id, 0) + 1
    
    def track_copy(self, original: np.ndarray, copied: np.ndarray, operation: str = "unknown") -> None:
        """配列コピーを追跡・警告"""
        if not self.enable_tracking:
            return
            
        with self.lock:
            warning = f"Array copy detected in {operation}: {original.shape} -> {copied.shape}"
            self.copy_warnings.append(warning)
            
            # 大きな配列のコピーは警告ログ出力
            if original.nbytes > 1024 * 1024:  # 1MB以上
                from .. import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Large array copy detected: {original.nbytes / 1024 / 1024:.1f}MB in {operation}")
    
    def get_report(self) -> Dict[str, Any]:
        """参照追跡レポートを取得"""
        with self.lock:
            return {
                'active_references': len(self.reference_counts),
                'total_copy_warnings': len(self.copy_warnings),
                'recent_warnings': self.copy_warnings[-10:] if self.copy_warnings else [],
                'reference_histogram': dict(self.reference_counts)
            }


# グローバル参照マネージャー
_global_reference_manager: Optional[ReferenceManager] = None


def get_reference_manager() -> ReferenceManager:
    """グローバル参照マネージャーを取得"""
    global _global_reference_manager
    if _global_reference_manager is None:
        config = get_config()
        collision_config = config.collision
        enable_tracking = getattr(collision_config, 'enable_memory_tracking', False)
        _global_reference_manager = ReferenceManager(enable_tracking=enable_tracking)
    return _global_reference_manager


def optimize_array_operations(func):
    """配列操作最適化デコレータ"""
    def wrapper(*args, **kwargs):
        pool = get_array_pool()
        ref_manager = get_reference_manager()
        
        # 関数実行前の統計
        start_stats = pool.get_stats()
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # パフォーマンス統計更新
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            end_stats = pool.get_stats()
            
            if ref_manager.enable_tracking:
                from .. import get_logger
                logger = get_logger(__name__)
                allocations = end_stats['total_allocations'] - start_stats['total_allocations']
                if allocations > 5:  # 多量allocation警告
                    logger.debug(f"High allocation in {func.__name__}: {allocations} arrays, {elapsed_ms:.1f}ms")
    
    return wrapper


# 便利関数

def create_optimized_search_result(
    triangle_indices: List[int],
    distances: List[float], 
    search_time_ms: float,
    query_point: np.ndarray,  # 参照を直接使用
    search_radius: float,
    num_nodes_visited: int
) -> OptimizedSearchResult:
    """最適化されたSearchResultを作成"""
    return OptimizedSearchResult(
        triangle_indices=triangle_indices,
        distances=distances,
        search_time_ms=search_time_ms,
        query_point_ref=query_point,  # copy()しない
        search_radius=search_radius,
        num_nodes_visited=num_nodes_visited
    )


def create_optimized_collision_event(
    event_id: str,
    event_type: str,
    timestamp: float,
    contact_position: np.ndarray,  # 参照を直接使用
    hand_position: np.ndarray,     # 参照を直接使用
    surface_normal: np.ndarray,    # 参照を直接使用
    **kwargs
) -> OptimizedCollisionEvent:
    """最適化されたCollisionEventを作成"""
    
    # 空間位置を計算（Z成分を0にセット）
    pool = get_array_pool()
    with pool.temporary_array((3,), 'float32') as spatial_pos:
        spatial_pos[0] = contact_position[0]
        spatial_pos[1] = 0.0
        spatial_pos[2] = contact_position[2]
        
        return OptimizedCollisionEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            duration_ms=kwargs.get('duration_ms', 0.0),
            _contact_position_ref=contact_position,  # 参照のみ
            _hand_position_ref=hand_position,        # 参照のみ
            _surface_normal_ref=surface_normal,      # 参照のみ
            intensity=kwargs.get('intensity', 0),
            velocity=kwargs.get('velocity', 0.0),
            penetration_depth=kwargs.get('penetration_depth', 0.0),
            contact_area=kwargs.get('contact_area', 0.0),
            pitch_hint=kwargs.get('pitch_hint', 0.5),
            timbre_hint=kwargs.get('timbre_hint', 0.5),
            _spatial_position_ref=spatial_pos.copy(),  # 一時配列のコピー
            triangle_index=kwargs.get('triangle_index', -1),
            hand_id=kwargs.get('hand_id', "unknown"),
            collision_type=kwargs.get('collision_type', "surface"),
            surface_properties=kwargs.get('surface_properties', {})
        )


@contextmanager
def memory_efficient_context():
    """メモリ効率的な処理コンテキスト"""
    pool = get_array_pool()
    ref_manager = get_reference_manager()
    
    # 開始時統計
    start_stats = pool.get_stats()
    start_time = time.perf_counter()
    
    try:
        yield {
            'pool': pool,
            'reference_manager': ref_manager,
            'inplace_ops': InPlaceOperations()
        }
    finally:
        # 統計レポート
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        end_stats = pool.get_stats()
        
        if ref_manager.enable_tracking:
            from .. import get_logger
            logger = get_logger(__name__)
            allocations = end_stats['total_allocations'] - start_stats['total_allocations']
            logger.debug(f"Memory context: {allocations} allocations, {elapsed_ms:.1f}ms, "
                        f"reuse_rate={end_stats['reuse_rate']:.1%}")


def get_memory_optimization_stats() -> Dict[str, Any]:
    """メモリ最適化統計の取得"""
    pool = get_array_pool()
    ref_manager = get_reference_manager()
    
    return {
        'array_pool': pool.get_stats(),
        'reference_tracking': ref_manager.get_report(),
        'timestamp': time.time()
    }


def reset_memory_optimization():
    """メモリ最適化統計のリセット"""
    global _global_array_pool, _global_reference_manager
    
    if _global_array_pool:
        _global_array_pool.clear_pool()
    
    if _global_reference_manager:
        _global_reference_manager = None 