#!/usr/bin/env python3
"""
インクリメンタル メッシュ更新モジュール

フレーム間での点群変化を監視し、変更領域のみを局所的に更新する
効率的なメッシュ生成システムを提供します。

主要機能:
- 点群変化検出
- 局所的メッシュ更新
- メッシュマージ機能
- パフォーマンス統計
"""

import time
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import numpy as np
import cv2

from .delaunay import TriangleMesh, DelaunayTriangulator
from .projection import PointCloudProjector, HeightMap
from .simplify import MeshSimplifier
from .. import get_logger

logger = get_logger(__name__)


@dataclass
class UpdateRegion:
    """更新領域情報"""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    change_ratio: float
    point_count: int


@dataclass
class IncrementalStats:
    """インクリメンタル更新統計"""
    total_updates: int = 0
    total_time_ms: float = 0.0
    full_updates: int = 0
    partial_updates: int = 0
    cache_hits: int = 0
    average_change_ratio: float = 0.0
    average_update_time_ms: float = 0.0


class IncrementalMeshUpdater:
    """インクリメンタル メッシュ更新器"""
    
    def __init__(
        self,
        change_threshold: float = 0.1,          # 10% 変更で更新
        max_age_frames: int = 30,               # 30フレームで強制更新
        grid_resolution: float = 0.02,          # 2cm解像度
        enable_caching: bool = True,
        enable_statistics: bool = True
    ):
        """
        初期化
        
        Args:
            change_threshold: 更新を行う変更率閾値
            max_age_frames: 強制全更新までのフレーム数
            grid_resolution: グリッド解像度 (m)
            enable_caching: キャッシュの有効化
            enable_statistics: 統計の有効化
        """
        self.change_threshold = change_threshold
        self.max_age_frames = max_age_frames
        self.grid_resolution = grid_resolution
        self.enable_caching = enable_caching
        self.enable_statistics = enable_statistics
        
        # コンポーネント
        self.projector = PointCloudProjector(resolution=grid_resolution)
        self.triangulator = DelaunayTriangulator()
        self.simplifier = MeshSimplifier()
        
        # 状態管理
        self.previous_heightmap: Optional[HeightMap] = None
        self.current_mesh: Optional[TriangleMesh] = None
        self.frame_count = 0
        self.last_full_update_frame = 0
        
        # キャッシュ
        self.region_cache: Dict[str, TriangleMesh] = {}
        
        # 統計
        self.stats = IncrementalStats()
    
    def update_mesh(
        self, 
        points_3d: np.ndarray,
        force_full_update: bool = False
    ) -> Tuple[TriangleMesh, bool]:
        """
        メッシュを更新
        
        Args:
            points_3d: 3D点群 (N, 3)
            force_full_update: 強制的な全更新
            
        Returns:
            (更新されたメッシュ, 全更新フラグ)
        """
        start_time = time.perf_counter()
        self.frame_count += 1
        
        # 点群を投影してハイトマップ生成
        current_heightmap = self.projector.project_points(points_3d)
        
        # 更新方針を決定
        update_decision = self._decide_update_strategy(
            current_heightmap, force_full_update
        )
        
        if update_decision["strategy"] == "full":
            # 全更新
            mesh, is_full_update = self._perform_full_update(current_heightmap)
            self.last_full_update_frame = self.frame_count
            
        elif update_decision["strategy"] == "incremental":
            # インクリメンタル更新
            mesh, is_full_update = self._perform_incremental_update(
                current_heightmap, update_decision["regions"]
            )
            
        else:
            # 更新なし（キャッシュ使用）
            mesh, is_full_update = self.current_mesh, False
            if self.enable_statistics:
                self.stats.cache_hits += 1
        
        # 状態更新
        self.previous_heightmap = current_heightmap
        self.current_mesh = mesh
        
        # 統計更新
        if self.enable_statistics:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._update_statistics(elapsed_ms, update_decision, is_full_update)
        
        return mesh, is_full_update
    
    def _decide_update_strategy(
        self, 
        current_heightmap: HeightMap,
        force_full_update: bool
    ) -> Dict[str, Any]:
        """更新戦略を決定"""
        
        # 強制全更新または初回更新
        if (force_full_update or 
            self.previous_heightmap is None or 
            self.current_mesh is None):
            return {"strategy": "full", "regions": []}
        
        # 最大エイジ超過チェック
        frames_since_full_update = self.frame_count - self.last_full_update_frame
        if frames_since_full_update >= self.max_age_frames:
            return {"strategy": "full", "regions": []}
        
        # 変更領域検出
        change_regions = self._detect_change_regions(current_heightmap)
        
        if not change_regions:
            return {"strategy": "cache", "regions": []}
        
        # 全体変更率計算
        total_change_ratio = sum(r.change_ratio for r in change_regions) / len(change_regions)
        
        if total_change_ratio >= self.change_threshold:
            return {"strategy": "full", "regions": change_regions}
        else:
            return {"strategy": "incremental", "regions": change_regions}
    
    def _detect_change_regions(self, current_heightmap: HeightMap) -> List[UpdateRegion]:
        """変更領域を検出"""
        if self.previous_heightmap is None:
            return []
        
        # ハイトマップサイズ確認
        if (current_heightmap.heights.shape != self.previous_heightmap.heights.shape):
            return []  # サイズ変更時は全更新
        
        # 差分計算
        height_diff = np.abs(current_heightmap.heights - self.previous_heightmap.heights)
        valid_mask = current_heightmap.valid_mask & self.previous_heightmap.valid_mask
        
        # 変更閾値（相対値）
        height_threshold = 0.02  # 2cm
        change_mask = (height_diff > height_threshold) & valid_mask
        
        if not np.any(change_mask):
            return []
        
        # 連結成分解析で変更領域をグループ化
        change_regions = self._group_change_regions(change_mask, valid_mask)
        
        return change_regions
    
    def _group_change_regions(
        self, 
        change_mask: np.ndarray, 
        valid_mask: np.ndarray
    ) -> List[UpdateRegion]:
        """変更領域をグループ化"""
        # モルフォロジー処理で小さなノイズを除去
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(
            change_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        
        # 連結成分検出
        num_labels, labels = cv2.connectedComponents(cleaned_mask.astype(np.uint8))
        
        regions = []
        height, width = change_mask.shape
        
        for label_id in range(1, num_labels + 1):  # ラベル0は背景
            component_mask = (labels == label_id)
            
            # 成分サイズチェック
            component_size = np.sum(component_mask)
            if component_size < 10:  # 最小サイズ閾値
                continue
            
            # バウンディングボックス計算
            rows, cols = np.where(component_mask)
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            
            # マージンを追加
            margin = 5
            x_min = max(0, x_min - margin)
            x_max = min(width - 1, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(height - 1, y_max + margin)
            
            # 変更率計算
            region_mask = valid_mask[y_min:y_max+1, x_min:x_max+1]
            region_change = change_mask[y_min:y_max+1, x_min:x_max+1]
            change_ratio = np.sum(region_change) / max(np.sum(region_mask), 1)
            
            regions.append(UpdateRegion(
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                change_ratio=change_ratio,
                point_count=component_size
            ))
        
        return regions
    
    def _perform_full_update(self, heightmap: HeightMap) -> Tuple[TriangleMesh, bool]:
        """全メッシュ更新"""
        try:
            mesh = self.triangulator.triangulate_heightmap(heightmap)
            simplified_mesh = self.simplifier.simplify_mesh(mesh)
            
            if self.enable_statistics:
                self.stats.full_updates += 1
            
            return simplified_mesh, True
            
        except Exception as e:
            logger.warning(f"Full mesh update failed: {e}")
            return self.current_mesh or TriangleMesh(
                vertices=np.empty((0, 3)), triangles=np.empty((0, 3), dtype=int)
            ), False
    
    def _perform_incremental_update(
        self, 
        heightmap: HeightMap, 
        regions: List[UpdateRegion]
    ) -> Tuple[TriangleMesh, bool]:
        """インクリメンタル更新"""
        try:
            updated_mesh = self.current_mesh
            
            for region in regions:
                # 領域メッシュを更新
                region_mesh = self._update_region_mesh(heightmap, region)
                if region_mesh:
                    # メッシュマージ（簡略実装）
                    updated_mesh = self._merge_meshes(updated_mesh, region_mesh, region)
            
            if self.enable_statistics:
                self.stats.partial_updates += 1
            
            return updated_mesh, False
            
        except Exception as e:
            logger.warning(f"Incremental update failed: {e}, falling back to full update")
            return self._perform_full_update(heightmap)
    
    def _update_region_mesh(
        self, 
        heightmap: HeightMap, 
        region: UpdateRegion
    ) -> Optional[TriangleMesh]:
        """領域メッシュ更新"""
        # 領域ハイトマップ抽出
        region_heights = heightmap.heights[
            region.y_min:region.y_max+1, 
            region.x_min:region.x_max+1
        ]
        region_valid = heightmap.valid_mask[
            region.y_min:region.y_max+1, 
            region.x_min:region.x_max+1
        ]
        
        if not np.any(region_valid):
            return None
        
        # 小さなハイトマップを作成
        region_heightmap = HeightMap(
            heights=region_heights,
            densities=np.ones_like(region_heights),
            bounds=heightmap.bounds,  # 簡略化
            resolution=heightmap.resolution,
            valid_mask=region_valid
        )
        
        # 三角分割
        try:
            region_mesh = self.triangulator.triangulate_heightmap(region_heightmap)
            return region_mesh
        except Exception as e:
            logger.debug(f"Region mesh update failed: {e}")
            return None
    
    def _merge_meshes(
        self, 
        base_mesh: TriangleMesh, 
        region_mesh: TriangleMesh, 
        region: UpdateRegion
    ) -> TriangleMesh:
        """メッシュマージ（簡略実装）"""
        # 簡易実装: 領域外の既存頂点 + 新しい領域メッシュ
        # 実際の実装では、境界での適切なステッチングが必要
        
        if base_mesh is None or base_mesh.num_vertices == 0:
            return region_mesh
        
        if region_mesh is None or region_mesh.num_vertices == 0:
            return base_mesh
        
        # 新しい複合メッシュを作成（簡略版）
        new_vertices = np.vstack([base_mesh.vertices, region_mesh.vertices])
        
        # 三角形インデックスを調整
        region_triangles_adjusted = region_mesh.triangles + base_mesh.num_vertices
        new_triangles = np.vstack([base_mesh.triangles, region_triangles_adjusted])
        
        return TriangleMesh(vertices=new_vertices, triangles=new_triangles)
    
    def _update_statistics(
        self, 
        elapsed_ms: float, 
        update_decision: Dict[str, Any], 
        is_full_update: bool
    ):
        """統計更新"""
        self.stats.total_updates += 1
        self.stats.total_time_ms += elapsed_ms
        
        if update_decision["regions"]:
            avg_change = np.mean([r.change_ratio for r in update_decision["regions"]])
            self.stats.average_change_ratio = (
                (self.stats.average_change_ratio * (self.stats.total_updates - 1) + avg_change) 
                / self.stats.total_updates
            )
        
        self.stats.average_update_time_ms = (
            self.stats.total_time_ms / self.stats.total_updates
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            "total_updates": self.stats.total_updates,
            "full_updates": self.stats.full_updates,
            "partial_updates": self.stats.partial_updates,
            "cache_hits": self.stats.cache_hits,
            "average_change_ratio": self.stats.average_change_ratio,
            "average_update_time_ms": self.stats.average_update_time_ms,
            "update_efficiency": (
                self.stats.partial_updates / max(self.stats.total_updates, 1)
            ),
            "current_frame": self.frame_count
        }
    
    def reset_statistics(self):
        """統計リセット"""
        self.stats = IncrementalStats()
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.region_cache.clear()
        logger.debug("Incremental mesh cache cleared")


# 便利関数

def create_incremental_updater(
    change_threshold: float = 0.1,
    grid_resolution: float = 0.02
) -> IncrementalMeshUpdater:
    """
    インクリメンタル更新器を作成（簡単なインターフェース）
    
    Args:
        change_threshold: 更新閾値
        grid_resolution: グリッド解像度
        
    Returns:
        インクリメンタル更新器
    """
    return IncrementalMeshUpdater(
        change_threshold=change_threshold,
        grid_resolution=grid_resolution
    ) 