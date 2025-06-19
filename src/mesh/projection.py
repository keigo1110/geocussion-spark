#!/usr/bin/env python3
"""
点群投影とハイトマップ生成

3D点群をXY平面に投影し、密度マップやハイトマップを生成する機能を提供します。
地形メッシュ生成の最初のステップとして使用されます。
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import numpy as np
import cv2
from scipy import ndimage


class ProjectionMethod(Enum):
    """投影方式の列挙"""
    MIN_HEIGHT = "min"          # 各グリッドセルの最小高度
    MAX_HEIGHT = "max"          # 各グリッドセルの最大高度
    MEAN_HEIGHT = "mean"        # 各グリッドセルの平均高度
    MEDIAN_HEIGHT = "median"    # 各グリッドセルの中央値高度
    DENSITY = "density"         # 各グリッドセルの点密度


@dataclass
class HeightMap:
    """ハイトマップデータ構造"""
    heights: np.ndarray          # 高度データ (H, W)
    densities: np.ndarray        # 点密度データ (H, W)
    bounds: Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
    resolution: float            # グリッド解像度 (m/pixel)
    valid_mask: np.ndarray      # 有効ピクセルマスク (H, W)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """マップサイズを取得"""
        return self.heights.shape
    
    @property
    def width(self) -> int:
        """幅を取得"""
        return self.heights.shape[1]
    
    @property
    def height(self) -> int:
        """高さを取得"""
        return self.heights.shape[0]
    
    def get_world_coordinates(self, row: int, col: int) -> Tuple[float, float]:
        """グリッド座標を世界座標に変換"""
        min_x, max_x, min_y, max_y = self.bounds
        x = min_x + col * self.resolution
        y = max_y - row * self.resolution  # Y軸は上下反転
        return x, y
    
    def get_grid_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """世界座標をグリッド座標に変換"""
        min_x, max_x, min_y, max_y = self.bounds
        col = int((x - min_x) / self.resolution)
        row = int((max_y - y) / self.resolution)  # Y軸は上下反転
        return row, col


class PointCloudProjector:
    """点群投影クラス"""
    
    def __init__(
        self,
        resolution: float = 0.01,  # 1cm解像度
        method: ProjectionMethod = ProjectionMethod.MEAN_HEIGHT,
        smooth_factor: float = 0.0,  # ガウシアン平滑化係数
        fill_holes: bool = True,     # 穴埋め処理
        min_points_per_cell: int = 1 # セルあたりの最小点数
    ):
        """
        初期化
        
        Args:
            resolution: グリッド解像度 (m/pixel)
            method: 投影方式
            smooth_factor: ガウシアン平滑化係数 (0で無効)
            fill_holes: 穴埋め処理を行うか
            min_points_per_cell: セルあたりの最小点数
        """
        self.resolution = resolution
        self.method = method
        self.smooth_factor = smooth_factor
        self.fill_holes = fill_holes
        self.min_points_per_cell = min_points_per_cell
        
        # パフォーマンス統計
        self.stats = {
            'total_projections': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_points': 0,
            'last_grid_size': (0, 0)
        }
    
    def project_points(self, points: np.ndarray) -> HeightMap:
        """
        点群をハイトマップに投影
        
        Args:
            points: 点群データ (N, 3) - (x, y, z)
            
        Returns:
            生成されたハイトマップ
        """
        start_time = time.perf_counter()
        
        if len(points) == 0:
            raise ValueError("Empty point cloud")
        
        if points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3), got {points.shape}")
        
        # バウンディングボックス計算
        min_x, min_y = np.min(points[:, :2], axis=0)
        max_x, max_y = np.max(points[:, :2], axis=0)
        
        # グリッドサイズ計算
        width = int(np.ceil((max_x - min_x) / self.resolution)) + 1
        height = int(np.ceil((max_y - min_y) / self.resolution)) + 1
        
        # 境界を調整（余裕を持たせる）
        margin = self.resolution
        bounds = (min_x - margin, max_x + margin, min_y - margin, max_y + margin)
        
        # グリッド座標計算
        grid_x = ((points[:, 0] - bounds[0]) / self.resolution).astype(np.int32)
        grid_y = ((bounds[3] - points[:, 1]) / self.resolution).astype(np.int32)  # Y軸反転
        
        # 範囲外の点を除外
        valid_mask = (
            (grid_x >= 0) & (grid_x < width) &
            (grid_y >= 0) & (grid_y < height)
        )
        
        if not np.any(valid_mask):
            raise ValueError("No valid points in grid")
        
        valid_points = points[valid_mask]
        valid_grid_x = grid_x[valid_mask]
        valid_grid_y = grid_y[valid_mask]
        
        # ハイトマップ生成
        heights, densities, valid_pixels = self._create_heightmap(
            valid_points[:, 2],  # Z座標
            valid_grid_x,
            valid_grid_y,
            height,
            width
        )
        
        # 後処理
        if self.fill_holes:
            heights = self._fill_holes(heights, valid_pixels)
            
        if self.smooth_factor > 0:
            heights = self._smooth_heightmap(heights, valid_pixels)
        
        # パフォーマンス統計更新
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(elapsed_ms, len(points), (height, width))
        
        return HeightMap(
            heights=heights,
            densities=densities,
            bounds=bounds,
            resolution=self.resolution,
            valid_mask=valid_pixels
        )
    
    def _create_heightmap(
        self,
        heights: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_height: int,
        grid_width: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ハイトマップを生成"""
        
        # 出力配列初期化
        height_map = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        density_map = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        # グリッドセルごとの点をグループ化
        coords = grid_y * grid_width + grid_x
        unique_coords, inverse_indices = np.unique(coords, return_inverse=True)
        
        for i, coord in enumerate(unique_coords):
            mask = inverse_indices == i
            cell_heights = heights[mask]
            
            if len(cell_heights) < self.min_points_per_cell:
                continue
                
            row = coord // grid_width
            col = coord % grid_width
            
            # 投影方式に応じて高度計算
            if self.method == ProjectionMethod.MIN_HEIGHT:
                height_value = np.min(cell_heights)
            elif self.method == ProjectionMethod.MAX_HEIGHT:
                height_value = np.max(cell_heights)
            elif self.method == ProjectionMethod.MEAN_HEIGHT:
                height_value = np.mean(cell_heights)
            elif self.method == ProjectionMethod.MEDIAN_HEIGHT:
                height_value = np.median(cell_heights)
            else:  # DENSITY
                height_value = len(cell_heights)
            
            height_map[row, col] = height_value
            density_map[row, col] = len(cell_heights)
        
        valid_mask = ~np.isnan(height_map)
        return height_map, density_map, valid_mask
    
    def _fill_holes(self, heights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """穴埋め処理（近傍補間）"""
        if not self.fill_holes or np.all(valid_mask):
            return heights
        
        # マスクされた配列を作成
        masked_heights = np.ma.masked_array(heights, mask=~valid_mask)
        
        # 小さな穴を補間で埋める
        filled_heights = heights.copy()
        
        # Invalid領域を特定
        invalid_mask = ~valid_mask
        
        # 小さな穴のみを対象にする（例：5x5以下の連結領域）
        labeled_holes, num_holes = ndimage.label(invalid_mask)
        
        for hole_id in range(1, num_holes + 1):
            hole_mask = labeled_holes == hole_id
            hole_size = np.sum(hole_mask)
            
            if hole_size <= 25:  # 5x5以下の穴
                # 境界の平均値で埋める
                dilated = ndimage.binary_dilation(hole_mask)
                boundary = dilated & ~hole_mask & valid_mask
                
                if np.any(boundary):
                    avg_height = np.mean(heights[boundary])
                    filled_heights[hole_mask] = avg_height
        
        return filled_heights
    
    def _smooth_heightmap(self, heights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """ガウシアン平滑化"""
        if self.smooth_factor <= 0:
            return heights
        
        # 有効領域のみを平滑化
        smoothed = heights.copy()
        valid_heights = heights[valid_mask]
        
        if len(valid_heights) > 0:
            # ガウシアンフィルタ適用
            sigma = self.smooth_factor / self.resolution
            smoothed = ndimage.gaussian_filter(heights, sigma=sigma, mode='nearest')
            
            # 無効領域は元の値を保持
            smoothed[~valid_mask] = heights[~valid_mask]
        
        return smoothed
    
    def _update_stats(self, elapsed_ms: float, num_points: int, grid_size: Tuple[int, int]):
        """パフォーマンス統計更新"""
        self.stats['total_projections'] += 1
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['average_time_ms'] = self.stats['total_time_ms'] / self.stats['total_projections']
        self.stats['last_num_points'] = num_points
        self.stats['last_grid_size'] = grid_size
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            'total_projections': 0,
            'total_time_ms': 0.0,
            'average_time_ms': 0.0,
            'last_num_points': 0,
            'last_grid_size': (0, 0)
        }


# 便利関数

def create_height_map(
    points: np.ndarray,
    resolution: float = 0.01,
    method: ProjectionMethod = ProjectionMethod.MEAN_HEIGHT
) -> HeightMap:
    """
    点群からハイトマップを作成（簡単なインターフェース）
    
    Args:
        points: 点群データ (N, 3)
        resolution: グリッド解像度
        method: 投影方式
        
    Returns:
        ハイトマップ
    """
    projector = PointCloudProjector(resolution=resolution, method=method)
    return projector.project_points(points)


def project_points_to_grid(
    points: np.ndarray,
    grid_size: Tuple[int, int],
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    点群を指定サイズのグリッドに投影
    
    Args:
        points: 点群データ (N, 3)
        grid_size: グリッドサイズ (height, width)
        bounds: 境界 (min_x, max_x, min_y, max_y)
        
    Returns:
        (heights, densities) のタプル
    """
    if bounds is None:
        min_x, min_y = np.min(points[:, :2], axis=0)
        max_x, max_y = np.max(points[:, :2], axis=0)
        bounds = (min_x, max_x, min_y, max_y)
    
    height, width = grid_size
    min_x, max_x, min_y, max_y = bounds
    
    # 解像度計算
    resolution_x = (max_x - min_x) / width
    resolution_y = (max_y - min_y) / height
    resolution = max(resolution_x, resolution_y)
    
    # 投影
    projector = PointCloudProjector(resolution=resolution)
    heightmap = projector.project_points(points)
    
    # サイズ調整
    if heightmap.shape != grid_size:
        heights_resized = cv2.resize(heightmap.heights, (width, height), interpolation=cv2.INTER_LINEAR)
        densities_resized = cv2.resize(heightmap.densities.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        return heights_resized, densities_resized.astype(np.int32)
    
    return heightmap.heights, heightmap.densities 