#!/usr/bin/env python3
"""
メッシュステージ: 地形メッシュ生成

点群データから地形メッシュを生成し、LOD処理や簡略化を行います。
"""

from typing import Optional, List, Tuple, Any
import numpy as np
from dataclasses import dataclass

from .base import PipelineStage, StageResult
from ...mesh.projection import PointCloudProjector, ProjectionMethod
from ...mesh.delaunay import DelaunayTriangulator
from ...mesh.simplify import MeshSimplifier

# オプショナルインポート
try:
    from ...mesh.lod_mesh import create_lod_mesh_generator
    HAS_LOD_MESH = True
except ImportError:
    HAS_LOD_MESH = False
    create_lod_mesh_generator = None

try:
    from ...mesh.delaunay_gpu import create_gpu_triangulator
    HAS_GPU_TRIANGULATOR = True
except ImportError:
    HAS_GPU_TRIANGULATOR = False
    create_gpu_triangulator = None


@dataclass
class MeshStageConfig:
    """メッシュステージの設定"""
    enable_mesh_generation: bool = True
    mesh_update_interval: int = 10
    max_mesh_skip_frames: int = 60
    # 投影設定
    projection_method: ProjectionMethod = ProjectionMethod.MEAN_HEIGHT
    mesh_resolution: float = 0.01
    # LOD設定
    enable_lod: bool = True
    lod_high_radius: float = 0.2
    lod_medium_radius: float = 0.5
    # 簡略化設定
    enable_simplification: bool = True
    mesh_quality: float = 0.3
    mesh_reduction: float = 0.7
    # GPU加速
    enable_gpu_triangulation: bool = False


@dataclass
class MeshStageResult(StageResult):
    """メッシュステージの処理結果"""
    vertices: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    mesh_updated: bool = False
    projected_points: Optional[np.ndarray] = None


class MeshStage(PipelineStage):
    """メッシュステージの実装"""
    
    def __init__(self, config: MeshStageConfig) -> None:
        """
        初期化
        
        Args:
            config: メッシュステージ設定
        """
        super().__init__(config)
        self.config: MeshStageConfig = config
        self.projector: Optional[PointCloudProjector] = None
        self.triangulator: Optional[Any] = None  # CPU or GPU
        self.simplifier: Optional[MeshSimplifier] = None
        self.lod_generator: Optional[Any] = None
        
        # 内部状態
        self._frame_count = 0
        self._last_mesh_update = 0
        self._cached_result: Optional[MeshStageResult] = None
        
    def initialize(self) -> bool:
        """ステージの初期化"""
        if not self.config.enable_mesh_generation:
            self.logger.info("メッシュ生成は無効化されています")
            self._initialized = True
            return True
            
        try:
            # 点群投影器初期化
            self.projector = PointCloudProjector(
                method=self.config.projection_method,
                resolution=self.config.mesh_resolution
            )
            self.logger.info(f"点群投影器を初期化しました: {self.config.projection_method}")
            
            # 三角形分割器初期化（GPU/CPU）
            if self.config.enable_gpu_triangulation and HAS_GPU_TRIANGULATOR:
                self.triangulator = create_gpu_triangulator()
                self.logger.info("GPU三角形分割器を初期化しました")
            else:
                self.triangulator = DelaunayTriangulator()
                self.logger.info("CPU三角形分割器を初期化しました")
            
            # メッシュ簡略化器初期化
            if self.config.enable_simplification:
                self.simplifier = MeshSimplifier(
                    quality_threshold=self.config.mesh_quality,
                    target_reduction=self.config.mesh_reduction
                )
                self.logger.info("メッシュ簡略化器を初期化しました")
            
            # LODジェネレーター初期化
            if self.config.enable_lod and HAS_LOD_MESH:
                self.lod_generator = create_lod_mesh_generator(
                    high_radius=self.config.lod_high_radius,
                    medium_radius=self.config.lod_medium_radius
                )
                self.logger.info("LODメッシュジェネレーターを初期化しました")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"メッシュステージの初期化に失敗: {e}")
            return False
    
    def process(self,
                point_cloud: Optional[np.ndarray] = None,
                colors: Optional[np.ndarray] = None,
                force_update: bool = False) -> MeshStageResult:
        """
        メッシュ生成を実行
        
        Args:
            point_cloud: 3D点群データ
            colors: 点群の色情報
            force_update: 強制的にメッシュを更新
            
        Returns:
            メッシュ生成結果
        """
        if not self._initialized:
            return MeshStageResult(
                success=False,
                error_message="Stage not initialized"
            )
        
        if not self.config.enable_mesh_generation:
            return MeshStageResult(success=True)
        
        # フレームカウント更新
        self._frame_count += 1
        
        # メッシュ更新判定
        should_update = (
            force_update or
            point_cloud is not None and (
                self._frame_count - self._last_mesh_update >= self.config.mesh_update_interval or
                self._cached_result is None or
                self._frame_count - self._last_mesh_update > self.config.max_mesh_skip_frames
            )
        )
        
        if not should_update:
            # キャッシュされた結果を返す
            if self._cached_result:
                return MeshStageResult(
                    success=True,
                    vertices=self._cached_result.vertices,
                    triangles=self._cached_result.triangles,
                    colors=self._cached_result.colors,
                    mesh_updated=False,
                    projected_points=self._cached_result.projected_points
                )
            else:
                return MeshStageResult(success=True, mesh_updated=False)
        
        if point_cloud is None or len(point_cloud) == 0:
            return MeshStageResult(
                success=False,
                error_message="No point cloud data provided"
            )
        
        try:
            # 点群を2D平面に投影
            projected_result = self.projector.project_points(point_cloud)
            
            # HeightMapオブジェクトの場合は点配列を生成
            if hasattr(projected_result, 'heights') and hasattr(projected_result, 'valid_mask'):
                # HeightMapから有効な点の2D座標を生成
                valid_indices = np.where(projected_result.valid_mask)
                y_coords, x_coords = valid_indices
                
                # グリッド座標を実世界座標に変換
                min_x, max_x, min_y, max_y = projected_result.bounds
                width, height = projected_result.shape[1], projected_result.shape[0]
                
                # 正規化座標を実座標に変換
                x_real = min_x + (x_coords / width) * (max_x - min_x)
                y_real = min_y + (y_coords / height) * (max_y - min_y)
                
                # 2D点群を作成 (N, 2)
                projected_points = np.column_stack([x_real, y_real])
                
                self.logger.debug(f"HeightMapから{len(projected_points)}個の2D点を生成しました")
            elif hasattr(projected_result, 'to_points'):
                projected_points = projected_result.to_points()
            elif hasattr(projected_result, 'points'):
                projected_points = projected_result.points
            else:
                projected_points = projected_result
            
            # 三角形分割
            result = self.triangulator.triangulate_points_2d(projected_points)
            if result is None:
                self.logger.warning("三角形分割に失敗しました")
                return MeshStageResult(
                    success=False,
                    error_message="Triangulation failed"
                )
            vertices_2d, triangles = result
            
            if triangles is None or len(triangles) == 0:
                self.logger.warning("三角形分割に失敗しました")
                return MeshStageResult(
                    success=False,
                    error_message="Triangulation failed"
                )
            
            # 頂点と三角形を準備
            vertices = point_cloud.copy()
            mesh_colors = colors.copy() if colors is not None else None
            
            # メッシュ簡略化
            if self.simplifier and len(triangles) > 1000:
                # TriangleMeshオブジェクトを作成
                from ...mesh.delaunay import TriangleMesh
                mesh = TriangleMesh(vertices, triangles)
                # 簡略化
                simplified_mesh = self.simplifier.simplify_mesh(mesh)
                if simplified_mesh:
                    vertices = simplified_mesh.vertices
                    triangles = simplified_mesh.triangles
                    self.logger.debug(f"メッシュを簡略化: {len(triangles)} triangles")
            
            # LOD処理
            if self.lod_generator:
                # TODO: カメラ位置に基づくLOD選択
                # 現在は簡略化のみ実装
                pass
            
            # 結果をキャッシュ
            self._cached_result = MeshStageResult(
                success=True,
                vertices=vertices,
                triangles=triangles,
                colors=mesh_colors,
                mesh_updated=True,
                projected_points=projected_result  # 元の投影結果を保存
            )
            self._last_mesh_update = self._frame_count
            
            self.logger.debug(f"メッシュ更新完了: {len(vertices)} vertices, {len(triangles)} triangles")
            
            return self._cached_result
            
        except Exception as e:
            self.logger.error(f"メッシュ生成エラー: {e}")
            return MeshStageResult(
                success=False,
                error_message=str(e)
            )
    
    def force_update(self) -> None:
        """次のフレームで強制的にメッシュを更新"""
        self._last_mesh_update = 0
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        self.projector = None
        self.triangulator = None
        self.simplifier = None
        self.lod_generator = None
        self._cached_result = None
        self._initialized = False
        self.logger.info("メッシュステージをクリーンアップしました")