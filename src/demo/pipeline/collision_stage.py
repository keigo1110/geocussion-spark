#!/usr/bin/env python3
"""
衝突検出ステージ: 手とメッシュの衝突検出

球体（手の関節）と三角形メッシュの衝突を検出します。
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from .base import PipelineStage, StageResult
from ...detection.tracker import TrackedHand
from ...collision.sphere_tri import check_sphere_triangle
from ...collision.events import CollisionEventQueue, CollisionEvent
from ...mesh.index import SpatialIndex, IndexType

# オプショナルインポート
try:
    from ...collision.distance_gpu import create_gpu_distance_calculator
    HAS_GPU_DISTANCE = True
except ImportError:
    HAS_GPU_DISTANCE = False
    create_gpu_distance_calculator = None


@dataclass
class CollisionStageConfig:
    """衝突検出ステージの設定"""
    enable_collision_detection: bool = True
    sphere_radius: float = 0.05
    enable_spatial_index: bool = True
    index_type: IndexType = IndexType.KDTREE
    # イベント設定
    event_cooldown: float = 0.15
    max_event_queue_size: int = 100
    # GPU加速
    enable_gpu_collision: bool = False


@dataclass
class CollisionStageResult(StageResult):
    """衝突検出ステージの処理結果"""
    collision_events: List[CollisionEvent] = None
    active_collisions: Dict[int, List[Tuple[int, float]]] = None  # hand_id -> [(triangle_idx, distance)]
    
    def __post_init__(self):
        if self.collision_events is None:
            self.collision_events = []
        if self.active_collisions is None:
            self.active_collisions = {}


class CollisionStage(PipelineStage):
    """衝突検出ステージの実装"""
    
    def __init__(self, config: CollisionStageConfig) -> None:
        """
        初期化
        
        Args:
            config: 衝突検出ステージ設定
        """
        super().__init__(config)
        self.config: CollisionStageConfig = config
        self.spatial_index: Optional[SpatialIndex] = None
        self.event_queue: Optional[CollisionEventQueue] = None
        self.gpu_calculator: Optional[Any] = None
        
    def initialize(self) -> bool:
        """ステージの初期化"""
        if not self.config.enable_collision_detection:
            self.logger.info("衝突検出は無効化されています")
            self._initialized = True
            return True
            
        try:
            
            # 空間インデックスは最初のメッシュで初期化
            if self.config.enable_spatial_index:
                self.logger.info("空間インデックスは最初のメッシュ処理時に初期化されます")
            
            # イベントキュー初期化
            self.event_queue = CollisionEventQueue(
                max_queue_size=self.config.max_event_queue_size
            )
            self.logger.info("衝突イベントキューを初期化しました")
            
            # GPU加速初期化
            if self.config.enable_gpu_collision and HAS_GPU_DISTANCE:
                self.gpu_calculator = create_gpu_distance_calculator()
                self.logger.info("GPU衝突計算器を初期化しました")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"衝突検出ステージの初期化に失敗: {e}")
            return False
    
    def process(self,
                tracked_hands: List[TrackedHand],
                vertices: Optional[np.ndarray] = None,
                triangles: Optional[np.ndarray] = None,
                mesh_updated: bool = False) -> CollisionStageResult:
        """
        衝突検出を実行
        
        Args:
            tracked_hands: トラッキング中の手のリスト
            vertices: メッシュ頂点
            triangles: メッシュ三角形
            mesh_updated: メッシュが更新されたかどうか
            
        Returns:
            衝突検出結果
        """
        if not self._initialized:
            return CollisionStageResult(
                success=False,
                error_message="Stage not initialized"
            )
        
        if not self.config.enable_collision_detection:
            return CollisionStageResult(success=True)
        
        if not tracked_hands or vertices is None or triangles is None:
            return CollisionStageResult(success=True)
        
        try:
            # 空間インデックスの初期化または更新
            if self.config.enable_spatial_index:
                if self.spatial_index is None:
                    from ...mesh.delaunay import TriangleMesh
                    mesh = TriangleMesh(vertices, triangles)
                    self.spatial_index = SpatialIndex(mesh, index_type=self.config.index_type)
                    self.logger.info(f"空間インデックスを初期化しました: {self.config.index_type}")
                elif mesh_updated:
                    self.spatial_index.build(vertices, triangles)
                    self.logger.debug("空間インデックスを更新しました")
            
            # 衝突検出実行
            active_collisions = {}
            new_events = []
            
            for tracked_hand in tracked_hands:
                hand = tracked_hand.hand
                hand_id = tracked_hand.id
                
                # 手の各関節について衝突検出
                hand_collisions = []
                
                for joint_idx, position in enumerate(hand.joints):
                    # 球体として衝突検出
                    sphere_center = position
                    sphere_radius = self.config.sphere_radius
                    
                    # 空間インデックスを使用した高速検索
                    if self.spatial_index:
                        candidate_triangles = self.spatial_index.query_sphere(
                            sphere_center, sphere_radius
                        )
                    else:
                        candidate_triangles = range(len(triangles))
                    
                    # 各候補三角形との衝突判定
                    for tri_idx in candidate_triangles:
                        triangle = vertices[triangles[tri_idx]]
                        
                        # GPU計算可能な場合
                        if self.gpu_calculator:
                            distance = self.gpu_calculator.compute_distance(
                                sphere_center, triangle
                            )
                            is_collision = distance <= sphere_radius
                        else:
                            # CPU計算（スタンドアロン関数使用）
                            contact = check_sphere_triangle(
                                sphere_center, sphere_radius, triangle
                            )
                            is_collision = contact is not None
                            distance = contact.penetration_depth if contact else float('inf')
                        
                        if is_collision:
                            hand_collisions.append((tri_idx, distance))
                            
                            # 衝突イベント生成
                            event = self._create_collision_event(
                                hand_id, joint_idx, tri_idx,
                                sphere_center, triangle, distance
                            )
                            if event:
                                new_events.append(event)
                
                if hand_collisions:
                    active_collisions[hand_id] = hand_collisions
            
            # イベントのフィルタリング（重複除去など）
            filtered_events = new_events
            
            self.logger.debug(f"衝突検出完了: {len(active_collisions)} hands, {len(filtered_events)} new events")
            
            return CollisionStageResult(
                success=True,
                collision_events=filtered_events,
                active_collisions=active_collisions
            )
            
        except Exception as e:
            self.logger.error(f"衝突検出エラー: {e}")
            return CollisionStageResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_collision_event(self,
                               hand_id: int,
                               joint_idx: int,
                               triangle_idx: int,
                               position: np.ndarray,
                               triangle: np.ndarray,
                               distance: float) -> Optional[CollisionEvent]:
        """衝突イベントを生成"""
        try:
            # 三角形の法線ベクトル計算
            v1 = triangle[1] - triangle[0]
            v2 = triangle[2] - triangle[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # 衝突の強度（距離に基づく）
            intensity = 1.0 - min(distance / self.config.sphere_radius, 1.0)
            
            return CollisionEvent(
                hand_id=hand_id,
                joint_index=joint_idx,
                triangle_index=triangle_idx,
                position=position.copy(),
                normal=normal,
                intensity=intensity
            )
            
        except Exception as e:
            self.logger.error(f"衝突イベント生成エラー: {e}")
            return None
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        self.spatial_index = None
        self.event_queue = None
        self.gpu_calculator = None
        self._initialized = False
        self.logger.info("衝突検出ステージをクリーンアップしました")