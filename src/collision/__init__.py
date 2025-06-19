"""
Geocussion-SP 衝突検出フェーズ

このパッケージは球体モデル化された手と地形メッシュの衝突判定を行い、
音響生成に必要な接触点情報と衝突イベントを生成する機能を提供します。

処理フロー:
1. 空間検索 (search.py) - BVHを使った近傍三角形の高速検索
2. 球-三角形判定 (sphere_tri.py) - 精密な衝突判定と接触点計算
3. イベント生成 (events.py) - 衝突強度と面属性のエンコード

予算時間: 5ms
"""

# 空間検索
from .search import (
    CollisionSearcher,
    SearchResult,
    search_nearby_triangles,
    batch_search_triangles
)

# 球-三角形衝突判定
from .sphere_tri import (
    SphereTriangleCollision,
    CollisionInfo,
    ContactPoint,
    check_sphere_triangle,
    calculate_contact_point,
    batch_collision_test
)

# 衝突イベント
from .events import (
    CollisionEvent,
    CollisionEventQueue,
    EventType,
    CollisionIntensity,
    create_collision_event,
    process_collision_events
)

__all__ = [
    # 空間検索
    'CollisionSearcher',
    'SearchResult',
    'search_nearby_triangles',
    'batch_search_triangles',
    
    # 球-三角形衝突
    'SphereTriangleCollision',
    'CollisionInfo',
    'ContactPoint',
    'check_sphere_triangle',
    'calculate_contact_point',
    'batch_collision_test',
    
    # イベント
    'CollisionEvent',
    'CollisionEventQueue',
    'EventType',
    'CollisionIntensity',
    'create_collision_event',
    'process_collision_events'
] 