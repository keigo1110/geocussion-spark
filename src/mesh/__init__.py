"""
Geocussion-SP 地形メッシュ生成フェーズ

このパッケージは点群から地形メッシュを生成し、衝突検出に必要な
属性とインデックスを構築する機能を提供します。

処理フロー:
1. 点群の2D投影 (projection.py)
2. Delaunay三角形分割 (delaunay.py) 
3. メッシュ簡略化 (simplify.py)
4. 属性計算 (attributes.py)
5. 空間インデックス構築 (index.py)

予算時間: 15ms
"""

# 2D投影とハイトマップ生成
from .projection import (
    PointCloudProjector,
    ProjectionMethod,
    HeightMap,
    create_height_map,
    project_points_to_grid
)

# Delaunay三角形分割
from .delaunay import (
    DelaunayTriangulator,
    TriangleMesh,
    create_mesh_from_heightmap,
    triangulate_points
)

# メッシュ簡略化
from .simplify import (
    MeshSimplifier,
    SimplificationMethod,
    simplify_mesh,
    reduce_triangle_count
)

# 属性計算（法線・曲率・勾配）
from .attributes import (
    AttributeCalculator,
    MeshAttributes,
    calculate_vertex_normals,
    calculate_face_normals,
    calculate_curvature,
    calculate_gradient,
    compute_mesh_attributes
)

# 空間インデックス
from .index import (
    SpatialIndex,
    IndexType,
    BVHNode,
    build_bvh_index,
    query_nearest_triangles,
    query_point_in_triangles
)

__all__ = [
    # 投影
    'PointCloudProjector',
    'ProjectionMethod', 
    'HeightMap',
    'create_height_map',
    'project_points_to_grid',
    
    # 三角形分割
    'DelaunayTriangulator',
    'TriangleMesh',
    'create_mesh_from_heightmap',
    'triangulate_points',
    
    # 簡略化
    'MeshSimplifier',
    'SimplificationMethod',
    'simplify_mesh',
    'reduce_triangle_count',
    
    # 属性
    'AttributeCalculator',
    'MeshAttributes',
    'calculate_vertex_normals',
    'calculate_face_normals', 
    'calculate_curvature',
    'calculate_gradient',
    'compute_mesh_attributes',
    
    # インデックス
    'SpatialIndex',
    'IndexType',
    'BVHNode',
    'build_bvh_index',
    'query_nearest_triangles',
    'query_point_in_triangles'
] 