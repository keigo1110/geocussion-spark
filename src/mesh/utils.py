from __future__ import annotations

"""Mesh utility helpers.

This module hosts small helper functions that are shared across mesh
sub-modules without introducing unwanted import cycles.
"""

from typing import Tuple
import numpy as np

from .delaunay import TriangleMesh


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_triangle_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute per-triangle normals (unit length)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals /= norms
    return normals


def compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray, tri_normals: np.ndarray) -> np.ndarray:
    """Compute vertex normals as area-weighted average of adjacent triangle normals."""
    vert_normals = np.zeros_like(vertices)
    for tri, n in zip(triangles, tri_normals):
        vert_normals[tri] += n
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True) + 1e-12
    vert_normals /= norms
    return vert_normals


def rebuild_mesh_postprocess(mesh: TriangleMesh) -> TriangleMesh:  # noqa: D401 simple verbs OK
    """Rebuild normals arrays for *mesh* and return the same instance.

    This is mainly used after geometry-only operations (e.g. mesh joined or
    regenerated) to ensure downstream consumers (collision / shading) see
    consistent data.
    """
    tri_normals = compute_triangle_normals(mesh.vertices, mesh.triangles)
    vert_normals = compute_vertex_normals(mesh.vertices, mesh.triangles, tri_normals)
    mesh.triangle_normals = tri_normals
    mesh.vertex_normals = vert_normals
    return mesh 