from __future__ import annotations

"""Mountain clustering utility (MOUNT-INS-01)

This helper is executed **only when the ground mesh is regenerated** (≤1 Hz),
so a pure-Python implementation with NumPy is十分 fast for typical meshes
(数万 triangles).  The result is cached inside FullPipelineViewer and is looked
up at runtime with O(1) array indexing, adding essentially zero latency to the
collision→audio path.
"""

from typing import Dict, List, Tuple
import numpy as np

try:
    import open3d as o3d  # type: ignore
except ImportError:  # pragma: no cover – viewer may work headless
    o3d = None  # type: ignore

# Importing here avoids heavy import cost at startup yet shares the same Enum
from src.sound.mapping import InstrumentType  # pylint: disable=wrong-import-position


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def _build_vertex_triangle_map(triangles: np.ndarray, n_vertices: int) -> List[List[int]]:
    """return adjacency: vertex_id -> list of triangle indices"""
    vt_map: List[List[int]] = [[] for _ in range(n_vertices)]
    for tidx, (v0, v1, v2) in enumerate(triangles):
        vt_map[v0].append(tidx)
        vt_map[v1].append(tidx)
        vt_map[v2].append(tidx)
    return vt_map


def cluster_mountains(
    mesh: "o3d.geometry.TriangleMesh",
    slope_thresh_deg: float = 35.0,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Flood-fill triangles that are reasonably co-planar to form *mountains*.

    Returns
    -------
    triangle_to_mountain : np.ndarray[int32]
        Mapping of triangle_index → mountain_id (0..M-1)
    mountain_area : Dict[int, float]
        Sum of surface area per mountain (m²)
    """
    if mesh.is_empty():
        return np.full((0,), -1, dtype=np.int32), {}

    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    n_tri = triangles.shape[0]

    # Calculate per-triangle normals (unit)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals /= norm_len

    # Triangle areas (0.5*|cross|)
    tri_area = 0.5 * norm_len.squeeze(-1)

    # Vertex→triangle adjacency for flood-fill
    vt_map = _build_vertex_triangle_map(triangles, vertices.shape[0])

    tri_to_mnt = np.full(n_tri, -1, dtype=np.int32)
    mountain_area: Dict[int, float] = {}
    slope_thresh = np.cos(np.deg2rad(slope_thresh_deg))  # dot threshold

    current_mnt = 0
    for t_idx in range(n_tri):
        if tri_to_mnt[t_idx] != -1:
            continue
        # Start new mountain
        stack = [t_idx]
        tri_to_mnt[t_idx] = current_mnt
        area_sum = 0.0

        while stack:
            tid = stack.pop()
            area_sum += float(tri_area[tid])
            # Iterate neighbors via shared vertices
            for v in triangles[tid]:
                for nb in vt_map[v]:
                    if tri_to_mnt[nb] != -1:
                        continue
                    # Planar similarity check via normal dot
                    if abs(np.dot(normals[tid], normals[nb])) >= slope_thresh:
                        tri_to_mnt[nb] = current_mnt
                        stack.append(nb)
        mountain_area[current_mnt] = area_sum
        current_mnt += 1

    return tri_to_mnt, mountain_area


# -----------------------------------------------------------------------------
# Instrument assignment
# -----------------------------------------------------------------------------

def assign_instruments_by_area(
    mountain_area: Dict[int, float],
    *,
    area_mid: float = 0.015,
    area_large: float = 0.03,
) -> Dict[int, InstrumentType]:
    """Simple 3-level mapping based on surface area (m²).

    area ≥ area_large   → SYNTH_PAD
    area ≥ area_mid     → STRING
    else                → MARIMBA
    """
    table: Dict[int, InstrumentType] = {}
    for mnt_id, area in mountain_area.items():
        if area >= area_large:
            table[mnt_id] = InstrumentType.SYNTH_PAD
        elif area >= area_mid:
            table[mnt_id] = InstrumentType.STRING
        else:
            table[mnt_id] = InstrumentType.MARIMBA
    return table 