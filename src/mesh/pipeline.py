from __future__ import annotations

"""Unified Mesh Pipeline

This module provides MeshPipeline – a facade that selects the best mesh-generation
strategy for a given frame.  Internally it can delegate to an IncrementalMeshUpdater
(local-patch update), a LODMeshGenerator (full regeneration with adaptive point
filtering) or simply return a cached mesh when appropriate.

The **first implementation goal** (T-MESH-101) is to wire-in an easy drop-in
replacement for the existing logic so that callers (mainly `FullPipelineViewer`)
can rely on a single entry point.

– Later milestones will extend the incremental path and add point-cloud diff logic.
"""

from dataclasses import dataclass
from typing import List, Optional
import time
import logging
import numpy as np

from .lod_mesh import LODMeshGenerator, TriangleMesh
from .incremental import IncrementalMeshUpdater
from ..detection.tracker import TrackedHand

logger = logging.getLogger(__name__)


@dataclass
class MeshPipelineStats:
    total_frames: int = 0
    lod_calls: int = 0
    incremental_calls: int = 0
    cache_hits: int = 0
    total_time_ms: float = 0.0

    def record(self, duration_ms: float, path: str) -> None:
        self.total_frames += 1
        self.total_time_ms += duration_ms
        if path == "lod":
            self.lod_calls += 1
        elif path == "inc":
            self.incremental_calls += 1
        elif path == "cache":
            self.cache_hits += 1

    @property
    def avg_time_ms(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_time_ms / self.total_frames


class MeshPipeline:  # pylint: disable=too-few-public-methods
    """Facade that hides mesh generation strategy details.

    Usage
    -----
    >>> pipeline = MeshPipeline()
    >>> mesh = pipeline.generate_mesh(points_3d, tracked_hands)
    """

    def __init__(
        self,
        lod_generator: Optional[LODMeshGenerator] = None,
        incremental_updater: Optional[IncrementalMeshUpdater] = None,
    ) -> None:
        self._lod = lod_generator or LODMeshGenerator()
        self._inc = incremental_updater  # may be None for the very first milestone
        self._cached_mesh: Optional[TriangleMesh] = None
        self._stats = MeshPipelineStats()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_mesh(
        self,
        points_3d: np.ndarray,
        tracked_hands: List[TrackedHand],
        force_update: bool = False,
    ) -> Optional[TriangleMesh]:
        """Return an up-to-date TriangleMesh or ``None`` if *points_3d* is empty.

        Parameters
        ----------
        points_3d
            Input point cloud (N×3) in metres.
        tracked_hands
            Hands currently detected/tracked.
        force_update
            Skip cache heuristics and recompute unconditionally.
        """
        if points_3d is None or len(points_3d) < 10:  # type: ignore[arg-type]
            return self._cached_mesh

        start = time.perf_counter()

        # Path selection: incremental > LOD > cache.
        selected_path = "cache"
        mesh: Optional[TriangleMesh] = None

        if force_update:
            # Explicit regeneration request
            mesh = self._lod.generate_mesh(points_3d, tracked_hands, force_update=True)
            selected_path = "lod"
        else:
            # If incremental updater is available and we have a previous mesh
            if self._inc and self._cached_mesh is not None:
                try:
                    mesh, did_full = self._inc.update_mesh(points_3d, force_full_update=False)
                    selected_path = "inc-full" if did_full else "inc"
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Incremental update failed – falling back to LOD (%s)", exc)
                    mesh = None

            # Fallback to LOD when incremental is not applicable/failed
            if mesh is None:
                mesh = self._lod.generate_mesh(points_3d, tracked_hands, force_update=False)
                selected_path = "lod"

        # Cache management
        if mesh is not None:
            self._cached_mesh = mesh
        else:
            mesh = self._cached_mesh  # keep previous mesh if generation failed

        elapsed = (time.perf_counter() - start) * 1000.0
        self._stats.record(elapsed, selected_path if selected_path else "cache")
        return mesh

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def get_stats(self) -> MeshPipelineStats:  # noqa: D401 simple verbs OK
        """Return aggregated performance statistics."""
        return self._stats


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_mesh_pipeline(
    *,
    enable_incremental: bool = False,
    lod_config: None | dict = None,
    inc_config: None | dict = None,
) -> MeshPipeline:
    """Convenience helper mirroring existing *create_* factory pattern."""

    lod_gen = LODMeshGenerator(**(lod_config or {}))
    inc_updater = IncrementalMeshUpdater(**(inc_config or {})) if enable_incremental else None
    return MeshPipeline(lod_generator=lod_gen, incremental_updater=inc_updater) 