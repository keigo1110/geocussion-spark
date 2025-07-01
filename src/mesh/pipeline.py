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
from typing import List, Optional, Tuple
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


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------


@dataclass
class MeshResult:
    """Return type for MeshPipeline.generate_mesh().

    Attributes
    ----------
    mesh : TriangleMesh | None
        The triangle mesh returned by the pipeline.  *None* signals that
        generation failed (e.g. not enough points).
    changed : bool
        True if a *new mesh* was generated during this call.  False when the
        cached mesh was reused.
    needs_refresh: bool
        True if the mesh needs to be refreshed due to point cloud changes.
    """

    mesh: Optional[TriangleMesh]
    changed: bool = False
    needs_refresh: bool = False


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

        # --- Point-cloud diff state (T-PERF-001) --------------------------
        # Keep previous samples/hashes so that we can detect meaningful changes
        # across successive frames without incurring heavy computations.
        self._prev_sample: Optional[np.ndarray] = None
        self._prev_vox_set: Optional[set[tuple[int, int, int]]] = None
        self._prev_count: Optional[int] = None

        # Fixed RNG for reproducible sampling (avoid every-frame diff jitter)
        import numpy as _np  # local import to avoid top-level requirement in modules missing NumPy
        self._rng = _np.random.default_rng(1234)

        # EWMA smoothers for voxel hash distance & count ratio
        from .utils import EWMASmoother

        self._ewma_voxel = EWMASmoother(alpha=0.3)
        self._ewma_count = EWMASmoother(alpha=0.3)

        # Debounce timer for expensive regeneration
        self._last_regen_ts: float = 0.0

        # Sample indices used for fixed subsampling
        self._sample_idx: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_mesh(
        self,
        points_3d: np.ndarray,
        tracked_hands: List[TrackedHand],
        force_update: bool = False,
    ) -> MeshResult:
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
            return MeshResult(self._cached_mesh, changed=False, needs_refresh=False)

        start = time.perf_counter()

        # Path selection: incremental > LOD > cache.
        selected_path = "cache"
        mesh: Optional[TriangleMesh] = None

        refresh_due_to_diff = False

        if not force_update and self._cached_mesh is not None:
            # Debounce: if last regeneration happened <0.2s ago use cache directly
            if (time.perf_counter() - self._last_regen_ts) < 0.2:
                elapsed = (time.perf_counter() - start) * 1000.0
                self._stats.record(elapsed, "cache")
                return MeshResult(self._cached_mesh, changed=False, needs_refresh=False)

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
            # Detect whether this mesh differs from cached one (object identity)
            mesh_changed = mesh is not self._cached_mesh
            self._cached_mesh = mesh
        else:
            mesh = self._cached_mesh  # keep previous mesh if generation failed
            mesh_changed = False

        elapsed = (time.perf_counter() - start) * 1000.0
        self._stats.record(elapsed, selected_path if selected_path else "cache")

        if selected_path != "cache":
            # remember last heavy regeneration ts
            self._last_regen_ts = time.perf_counter()

        return MeshResult(mesh, changed=mesh_changed, needs_refresh=(refresh_due_to_diff or mesh_changed))

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def get_stats(self) -> MeshPipelineStats:  # noqa: D401 simple verbs OK
        """Return aggregated performance statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers – point-cloud diff
    # ------------------------------------------------------------------

    _SAMPLE_SIZE: int = 5000
    _DIFF_THRESHOLD_M: float = 0.03  # relaxed threshold (T-PERF-001)

    def _evaluate_pointcloud_change(self, points_3d: np.ndarray) -> Tuple[bool, bool]:
        """Return (significant_change, needs_refresh_only).

        significant_change → mesh must be regenerated.
        needs_refresh_only → cached mesh can stay but viewer should refresh.
        """

        # Metric A: mean distance on sampled points
        significant = self._distance_based_change(points_3d)

        # Metric B: voxel hash Jaccard distance
        voxel_changed = self._voxel_hash_change(points_3d)

        # Metric C: point count ratio
        count_changed = self._count_change(points_3d)

        # Decision
        if significant or voxel_changed >= 0.05 or count_changed >= 0.10:
            return True, False  # force regenerate

        # If voxel hash changed moderately (>0.02) mark viewer refresh needed
        needs_refresh = voxel_changed >= 0.02 or count_changed >= 0.05
        return False, needs_refresh

    def _distance_based_change(self, points_3d: np.ndarray) -> bool:
        """Simple mean distance sample heuristic."""
        sample_size = min(self._SAMPLE_SIZE, len(points_3d))

        if self._sample_idx is None or len(self._sample_idx) != sample_size:
            # (re)initialise fixed sample indices for current point cloud size
            self._sample_idx = self._rng.choice(points_3d.shape[0], size=sample_size, replace=False)

        sample = points_3d[self._sample_idx]

        if self._prev_sample is None:
            self._prev_sample = sample.copy()
            return True

        # Use median distance which is more robust to outliers
        median_dist = float(np.median(np.linalg.norm(sample - self._prev_sample, axis=1)))
        self._prev_sample = sample.copy()
        return median_dist > self._DIFF_THRESHOLD_M

    def _voxel_hash_change(self, points_3d: np.ndarray) -> float:
        try:
            import numpy as _np
        except ImportError:
            return 0.0

        vsz = 0.03
        vox = _np.floor(points_3d / vsz).astype(_np.int32)
        vox_set = { (int(x), int(y), int(z)) for x, y, z in vox }

        if self._prev_vox_set is None:
            self._prev_vox_set = vox_set
            return 1.0  # treat as full change first time

        inter = len(vox_set & self._prev_vox_set)
        union = len(vox_set | self._prev_vox_set)
        jaccard_dist = 1.0 - (inter / union) if union else 0.0

        # EWMA smoothing
        smoothed = self._ewma_voxel.update(jaccard_dist)

        self._prev_vox_set = vox_set
        return smoothed

    def _count_change(self, points_3d: np.ndarray) -> float:
        curr = len(points_3d)
        if self._prev_count is None or self._prev_count == 0:
            self._prev_count = curr
            return 1.0

        diff_ratio = abs(curr - self._prev_count) / self._prev_count
        self._prev_count = curr

        return self._ewma_count.update(diff_ratio)


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