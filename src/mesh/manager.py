from __future__ import annotations

"""PipelineManager

Owns cached mesh & diff logic.  Decides when to regenerate mesh via
MeshPipeline.  Acts as Facade to the viewer.
T-MESH-200 – step 2
"""

from typing import Optional, List
import time
import numpy as np

from .pipeline import MeshPipeline, MeshResult
from ..detection.tracker import TrackedHand


class PipelineManager:
    """Manage mesh cache and trigger regeneration when dirty."""

    def __init__(self, mesh_pipeline: MeshPipeline) -> None:
        self._pipe = mesh_pipeline
        self._cached_mesh: Optional[MeshResult] = None
        self._mesh_version: int = 0
        self._last_update_ts: float = 0.0

    # ------------------------------------------------------------------
    def update_if_needed(
        self,
        points: Optional[np.ndarray],
        hands: List[TrackedHand],
        *,
        force: bool = False,
    ) -> MeshResult:
        """Return up-to-date mesh (may be cached)."""
        now = time.perf_counter()
        # Debounce heavy regenerate (<0.2 s)
        if not force and self._cached_mesh and (now - self._last_update_ts) < 0.2:
            return self._cached_mesh

        if points is None or len(points) < 100:
            return self._cached_mesh if self._cached_mesh else MeshResult(None)

        res = self._pipe.generate_mesh(points, hands, force_update=force)
        if res.changed:
            self._mesh_version += 1
            self._cached_mesh = res
            self._last_update_ts = now
        elif res.needs_refresh:
            # update only version to notify viewer
            self._cached_mesh = res
            self._mesh_version += 1
        else:
            self._cached_mesh = res

        # If mesh not changed but generator might have async future, call again quickly
        if res.mesh is None and getattr(self._pipe, "_lod", None) is not None:
            # check if async completed
            async_mesh = getattr(self._pipe._lod, "_pending_future", None)  # type: ignore[attr-defined]
            if async_mesh is not None and async_mesh.done():
                try:
                    mesh_val = async_mesh.result()
                    self._pipe._lod._pending_future = None  # type: ignore[attr-defined]
                    res = MeshResult(mesh_val, changed=True, needs_refresh=True)
                except Exception:  # pragma: no cover
                    pass

        return res

    # ------------------------------------------------------------------
    def get_version(self) -> int:
        return self._mesh_version 