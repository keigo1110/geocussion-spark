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

    def __init__(self, mesh_pipeline: MeshPipeline, *, min_interval_sec: float = 1.0) -> None:
        self._pipe = mesh_pipeline
        self._cached_mesh: Optional[MeshResult] = None
        self._mesh_version: int = 0
        self._last_update_ts: float = 0.0
        # Track whether hands were present in previous frame (P-HAND-003)
        self._prev_hands_present: bool = False
        self._min_interval = max(0.2, min_interval_sec)

    # ------------------------------------------------------------------
    def update_if_needed(
        self,
        points: Optional[np.ndarray],
        hands: List[TrackedHand],
        *,
        hands_present_override: Optional[bool] = None,
        force: bool = False,
    ) -> MeshResult:
        """Return up-to-date mesh (may be cached)."""
        now = time.perf_counter()
        # ---------------- Hand presence gating (P-HAND-003) ---------------
        hands_present = hands_present_override if hands_present_override is not None else bool(hands)

        if hands_present:
            # If any hand exists, ALWAYS postpone heavy regenerate to keep FPS high
            # Never generate mesh while hands visible to avoid hand geometry
            self._prev_hands_present = True
            return self._cached_mesh if self._cached_mesh is not None else MeshResult(None)

        # If hands were present in previous frame, reset flag but do NOT force
        # regeneration here – defer to external MeshUpdateScheduler which applies
        # a configurable grace period (e.g. 1 s) before allowing heavy updates.
        # This guarantees that mesh updates (including user-triggered ones) are
        # fully suppressed while hands are visible or terrain is still moving.
        if self._prev_hands_present:
            self._prev_hands_present = False

        # Strict gating – only regenerate when explicit *force* flag is True.
        # This flag is provided by MeshUpdateScheduler after grace-period logic,
        # ensuring mesh updates never occur while hands are visible or terrain
        # is still deforming.
        if not force:
            return self._cached_mesh if self._cached_mesh is not None else MeshResult(None)

        if points is None or len(points) < 100:
            return self._cached_mesh if self._cached_mesh else MeshResult(None)

        res = self._pipe.generate_mesh(points, hands, force_update=True)
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