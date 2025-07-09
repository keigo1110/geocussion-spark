from __future__ import annotations

"""TerrainChangeDetector

Detect significant changes in sand terrain from depth images or point clouds.
Used to pause audio during large terrain modification.

Simple image-diff + low-pass approach; avoids heavy compute.
"""

from typing import Optional
import numpy as np
import time


class TerrainChangeDetector:
    """Detect large-scale terrain modifications.

    Parameters
    ----------
    diff_threshold_mm:
        Per-pixel height difference [mm] considered as change.
    change_ratio:
        Fraction of pixels exceeding threshold to regard as *large* change.
    cooldown_sec:
        Minimum duration to keep *deforming* state once triggered.
    """

    def __init__(
        self,
        *,
        diff_threshold_mm: float = 20.0,
        change_ratio: float = 0.05,
        cooldown_sec: float = 1.0,
    ) -> None:
        self._th_mm = diff_threshold_mm
        self._ratio = change_ratio
        self._cooldown = cooldown_sec
        self._prev_depth: Optional[np.ndarray] = None
        self._last_change_ts: float = 0.0
        self._deforming: bool = False

    # ------------------------------------------------------------------
    def update(self, depth_mm: np.ndarray, *, hands_present: bool) -> bool:  # noqa: D401
        """Process new depth frame; return *True* if terrain is currently deforming."""
        if depth_mm is None:
            return self._deforming

        # Work in int16 mm
        cur = depth_mm.astype(np.int32)

        if self._prev_depth is not None and cur.shape == self._prev_depth.shape:
            diff = np.abs(cur - self._prev_depth)
            changed_frac = float(np.count_nonzero(diff > self._th_mm)) / diff.size
            large_change = changed_frac >= self._ratio
        else:
            large_change = False

        now = time.perf_counter()

        # Trigger conditions -------------------------------------------------
        if large_change:
            self._deforming = True
            self._last_change_ts = now
        elif self._deforming:
            # Maintain deforming for cooldown or if hands still present
            if (now - self._last_change_ts) > self._cooldown and not hands_present:
                self._deforming = False

        # Update stored depth every frame (cheap)
        self._prev_depth = cur.copy()
        return self._deforming

    # ------------------------------------------------------------------
    @property
    def is_deforming(self) -> bool:  # noqa: D401 – simple
        return self._deforming 