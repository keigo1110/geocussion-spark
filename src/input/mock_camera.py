#!/usr/bin/env python3
"""Mock camera implementation used in headless mode or CI.

This class mimics the minimal subset of the `OrbbecCamera` interface
required by `FullPipelineViewer` so that type-checkers (mypy) and
runtime both remain satisfied when real hardware is absent.
"""

from __future__ import annotations

import time
import numpy as np
from types import SimpleNamespace
from typing import Any, Optional

from src.types import CameraIntrinsics, FrameData, OBFormat

__all__ = ["MockCamera"]


class MockCamera:
    """Very small drop-in replacement for `OrbbecCamera`."""

    def __init__(self, width: int = 424, height: int = 240, fx: float = 364.0, fy: float = 364.0):
        self.depth_intrinsics: CameraIntrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )
        self.has_color: bool = True
        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # Public API expected by FullPipelineViewer / DualViewer
    # ------------------------------------------------------------------
    def get_frame(self, timeout_ms: int = 100) -> FrameData:  # type: ignore[override]
        """Return a synthetic depth+color frame bundle.

        The object mimics the attributes accessed by the viewer code and is
        wrapped into the project-wide ``FrameData`` dataclass for maximum
        compatibility with downstream pipelines.
        """

        self._frame_counter += 1

        depth_pixels = self.depth_intrinsics.width * self.depth_intrinsics.height
        depth_data = np.random.randint(500, 2000, depth_pixels, dtype=np.uint16)

        depth_frame = SimpleNamespace(get_data=lambda: depth_data.tobytes())

        # Produce RGB frame bytes (720×1280×3) – resolution not critical for tests
        color_bytes = np.random.randint(0, 255, 720 * 1280 * 3, dtype=np.uint8)
        color_frame = SimpleNamespace(
            get_data=lambda: color_bytes.tobytes(),
            get_format=lambda: OBFormat.RGB,
        )

        # Timestamp in milliseconds for compatibility with real SDK
        ts_ms = time.perf_counter() * 1000.0

        return FrameData(
            depth_frame=depth_frame,
            color_frame=color_frame,
            timestamp_ms=ts_ms,
            frame_number=self._frame_counter,
            points=None,
        )

    # The following dummies satisfy `ManagedResource`-style clean-up calls
    def start(self) -> bool:  # noqa: D401
        return True

    def stop(self) -> None:  # noqa: D401
        return None

    def cleanup(self) -> bool:  # noqa: D401
        return True

    # Context manager helpers -------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False 