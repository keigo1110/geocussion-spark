from __future__ import annotations

"""PointCloudProducer

Background thread that continuously converts incoming depth frames to point
clouds and publishes the latest result via an internal lock-free slot.

T-MESH-200 – step 1
"""

from threading import Thread, Event, Lock
from time import perf_counter, sleep
from typing import Callable, Optional, Tuple
import numpy as np


class PointCloudProducer(Thread):
    """Continuously generate point clouds from depth frames.

    Parameters
    ----------
    fetch_depth
        Callable that returns ``np.ndarray`` depth image or *None* when no new
        frame is available.
    to_pointcloud
        Callable that converts ``depth_image`` → ``np.ndarray`` (N×3) point
        cloud in metres.  Must be *thread-safe*.
    interval_s
        Desired polling interval in seconds (default 0.03 ≈ 33 Hz).
    """

    def __init__(
        self,
        fetch_depth: Callable[[], Optional[np.ndarray]],
        to_pointcloud: Callable[[np.ndarray], Tuple[np.ndarray, object]],
        *,
        interval_s: float = 0.03,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._fetch_depth = fetch_depth
        self._to_pointcloud = to_pointcloud
        self._interval = interval_s

        self._stop_event = Event()
        self._lock = Lock()
        self._latest_points: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0

    # ------------------------------------------------------------------
    # Thread API
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 – simple verb OK
        while not self._stop_event.is_set():
            start = perf_counter()
            depth = self._fetch_depth()
            if depth is not None:
                try:
                    points, _ = self._to_pointcloud(depth)
                    with self._lock:
                        self._latest_points = points
                        self._latest_ts = perf_counter()
                except Exception:  # pylint: disable=broad-except
                    # Silently ignore conversion errors to keep thread alive
                    pass
            elapsed = perf_counter() - start
            sleep(max(0.0, self._interval - elapsed))

    def stop(self, *, timeout: float = 1.0) -> None:
        """Signal the thread to stop and join."""
        self._stop_event.set()
        self.join(timeout)

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------
    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        """Return latest point cloud and timestamp (seconds)."""
        with self._lock:
            return self._latest_points, self._latest_ts

    def get_latest_points(self) -> Optional[np.ndarray]:
        """Return latest point cloud."""
        with self._lock:
            return self._latest_points

    def get_latest_timestamp(self) -> float:
        """Return latest timestamp (seconds)."""
        with self._lock:
            return self._latest_ts

    def is_running(self) -> bool:
        """Return whether the thread is running."""
        return self.is_alive()

    def is_stopped(self) -> bool:
        """Return whether the thread is stopped."""
        return not self.is_alive()

    def start(self) -> None:
        """Start the thread."""
        super().start()

    def join(self, timeout: float = None) -> None:
        """Wait for the thread to complete execution."""
        super().join(timeout)

    def stop_and_join(self, timeout: float = 1.0) -> None:
        """Signal the thread to stop and wait for it to complete execution."""
        self.stop(timeout=timeout)
        self.join(timeout)

    def get_interval(self) -> float:
        """Return the polling interval in seconds."""
        return self._interval

    def set_interval(self, interval_s: float) -> None:
        """Set the polling interval in seconds."""
        self._interval = interval_s

    def get_fetch_depth(self) -> Callable[[], Optional[np.ndarray]]:
        """Return the fetch depth callable."""
        return self._fetch_depth

    def get_to_pointcloud(self) -> Callable[[np.ndarray], Tuple[np.ndarray, object]]:
        """Return the to point cloud callable."""
        return self._to_pointcloud

    def get_latest_point_cloud(self) -> Optional[np.ndarray]:
        """Return the latest point cloud."""
        return self.get_latest_points()

    def get_latest_point_cloud_timestamp(self) -> float:
        """Return the latest point cloud timestamp (seconds)."""
        return self.get_latest_timestamp()

    def get_point_cloud_producer_status(self) -> dict:
        """Return the status of the point cloud producer."""
        return {
            "is_running": self.is_running(),
            "is_stopped": self.is_stopped(),
            "latest_point_cloud": self.get_latest_point_cloud(),
            "latest_point_cloud_timestamp": self.get_latest_point_cloud_timestamp(),
            "interval": self.get_interval(),
            "fetch_depth": self.get_fetch_depth(),
            "to_pointcloud": self.get_to_pointcloud(),
        } 