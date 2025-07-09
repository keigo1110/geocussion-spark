from __future__ import annotations

"""MeshUpdateScheduler

Decide when to trigger heavy terrain mesh regeneration for sand-pit use-case.

FR-1   通常時 (手無し) は 1 Hz 程度で更新
FR-2   手検出中は更新しない
FR-3   手 → 無し 遷移で即更新

Lightweight utility – no external deps except stdlib.
"""

import time


class MeshUpdateScheduler:
    """Frame-by-frame mesh update decision helper.

    Usage::
        scheduler = MeshUpdateScheduler(base_interval_sec=1.0)
        should_force = scheduler.should_update(hands_present)
    """

    def __init__(self, base_interval_sec: float = 1.0, grace_period_sec: float = 1.0) -> None:
        self._base_interval = max(0.1, base_interval_sec)
        self._grace = grace_period_sec
        self._last_update_ts: float = 0.0
        self._last_hand_seen_ts: float = 0.0
        self._prev_hands_present: bool = False

    # ------------------------------------------------------------------
    def mark_updated(self) -> None:
        """Inform scheduler that a mesh update *did* happen now."""
        self._last_update_ts = time.perf_counter()

    # ------------------------------------------------------------------
    def should_update(self, hands_present: bool) -> bool:
        """Return ``True`` when viewer should regenerate mesh.

        Algorithm:
        1. If hands *now* present → never update (save perf).
        2. If hands disappeared since last frame → update once immediately.
        3. Otherwise update every ``base_interval`` seconds.
        """
        now = time.perf_counter()

        # Case 1 – hands visible: no update, just remember state.
        if hands_present:
            self._prev_hands_present = True
            self._last_hand_seen_ts = now
            return False

        # Consider grace period where hands were recently present
        if (now - self._last_hand_seen_ts) < self._grace:
            return False

        # Hands NOT present below --------------------------------------
        # If previously there *were* hands, trigger one-shot update.
        if self._prev_hands_present:
            self._prev_hands_present = False
            self._last_update_ts = now  # avoid double-trigger in same frame
            return True

        # Periodic update every base_interval
        if (now - self._last_update_ts) >= self._base_interval:
            self._last_update_ts = now
            return True

        # Otherwise no need.
        return False

    # ------------------------------------------------------------------
    @property
    def base_interval(self) -> float:  # noqa: D401 – simple
        """Return configured interval seconds."""
        return self._base_interval 