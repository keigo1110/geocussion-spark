"""HandMasker

Utility to blank depth pixels corresponding to detected hands and
provide adaptive 3-D exclusion spheres for point-cloud generation.
P-HAND-002 implementation.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
import numpy as np
import cv2

from src.data_types import HandDetectionResult, Hand3D, TrackingState

__all__ = ["HandMasker"]


class HandMasker:  # pylint: disable=too-few-public-methods
    """Generate masks for hand regions in depth image and exclusion spheres.

    Parameters
    ----------
    inflate_bbox : float, default 1.3
        Scale factor to inflate 2-D bounding boxes before blanking.
    base_radius_m : float, default 0.03
        Minimum radius in metres for 3-D exclusion spheres.
    vel_scale : float, default 0.5
        Additional metres of radius per 1 m/s of hand velocity.
    max_radius_m : float, default 0.06
        Maximum adaptive radius.
    """

    def __init__(
        self,
        inflate_bbox: float = 1.3,
        base_radius_m: float = 0.03,
        vel_scale: float = 0.5,
        max_radius_m: float = 0.06,
    ) -> None:
        self.inflate = inflate_bbox
        self.base_r = base_radius_m
        self.vel_scale = vel_scale
        self.max_r = max_radius_m

    # ------------------------------------------------------------------
    def apply_mask(
        self,
        depth_image: np.ndarray,
        hands_2d: Sequence[HandDetectionResult],
        tracked_hands: Sequence[Hand3D],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blank hand pixels in *depth_image* and return exclusion spheres.

        Returns
        -------
        masked_depth : np.ndarray
            Copy of depth image with hand regions zeroed.
        centers_3d : np.ndarray (N, 3)
            Centre positions for exclusion spheres (may be empty).
        radii : np.ndarray (N,)
            Radius per centre.
        """
        masked = depth_image.copy()

        # --- Handle resolution mismatch between RGB (detection) and depth ---
        h_img, w_img = depth_image.shape[:2]

        # Estimate original resolution from bounding boxes (assume same for all)
        if hands_2d:
            max_x_bb = max((bx + bw) for bx, _, bw, _ in (h.bounding_box for h in hands_2d))
            max_y_bb = max((by + bh) for _, by, _, bh in (h.bounding_box for h in hands_2d))
            src_w = max(max_x_bb, w_img)
            src_h = max(max_y_bb, h_img)
        else:
            src_w, src_h = w_img, h_img

        scale_x = w_img / src_w if src_w > 0 else 1.0
        scale_y = h_img / src_h if src_h > 0 else 1.0

        for hand in hands_2d:
            x, y, w, h = hand.bounding_box
            # Scale bbox to depth resolution
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y

            cx = x + w / 2.0
            cy = y + h / 2.0
            w *= self.inflate
            h *= self.inflate
            x0 = int(max(cx - w / 2.0, 0))
            y0 = int(max(cy - h / 2.0, 0))
            x1 = int(min(cx + w / 2.0, w_img - 1))
            y1 = int(min(cy + h / 2.0, h_img - 1))
            masked[y0:y1, x0:x1] = 0

        # --- 3-D exclusion spheres ------------------------------------
        centers = []
        radii = []
        for hand in tracked_hands:
            if hand.position is None:
                continue
            # velocity magnitude (if available)
            vel = getattr(hand, "velocity", np.zeros(3))
            vel_mag = float(np.linalg.norm(vel))
            radius = min(self.max_r, self.base_r + self.vel_scale * vel_mag)
            centers.append(np.asarray(hand.position, dtype=np.float32))
            radii.append(radius)

        if centers:
            centers_arr = np.vstack(centers)
            radii_arr = np.asarray(radii, dtype=np.float32)
        else:
            centers_arr = np.empty((0, 3), dtype=np.float32)
            radii_arr = np.empty((0,), dtype=np.float32)

        return masked, centers_arr, radii_arr 