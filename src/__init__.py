#!/usr/bin/env python3
"""
Geocussion-SP メインパッケージ

プロジェクト全体で使用される共通設定とロギング機能を提供します。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# プロジェクト情報
__version__ = "0.1.0"
__author__ = "Geocussion Development Team"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_style: str = "detailed"
) -> logging.Logger:
    """
    プロジェクト全体の統一ログ設定
    
    Args:
        level: ログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: ログファイルパス（Noneならコンソールのみ）
        format_style: フォーマットスタイル ("simple", "detailed", "debug")
    
    Returns:
        設定済みルートロガー
    """
    # ログレベル設定
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # フォーマット選択
    formats = {
        "simple": "%(levelname)s: %(message)s",
        "detailed": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "debug": "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"
    }
    
    log_format = formats.get(format_style, formats["detailed"])
    formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
    
    # ルートロガー設定
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 既存ハンドラークリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # ファイルハンドラー（オプション）
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    モジュール用ロガーを取得
    
    Args:
        name: ロガー名（通常は __name__ を使用）
    
    Returns:
        設定済みロガー
    """
    return logging.getLogger(name)


# デフォルト設定の初期化
_default_logger_initialized = False

def ensure_default_logging() -> None:
    """デフォルトロギングが初期化されていることを確認"""
    global _default_logger_initialized
    if not _default_logger_initialized:
        setup_logging(level="INFO", format_style="detailed")
        _default_logger_initialized = True


# モジュールロード時に最低限の設定を適用
ensure_default_logging()

# -----------------------------------------------------------------------------
# OpenCV compatibility shim (global)
# -----------------------------------------------------------------------------
try:
    import cv2  # type: ignore
    import types as _types

    # cv2.error may be missing in some stripped wheels; provide placeholder
    if not hasattr(cv2, "error"):
        class _CV2Error(Exception):
            """Fallback OpenCV error class when cv2.error is unavailable."""
            pass
        cv2.error = _CV2Error  # type: ignore

    # Provide minimal stub for cv2.cuda when OpenCV built without CUDA
    if not hasattr(cv2, "cuda"):
        _cuda_stub = _types.SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0)
        cv2.cuda = _cuda_stub  # type: ignore

    # Ensure cv2.Tracker is present (legacy tracking API)
    if not hasattr(cv2, "Tracker"):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "Tracker"):
            cv2.Tracker = cv2.legacy.Tracker  # type: ignore
        else:
            class _DummyTracker:  # type: ignore
                pass
            cv2.Tracker = _DummyTracker  # type: ignore

    # Provide simple CPU fallbacks for common image ops if they are missing
    import numpy as _np
    try:
        import scipy.ndimage as _ndi
    except ImportError:
        _ndi = None

    if not hasattr(cv2, "medianBlur") and _ndi is not None:
        def _medianBlur(img, ksize):
            return _ndi.median_filter(img, size=(ksize, ksize))
        cv2.medianBlur = _medianBlur  # type: ignore

    if not hasattr(cv2, "GaussianBlur") and _ndi is not None:
        def _gaussianBlur(img, ksize, sigma):
            if isinstance(ksize, tuple):
                k = max(ksize)
            else:
                k = ksize
            return _ndi.gaussian_filter(img, sigma=sigma or (k/6.0))
        cv2.GaussianBlur = _gaussianBlur  # type: ignore

    if not hasattr(cv2, "bilateralFilter") and _ndi is not None:
        def _bilateralFilter(img, d, sigma_color, sigma_space):
            # Approximate by gaussian blur when true bilateral unavailable
            return _ndi.gaussian_filter(img, sigma=sigma_space/3.0)
        cv2.bilateralFilter = _bilateralFilter  # type: ignore

    if not hasattr(cv2, "destroyAllWindows"):
        cv2.destroyAllWindows = lambda: None  # type: ignore

    if not hasattr(cv2, "inpaint"):
        import numpy as _np

        def _inpaint_stub(src, inpaintMask, inpaintRadius, flags):  # type: ignore
            """Very simple CPU fallback for cv2.inpaint.

            This implementation does **not** attempt to replicate the exact
            behaviour of OpenCV's Telea/N-S inpainting algorithms, but provides
            a reasonable approximation that prevents NaN propagation in the
            downstream depth processing pipeline.

            Strategy:
              1. Copy the source image.
              2. For each masked pixel, assign the value of the nearest valid
                 neighbour using a distance transform.  SciPy is used when
                 available for performance; otherwise a mean-value fallback is
                 applied.
            """

            dst = src.copy()

            # Ensure inputs are np.ndarray
            src_arr = _np.asarray(src)
            mask_arr = _np.asarray(inpaintMask).astype(bool)

            if not mask_arr.any():
                return dst

            try:
                # Prefer SciPy's EDT for efficiency if available
                import scipy.ndimage as _ndi  # pylint: disable=import-error

                # distance_transform_edt returns indices of nearest non-masked
                # point for each masked location (when return_indices=True)
                _, indices = _ndi.distance_transform_edt(mask_arr,
                                                         return_distances=True,
                                                         return_indices=True)
                dst[mask_arr] = src_arr[tuple(indices[:, mask_arr])]
            except Exception:  # noqa: BLE001 — fall back gracefully
                # Fallback: fill with global median of valid pixels
                valid_vals = src_arr[~mask_arr]
                fill_val = int(_np.median(valid_vals)) if valid_vals.size else 0
                dst[mask_arr] = fill_val

            return dst

        cv2.inpaint = _inpaint_stub  # type: ignore

    # Constants for inpaint algorithms (may be missing in stub builds)
    if not hasattr(cv2, "INPAINT_NS"):
        cv2.INPAINT_NS = 0  # type: ignore
    if not hasattr(cv2, "INPAINT_TELEA"):
        cv2.INPAINT_TELEA = 1  # type: ignore

except ImportError:
    pass  # cv2 not installed; other modules will handle their own fallbacks 