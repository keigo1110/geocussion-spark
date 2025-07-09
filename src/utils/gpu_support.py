#!/usr/bin/env python3
"""GPU サポートユーティリティ

CuPy など GPU ライブラリの有無を一度だけ判定し、
各モジュールが共通のログフォーマットで利用状況を報告できるようにします。
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Final

logger = logging.getLogger(__name__)

GPU_AVAILABLE_CACHE: Final[str] = "_gpu_available_cached"

@lru_cache(maxsize=1)
def is_gpu_available() -> bool:
    """CuPy がインポート可能かを判定（結果をキャッシュ）。"""
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def log_gpu_status(component: str, enabled: bool, *, logger_obj: logging.Logger | None = None) -> None:
    """共通フォーマットで GPU 使用可否をログ出力。

    Parameters
    ----------
    component : str
        ログ出力対象となるコンポーネント名。
    enabled : bool
        GPU が有効であれば True。
    logger_obj : logging.Logger | None
        出力先ロガー。省略時は module logger を使用。
    """
    _logger = logger_obj or logger
    if enabled:
        _logger.info("🚀 %s: GPU acceleration enabled", component)
    else:
        _logger.warning("⚠️  %s: GPU unavailable – falling back to CPU", component) 