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