#!/usr/bin/env python3
"""
Geocussion-SP 統一デモシステム

全デモスクリプトの共通機能を提供し、コード重複を排除します。
Clean Architectureパターンに基づく設計で保守性を向上させています。
"""

# 設定関連のインポート
from .config import (
    DemoMode,
    DemoConfiguration,
    create_common_argument_parser,
    parse_arguments_to_config,
    run_test_mode
)

# ランナーのインポート
from .runner import DemoRunner

# 統合ビューワーのインポート  
from .integrated_viewer import IntegratedGeocussionViewer
from .pipeline_wrapper import HandledPipeline, HandledPipelineConfig, PipelineResults

__all__ = [
    # 設定
    'DemoMode',
    'DemoConfiguration',
    'create_common_argument_parser',
    'parse_arguments_to_config',
    'run_test_mode',
    # ランナー
    'DemoRunner',
    # 統合システム
    'IntegratedGeocussionViewer',
    'HandledPipeline',
    'HandledPipelineConfig',
    'PipelineResults',
]