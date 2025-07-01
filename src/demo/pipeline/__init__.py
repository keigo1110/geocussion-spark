#!/usr/bin/env python3
"""
パイプラインパッケージ

ステージベースのパイプライン処理システムを提供します。
"""

# 基底クラス
from .base import PipelineStage, StageResult

# 各ステージ
from .input_stage import InputStage, InputStageConfig, InputStageResult
from .detection_stage import DetectionStage, DetectionStageConfig, DetectionStageResult
from .mesh_stage import MeshStage, MeshStageConfig, MeshStageResult
from .collision_stage import CollisionStage, CollisionStageConfig, CollisionStageResult
from .audio_stage import AudioStage, AudioStageConfig, AudioStageResult

# オーケストレーター
from .orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResults


__all__ = [
    # 基底クラス
    'PipelineStage',
    'StageResult',
    # 入力ステージ
    'InputStage',
    'InputStageConfig',
    'InputStageResult',
    # 検出ステージ
    'DetectionStage',
    'DetectionStageConfig',
    'DetectionStageResult',
    # メッシュステージ
    'MeshStage',
    'MeshStageConfig',
    'MeshStageResult',
    # 衝突検出ステージ
    'CollisionStage',
    'CollisionStageConfig',
    'CollisionStageResult',
    # 音響ステージ
    'AudioStage',
    'AudioStageConfig',
    'AudioStageResult',
    # オーケストレーター
    'PipelineOrchestrator',
    'PipelineConfig',
    'PipelineResults',
]