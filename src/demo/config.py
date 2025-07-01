#!/usr/bin/env python3
"""
Geocussion-SP デモ設定モジュール

デモシステムの設定、引数解析、テストモード実行を管理します。
"""

from typing import Dict, Any, Optional, List
import argparse
from dataclasses import dataclass
from enum import Enum

# 共通インポート
from ..debug.pipeline_controller import PipelineConfiguration
from ..sound.mapping import ScaleType, InstrumentType
from ..input.depth_filter import FilterType
from .. import get_logger

logger = get_logger(__name__)


class DemoMode(Enum):
    """デモモードの定義"""
    BASIC_VIEWER = "basic"           # 基本ビューワー（RGB+深度+点群）
    HAND_DETECTION = "hands"         # 手検出機能付き
    COLLISION_DETECTION = "collision" # 衝突検出機能付き（フル機能）
    CLEAN_ARCHITECTURE = "clean"     # Clean Architectureデモ


@dataclass
class DemoConfiguration:
    """統一デモ設定"""
    # 実行モード
    mode: DemoMode = DemoMode.BASIC_VIEWER
    test_mode: bool = False
    
    # ログ設定
    log_level: str = "INFO"
    
    # 入力系設定
    enable_filter: bool = True
    filter_type: FilterType = FilterType.COMBINED
    enable_hand_detection: bool = False
    enable_tracking: bool = False
    min_detection_confidence: float = 0.1
    use_gpu_mediapipe: bool = False
    
    # メッシュ生成設定
    enable_mesh_generation: bool = False
    mesh_update_interval: int = 5
    max_mesh_skip_frames: int = 60
    mesh_resolution: float = 0.01
    mesh_quality: float = 0.3
    mesh_reduction: float = 0.7
    
    # 衝突検出設定
    enable_collision_detection: bool = False
    enable_collision_visualization: bool = False
    sphere_radius: float = 0.05
    
    # 音響合成設定
    enable_audio_synthesis: bool = False
    audio_scale: ScaleType = ScaleType.PENTATONIC
    audio_instrument: InstrumentType = InstrumentType.MARIMBA
    audio_polyphony: int = 16
    audio_master_volume: float = 0.7
    
    # UI設定
    window_width: int = 640
    window_height: int = 480
    update_interval: int = 3
    point_size: float = 2.0
    
    def to_pipeline_config(self) -> PipelineConfiguration:
        """PipelineConfigurationに変換"""
        return PipelineConfiguration(
            # 入力設定
            enable_filter=self.enable_filter,
            enable_hand_detection=self.enable_hand_detection,
            enable_tracking=self.enable_tracking,
            min_detection_confidence=self.min_detection_confidence,
            use_gpu_mediapipe=self.use_gpu_mediapipe,
            
            # メッシュ設定
            enable_mesh_generation=self.enable_mesh_generation,
            mesh_update_interval=self.mesh_update_interval,
            max_mesh_skip_frames=self.max_mesh_skip_frames,
            
            # 衝突設定
            enable_collision_detection=self.enable_collision_detection,
            enable_collision_visualization=self.enable_collision_visualization,
            sphere_radius=self.sphere_radius,
            
            # 音響設定
            enable_audio_synthesis=self.enable_audio_synthesis,
            audio_scale=self.audio_scale,
            audio_instrument=self.audio_instrument,
            audio_polyphony=self.audio_polyphony,
            audio_master_volume=self.audio_master_volume
        )


def create_common_argument_parser() -> argparse.ArgumentParser:
    """共通引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        description="Geocussion-SP 統一デモシステム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 実行モード
    parser.add_argument('--mode', 
                       choices=[mode.value for mode in DemoMode],
                       default=DemoMode.BASIC_VIEWER.value,
                       help='実行モード')
    parser.add_argument('--test', action='store_true',
                       help='テストモードで実行')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='ログレベル')
    
    # 入力系オプション
    input_group = parser.add_argument_group('入力系オプション')
    input_group.add_argument('--no-filter', action='store_true',
                           help='深度フィルタを無効化')
    input_group.add_argument('--filter-type',
                           choices=['median', 'bilateral', 'temporal', 'combined'],
                           default='combined',
                           help='フィルタタイプ')
    input_group.add_argument('--no-hand-detection', action='store_true',
                           help='手検出を無効化')
    input_group.add_argument('--no-tracking', action='store_true',
                           help='手トラッキングを無効化')
    input_group.add_argument('--detection-confidence', type=float, default=0.1,
                           help='手検出信頼度閾値')
    input_group.add_argument('--gpu', action='store_true',
                           help='MediaPipeでGPUを使用')
    
    # メッシュ系オプション
    mesh_group = parser.add_argument_group('メッシュ生成オプション')
    mesh_group.add_argument('--no-mesh', action='store_true',
                          help='メッシュ生成を無効化')
    mesh_group.add_argument('--mesh-interval', type=int, default=5,
                          help='メッシュ更新間隔（フレーム）')
    mesh_group.add_argument('--max-mesh-skip', type=int, default=60,
                          help='最大メッシュスキップフレーム数')
    mesh_group.add_argument('--mesh-resolution', type=float, default=0.01,
                          help='メッシュ解像度（メートル）')
    mesh_group.add_argument('--mesh-quality', type=float, default=0.3,
                          help='メッシュ品質閾値')
    mesh_group.add_argument('--mesh-reduction', type=float, default=0.7,
                          help='メッシュ削減率')
    
    # 衝突検出オプション
    collision_group = parser.add_argument_group('衝突検出オプション')
    collision_group.add_argument('--no-collision', action='store_true',
                               help='衝突検出を無効化')
    collision_group.add_argument('--no-collision-viz', action='store_true',
                               help='衝突可視化を無効化')
    collision_group.add_argument('--sphere-radius', type=float, default=0.05,
                               help='衝突球半径（メートル）')
    
    # 音響合成オプション
    audio_group = parser.add_argument_group('音響合成オプション')
    audio_group.add_argument('--no-audio', action='store_true',
                           help='音響合成を無効化')
    audio_group.add_argument('--audio-scale',
                           choices=['chromatic', 'major', 'minor', 'pentatonic', 'blues'],
                           default='pentatonic',
                           help='音階')
    audio_group.add_argument('--audio-instrument',
                           choices=['marimba', 'bell', 'crystal', 'drum', 'water_drop', 'wind', 'string', 'synth_pad'],
                           default='marimba',
                           help='楽器')
    audio_group.add_argument('--audio-polyphony', type=int, default=16,
                           help='ポリフォニー数')
    audio_group.add_argument('--audio-volume', type=float, default=0.7,
                           help='音量（0.0-1.0）')
    
    # UI設定オプション
    ui_group = parser.add_argument_group('UI設定オプション')
    ui_group.add_argument('--window-width', type=int, default=640,
                        help='ウィンドウ幅')
    ui_group.add_argument('--window-height', type=int, default=480,
                        help='ウィンドウ高さ')
    ui_group.add_argument('--update-interval', type=int, default=3,
                        help='点群更新間隔（フレーム）')
    ui_group.add_argument('--point-size', type=float, default=2.0,
                        help='点群のサイズ')
    
    return parser


def parse_arguments_to_config(args: argparse.Namespace) -> DemoConfiguration:
    """引数をDemoConfigurationに変換"""
    # DemoModeの変換
    mode = DemoMode(args.mode)
    
    # フィルタタイプの変換
    filter_type_map = {
        'median': FilterType.MEDIAN,
        'bilateral': FilterType.BILATERAL,
        'temporal': FilterType.TEMPORAL,
        'combined': FilterType.COMBINED
    }
    filter_type = filter_type_map[args.filter_type]
    
    # ScaleTypeの変換
    scale_type_map = {
        'chromatic': ScaleType.CHROMATIC,
        'major': ScaleType.MAJOR,
        'minor': ScaleType.MINOR,
        'pentatonic': ScaleType.PENTATONIC,
        'blues': ScaleType.BLUES
    }
    audio_scale = scale_type_map[args.audio_scale]
    
    # InstrumentTypeの変換
    instrument_type_map = {
        'marimba': InstrumentType.MARIMBA,
        'bell': InstrumentType.BELL,
        'crystal': InstrumentType.CRYSTAL,
        'drum': InstrumentType.DRUM,
        'water_drop': InstrumentType.WATER_DROP,
        'wind': InstrumentType.WIND,
        'string': InstrumentType.STRING,
        'synth_pad': InstrumentType.SYNTH_PAD
    }
    audio_instrument = instrument_type_map[args.audio_instrument]
    
    # モード別デフォルト設定
    enable_hand_detection = not args.no_hand_detection
    enable_mesh_generation = not args.no_mesh
    enable_collision_detection = not args.no_collision
    enable_audio_synthesis = not args.no_audio
    
    if mode == DemoMode.BASIC_VIEWER:
        enable_hand_detection = False
        enable_mesh_generation = False
        enable_collision_detection = False
        enable_audio_synthesis = False
    elif mode == DemoMode.HAND_DETECTION:
        enable_hand_detection = True
        enable_mesh_generation = False
        enable_collision_detection = False
        enable_audio_synthesis = False
    elif mode == DemoMode.COLLISION_DETECTION:
        enable_hand_detection = True
        enable_mesh_generation = True
        enable_collision_detection = True
        enable_audio_synthesis = True
    
    return DemoConfiguration(
        mode=mode,
        test_mode=args.test,
        log_level=args.log_level,
        
        # 入力系
        enable_filter=not args.no_filter,
        filter_type=filter_type,
        enable_hand_detection=enable_hand_detection,
        enable_tracking=not args.no_tracking,
        min_detection_confidence=args.detection_confidence,
        use_gpu_mediapipe=args.gpu,
        
        # メッシュ系
        enable_mesh_generation=enable_mesh_generation,
        mesh_update_interval=args.mesh_interval,
        max_mesh_skip_frames=args.max_mesh_skip,
        mesh_resolution=args.mesh_resolution,
        mesh_quality=args.mesh_quality,
        mesh_reduction=args.mesh_reduction,
        
        # 衝突検出系
        enable_collision_detection=enable_collision_detection,
        enable_collision_visualization=not args.no_collision_viz,
        sphere_radius=args.sphere_radius,
        
        # 音響系
        enable_audio_synthesis=enable_audio_synthesis,
        audio_scale=audio_scale,
        audio_instrument=audio_instrument,
        audio_polyphony=args.audio_polyphony,
        audio_master_volume=args.audio_volume,
        
        # UI系
        window_width=args.window_width,
        window_height=args.window_height,
        update_interval=args.update_interval,
        point_size=args.point_size
    )


def run_test_mode() -> int:
    """統一テストモードを実行"""
    logger.info("統一テストモードを実行中...")
    
    try:
        import unittest
        import sys
        import os
        
        # テストディレクトリをパスに追加
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        test_dir = os.path.join(project_root, 'tests')
        sys.path.insert(0, test_dir)
        
        # テストスイートを作成
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern='*_test.py')
        
        # テスト実行
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            logger.info("全テストが正常に完了しました！")
            return 0
        else:
            logger.error(f"テスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
            return 1
            
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        return 1


__all__ = [
    'DemoMode',
    'DemoConfiguration', 
    'create_common_argument_parser',
    'parse_arguments_to_config',
    'run_test_mode'
]