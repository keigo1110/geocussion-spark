#!/usr/bin/env python3
"""
Clean Architecture適用デモ
責務分離されたGeocussion-SPパイプライン実行
"""

import argparse
import signal
import sys
import os

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Clean Architecture適用コンポーネント
from src.debug.ui_viewer import GeocussionUIViewer
from src.debug.pipeline_controller import PipelineConfiguration
from src.sound.mapping import ScaleType, InstrumentType

# ログ設定
from src import setup_logging, get_logger

def signal_handler(sig, frame):
    """シグナルハンドラー"""
    print("\nShutting down gracefully...")
    sys.exit(0)

def create_configuration(args) -> PipelineConfiguration:
    """コマンドライン引数からパイプライン設定を作成"""
    return PipelineConfiguration(
        # 入力設定
        enable_filter=not args.no_filter,
        enable_hand_detection=not args.no_hand_detection,
        enable_tracking=not args.no_tracking,
        min_detection_confidence=args.detection_confidence,
        use_gpu_mediapipe=args.gpu,
        
        # メッシュ生成設定
        enable_mesh_generation=not args.no_mesh,
        mesh_update_interval=args.mesh_interval,
        max_mesh_skip_frames=args.max_mesh_skip,
        mesh_resolution=args.mesh_resolution,
        mesh_quality_threshold=args.mesh_quality,
        mesh_reduction=args.mesh_reduction,
        
        # 衝突検出設定
        enable_collision_detection=not args.no_collision,
        enable_collision_visualization=not args.no_collision_viz,
        sphere_radius=args.sphere_radius,
        
        # 音響合成設定
        enable_audio_synthesis=args.audio,
        audio_scale=ScaleType(args.audio_scale),
        audio_instrument=InstrumentType(args.audio_instrument),
        audio_polyphony=args.audio_polyphony,
        audio_master_volume=args.audio_volume,
        audio_cooldown_time=args.audio_cooldown
    )

def main():
    """メイン関数"""
    # コマンドライン引数解析
    parser = argparse.ArgumentParser(
        description="Geocussion-SP Clean Architecture Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 入力関連オプション
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--no-filter', action='store_true',
                           help='Disable depth filtering')
    input_group.add_argument('--no-hand-detection', action='store_true',
                           help='Disable hand detection')
    input_group.add_argument('--no-tracking', action='store_true',
                           help='Disable hand tracking')
    input_group.add_argument('--detection-confidence', type=float, default=0.1,
                           help='Minimum hand detection confidence')
    input_group.add_argument('--gpu', action='store_true',
                           help='Use GPU for MediaPipe')
    
    # メッシュ生成関連オプション
    mesh_group = parser.add_argument_group('Mesh Generation Options')
    mesh_group.add_argument('--no-mesh', action='store_true',
                          help='Disable mesh generation')
    mesh_group.add_argument('--mesh-interval', type=int, default=5,
                          help='Mesh update interval (frames)')
    mesh_group.add_argument('--max-mesh-skip', type=int, default=60,
                          help='Maximum mesh skip frames')
    mesh_group.add_argument('--mesh-resolution', type=float, default=0.01,
                          help='Mesh resolution (meters)')
    mesh_group.add_argument('--mesh-quality', type=float, default=0.3,
                          help='Mesh quality threshold')
    mesh_group.add_argument('--mesh-reduction', type=float, default=0.7,
                          help='Mesh reduction ratio')
    
    # 衝突検出関連オプション
    collision_group = parser.add_argument_group('Collision Detection Options')
    collision_group.add_argument('--no-collision', action='store_true',
                               help='Disable collision detection')
    collision_group.add_argument('--no-collision-viz', action='store_true',
                               help='Disable collision visualization')
    collision_group.add_argument('--sphere-radius', type=float, default=0.05,
                               help='Collision sphere radius (meters)')
    
    # 音響合成関連オプション
    audio_group = parser.add_argument_group('Audio Synthesis Options')
    audio_group.add_argument('--audio', action='store_true',
                           help='Enable audio synthesis')
    audio_group.add_argument('--audio-scale', default='pentatonic',
                           choices=[s.value for s in ScaleType],
                           help='Audio scale type')
    audio_group.add_argument('--audio-instrument', default='marimba',
                           choices=[i.value for i in InstrumentType],
                           help='Audio instrument type')
    audio_group.add_argument('--audio-polyphony', type=int, default=4,
                           help='Audio polyphony')
    audio_group.add_argument('--audio-volume', type=float, default=0.7,
                           help='Audio master volume')
    audio_group.add_argument('--audio-cooldown', type=float, default=0.15,
                           help='Audio cooldown time (seconds)')
    
    # 一般オプション
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (exit after initialization)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    logger = get_logger(__name__)
    
    logger.info("Starting Geocussion-SP Clean Architecture Demo")
    
    # シグナルハンドラー設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # パイプライン設定作成
        config = create_configuration(args)
        logger.info(f"Pipeline configuration created")
        logger.debug(f"Configuration: {config}")
        
        # UIビューワー初期化
        logger.info("Initializing UI viewer...")
        viewer = GeocussionUIViewer(
            enable_filter=config.enable_filter,
            enable_hand_detection=config.enable_hand_detection,
            enable_tracking=config.enable_tracking,
            min_detection_confidence=config.min_detection_confidence,
            use_gpu_mediapipe=config.use_gpu_mediapipe,
            
            enable_mesh_generation=config.enable_mesh_generation,
            mesh_update_interval=config.mesh_update_interval,
            max_mesh_skip_frames=config.max_mesh_skip_frames,
            
            enable_collision_detection=config.enable_collision_detection,
            enable_collision_visualization=config.enable_collision_visualization,
            sphere_radius=config.sphere_radius,
            
            enable_audio_synthesis=config.enable_audio_synthesis,
            audio_scale=config.audio_scale,
            audio_instrument=config.audio_instrument,
            audio_polyphony=config.audio_polyphony,
            audio_master_volume=config.audio_master_volume
        )
        
        # 初期化
        if not viewer.initialize():
            logger.error("Failed to initialize viewer")
            return 1
        
        logger.info("Viewer initialized successfully")
        
        # テストモードの場合は初期化後に終了
        if args.test:
            logger.info("Test mode: initialization successful, exiting")
            viewer.cleanup()
            return 0
        
        # メインループ実行
        logger.info("Starting main loop...")
        logger.info("Press ESC or Ctrl+C to exit")
        
        # 操作ガイド表示
        print("\n=== Geocussion-SP Clean Architecture Demo ===")
        print("責務分離されたクリーンアーキテクチャ実装")
        print("\n=== コントロール ===")
        print("ESC: 終了")
        print("H: ヘルプ表示")
        print("M: メッシュ生成 ON/OFF")
        print("C: 衝突検出 ON/OFF")
        print("V: 衝突可視化 ON/OFF")
        print("N: メッシュ強制更新")
        print("+/-: 球半径調整")
        print("P: パフォーマンス統計表示")
        if config.enable_audio_synthesis:
            print("A: 音響合成 ON/OFF")
            print("S: 音階切り替え")
            print("I: 楽器切り替え")
            print("1/2: 音量調整")
        print("\n=== アーキテクチャ特徴 ===")
        print("✓ UI層とビジネスロジック層の完全分離")
        print("✓ Observer パターンによる疎結合")
        print("✓ 設定駆動型パイプライン制御")
        print("✓ ManagedResource による自動リソース管理")
        print("✓ 統一ログシステム")
        print("=====================================\n")
        
        # メインループ
        viewer.run()
        
        logger.info("Main loop finished")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
        
    finally:
        # クリーンアップ
        try:
            if 'viewer' in locals():
                logger.info("Cleaning up viewer...")
                viewer.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    logger.info("Demo finished")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 