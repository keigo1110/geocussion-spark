#!/usr/bin/env python3
"""
Geocussion-SP 統一デモランナー

全デモモードを統一的に実行するランナークラス。
Clean Architectureパターンに基づく設計で、各デモモードの実行を統括します。
"""

import sys
import signal
import traceback
from typing import Optional
from contextlib import contextmanager

from . import (
    DemoMode, DemoConfiguration, 
    create_common_argument_parser, parse_arguments_to_config, run_test_mode
)
from ..debug.ui_viewer import GeocussionUIViewer
from ..debug.dual_viewer import DualViewer
from .. import setup_logging, get_logger


class DemoRunner:
    """統一デモランナー"""
    
    def __init__(self, config: DemoConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.viewer: Optional[object] = None
        self.is_running = False
        
        # ログシステム初期化
        setup_logging(level=config.log_level, format_style="detailed")
        
        # シグナルハンドラ設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """シグナルハンドラ"""
        self.logger.info(f"Signal {signum} received, shutting down...")
        self.is_running = False
        if self.viewer and hasattr(self.viewer, 'cleanup'):
            self.viewer.cleanup()
        sys.exit(0)
    
    def run(self) -> int:
        """デモを実行"""
        try:
            self.logger.info(f"Starting demo in {self.config.mode.value} mode")
            
            # テストモード
            if self.config.test_mode:
                return run_test_mode()
            
            # デモモード別実行
            if self.config.mode == DemoMode.BASIC_VIEWER:
                return self._run_basic_viewer()
            elif self.config.mode == DemoMode.HAND_DETECTION:
                return self._run_hand_detection()
            elif self.config.mode == DemoMode.COLLISION_DETECTION:
                return self._run_collision_detection()
            elif self.config.mode == DemoMode.CLEAN_ARCHITECTURE:
                return self._run_clean_architecture()
            else:
                self.logger.error(f"Unknown demo mode: {self.config.mode}")
                return 1
                
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Demo execution error: {e}", exc_info=True)
            return 1
        finally:
            self._cleanup()
    
    def _run_basic_viewer(self) -> int:
        """基本ビューワーを実行"""
        self.logger.info("Running basic viewer (RGB + Depth + Point Cloud)")
        
        self.viewer = DualViewer(
            enable_filter=self.config.enable_filter,
            update_interval=self.config.update_interval,
            point_size=self.config.point_size,
            rgb_window_size=(self.config.window_width, self.config.window_height)
        )
        
        self._print_basic_controls()
        self.viewer.run()
        return 0
    
    def _run_hand_detection(self) -> int:
        """手検出デモを実行"""
        self.logger.info("Running hand detection demo")
        
        self.viewer = DualViewer(
            enable_filter=self.config.enable_filter,
            enable_hand_detection=self.config.enable_hand_detection,
            enable_tracking=self.config.enable_tracking,
            update_interval=self.config.update_interval,
            point_size=self.config.point_size,
            rgb_window_size=(self.config.window_width, self.config.window_height),
            min_detection_confidence=self.config.min_detection_confidence,
            use_gpu_mediapipe=self.config.use_gpu_mediapipe
        )
        
        self._print_hand_detection_controls()
        self.viewer.run()
        return 0
    
    def _run_collision_detection(self) -> int:
        """衝突検出デモを実行（レガシーモード）"""
        self.logger.info("Running collision detection demo (legacy mode)")
        
        # レガシーFullPipelineViewerを使用
        try:
            # 動的インポート（循環参照回避）
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # demo_collision_detectionを直接インポート
            demo_path = os.path.join(project_root, 'demo_collision_detection.py')
            spec = self._import_file_as_module('demo_collision_detection', demo_path)
            
            if spec and spec.loader:
                demo_module = spec.loader.load_module(spec)
                
                # FullPipelineViewerを作成
                from demo_collision_detection import FullPipelineViewer
                
                self.viewer = FullPipelineViewer(
                    enable_mesh_generation=self.config.enable_mesh_generation,
                    enable_collision_detection=self.config.enable_collision_detection,
                    enable_collision_visualization=self.config.enable_collision_visualization,
                    sphere_radius=self.config.sphere_radius,
                    mesh_update_interval=self.config.mesh_update_interval,
                    max_mesh_skip_frames=self.config.max_mesh_skip_frames,
                    enable_audio_synthesis=self.config.enable_audio_synthesis,
                    audio_scale=self.config.audio_scale,
                    audio_instrument=self.config.audio_instrument,
                    audio_polyphony=self.config.audio_polyphony,
                    audio_master_volume=self.config.audio_master_volume
                )
                
                self._print_collision_controls()
                self.viewer.run()
                return 0
                
            else:
                self.logger.error("Failed to load collision detection module")
                return 1
                
        except Exception as e:
            self.logger.error(f"Error running collision detection: {e}")
            # フォールバックとしてClean Architectureモードを実行
            self.logger.info("Falling back to clean architecture mode")
            return self._run_clean_architecture()
    
    def _run_clean_architecture(self) -> int:
        """Clean Architectureデモを実行"""
        self.logger.info("Running clean architecture demo")
        
        # パイプライン設定変換
        pipeline_config = self.config.to_pipeline_config()
        
        # UIビューワー初期化
        self.viewer = GeocussionUIViewer(
            enable_filter=self.config.enable_filter,
            enable_hand_detection=self.config.enable_hand_detection,
            enable_tracking=self.config.enable_tracking,
            min_detection_confidence=self.config.min_detection_confidence,
            use_gpu_mediapipe=self.config.use_gpu_mediapipe,
            
            enable_mesh_generation=self.config.enable_mesh_generation,
            mesh_update_interval=self.config.mesh_update_interval,
            max_mesh_skip_frames=self.config.max_mesh_skip_frames,
            
            enable_collision_detection=self.config.enable_collision_detection,
            enable_collision_visualization=self.config.enable_collision_visualization,
            sphere_radius=self.config.sphere_radius,
            
            enable_audio_synthesis=self.config.enable_audio_synthesis,
            audio_scale=self.config.audio_scale,
            audio_instrument=self.config.audio_instrument,
            audio_polyphony=self.config.audio_polyphony,
            audio_master_volume=self.config.audio_master_volume
        )
        
        # 初期化
        if not self.viewer.initialize():
            self.logger.error("Failed to initialize viewer")
            return 1
        
        self.logger.info("Viewer initialized successfully")
        
        self._print_clean_architecture_controls()
        
        # メインループ実行
        self.logger.info("Starting main loop...")
        self.viewer.run()
        
        return 0
    
    def _import_file_as_module(self, module_name: str, file_path: str):
        """ファイルをモジュールとしてインポート"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            return spec
        except Exception as e:
            self.logger.error(f"Failed to import module {module_name}: {e}")
            return None
    
    def _print_basic_controls(self) -> None:
        """基本ビューワーのコントロール表示"""
        print("\n=== Geocussion-SP 基本ビューワー ===")
        print("RGB + 深度 + 点群表示")
        print("\n=== コントロール ===")
        print("ESC/Q: 終了")
        print("F: フィルタ ON/OFF")
        print("R: フィルタ履歴リセット")
        print("S: 点群保存")
        print("+/-: 点サイズ調整")
        print("=====================================\n")
    
    def _print_hand_detection_controls(self) -> None:
        """手検出デモのコントロール表示"""
        print("\n=== Geocussion-SP 手検出デモ ===")
        print("手検出 + トラッキング + 3D投影")
        print("\n=== コントロール ===")
        print("ESC/Q: 終了")
        print("H: 手検出 ON/OFF")
        print("T: トラッキング ON/OFF")
        print("Y: トラッカーリセット")
        print("F: フィルタ ON/OFF")
        print("R: フィルタ履歴リセット")
        print("=====================================\n")
    
    def _print_collision_controls(self) -> None:
        """衝突検出デモのコントロール表示"""
        print("\n=== Geocussion-SP 衝突検出デモ ===")
        print("全フェーズ統合（手検出+メッシュ+衝突+音響）")
        print("\n=== コントロール ===")
        print("ESC/Q: 終了")
        print("M: メッシュ生成 ON/OFF")
        print("C: 衝突検出 ON/OFF")
        print("V: 衝突可視化 ON/OFF")
        print("N: メッシュ強制更新")
        print("+/-: 球半径調整")
        print("P: パフォーマンス統計表示")
        if self.config.enable_audio_synthesis:
            print("A: 音響合成 ON/OFF")
        print("=====================================\n")
    
    def _print_clean_architecture_controls(self) -> None:
        """Clean Architectureデモのコントロール表示"""
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
        if self.config.enable_audio_synthesis:
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
    
    def _cleanup(self) -> None:
        """クリーンアップ処理"""
        if self.viewer and hasattr(self.viewer, 'cleanup'):
            try:
                self.logger.info("Cleaning up viewer...")
                self.viewer.cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")


def main() -> int:
    """メイン関数"""
    try:
        # 引数解析
        parser = create_common_argument_parser()
        args = parser.parse_args()
        
        # 設定作成
        config = parse_arguments_to_config(args)
        
        # ランナー実行
        runner = DemoRunner(config)
        return runner.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 