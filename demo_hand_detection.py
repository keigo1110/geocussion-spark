#!/usr/bin/env python3
"""
Geocussion-SP 手検出フェーズ統合デモ（互換性エイリアス）

注意: このスクリプトは互換性のために残されています。
新しい統一デモシステムを使用することを推奨します:
    python demo_unified.py --mode hands

使用方法:
    python demo_hand_detection.py
"""

import sys
import os

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("【互換性モード】demo_hand_detection.py")
print("新しい統一デモシステムにリダイレクトしています...")
print("今後は 'python demo_unified.py --mode hands' をご使用ください。")
print()

# 引数を統一システム用に変換
unified_args = ['--mode', 'hands']

# 既存の引数を統一システムの引数に変換
original_args = sys.argv[1:]
for arg in original_args:
    if arg == '--test':
        unified_args.append('--test')
    elif arg == '--no-filter':
        unified_args.append('--no-filter')
    elif arg == '--no-hand-detection':
        unified_args.append('--no-hand-detection')
    elif arg == '--no-tracking':
        unified_args.append('--no-tracking')
    elif arg == '--gpu-mediapipe':
        unified_args.append('--gpu')
    elif arg.startswith('--min-confidence='):
        confidence = arg.split('=')[1]
        unified_args.extend(['--detection-confidence', confidence])
    elif arg.startswith('--update-interval='):
        interval = arg.split('=')[1]
        unified_args.extend(['--update-interval', interval])
    elif arg.startswith('--point-size='):
        size = arg.split('=')[1]
        unified_args.extend(['--point-size', size])
    elif arg.startswith('--window-width='):
        width = arg.split('=')[1]
        unified_args.extend(['--window-width', width])
    elif arg.startswith('--window-height='):
        height = arg.split('=')[1]
        unified_args.extend(['--window-height', height])

# 統一システムに移譲
sys.argv = ['demo_unified.py'] + unified_args

try:
    from src.demo.runner import main
    exit_code = main()
    sys.exit(exit_code)
except ImportError as e:
    print(f"エラー: 統一デモシステムを読み込めません: {e}")
    print("demo_unified.py を直接実行してください。")
    sys.exit(1)


def create_hand_detection_viewer(args):
    """手検出機能付きDualViewerを作成"""
    
    # ウィンドウサイズ設定
    if args.high_resolution:
        rgb_window_size = (1280, 720)
    else:
        rgb_window_size = (args.window_width, args.window_height)
    
    # DualViewer作成（手検出機能付き）
    viewer = DualViewer(
        enable_filter=not args.no_filter,
        enable_hand_detection=not args.no_hand_detection,
        enable_tracking=not args.no_tracking,
        update_interval=args.update_interval,
        point_size=args.point_size,
        rgb_window_size=rgb_window_size,
        min_detection_confidence=args.min_confidence,
        use_gpu_mediapipe=args.gpu_mediapipe
    )
    
    return viewer


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Geocussion-SP 手検出フェーズ統合デモ（DualViewer使用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python demo_hand_detection.py                    # デフォルト設定
    python demo_hand_detection.py --no-filter        # フィルタ無効
    python demo_hand_detection.py --gpu-mediapipe    # MediaPipe GPU使用
    python demo_hand_detection.py --no-tracking      # トラッキング無効
    python demo_hand_detection.py --no-hand-detection # 手検出無効
    python demo_hand_detection.py --high-resolution  # 高解像度表示

操作方法:
    RGB Window:
        Q/ESC: 終了
        F: 深度フィルタ ON/OFF
        R: フィルタ履歴リセット
        H: 手検出 ON/OFF
        T: トラッキング ON/OFF
        Y: トラッカーリセット
    
    3D Viewer:
        マウス: 回転/パン/ズーム
        R: 視点リセット
        +/-: 点サイズ変更
        S: 点群保存
        """
    )
    
    # 基本設定
    parser.add_argument('--no-filter', action='store_true', help='深度フィルタを無効にする')
    parser.add_argument('--no-hand-detection', action='store_true', help='手検出を無効にする')
    parser.add_argument('--no-tracking', action='store_true', help='トラッキングを無効にする')
    parser.add_argument('--gpu-mediapipe', action='store_true', help='MediaPipeでGPUを使用')
    
    # 手検出設定
    parser.add_argument('--min-confidence', type=float, default=0.7, help='最小検出信頼度 (0.0-1.0)')
    
    # 表示設定
    parser.add_argument('--update-interval', type=int, default=3, help='点群更新間隔（フレーム数）')
    parser.add_argument('--point-size', type=float, default=2.0, help='点群の点サイズ')
    parser.add_argument('--high-resolution', action='store_true', help='高解像度表示 (1280x720)')
    
    # ウィンドウサイズ
    parser.add_argument('--window-width', type=int, default=640, help='RGBウィンドウの幅')
    parser.add_argument('--window-height', type=int, default=480, help='RGBウィンドウの高さ')
    
    # テストモード
    parser.add_argument('--test', action='store_true', help='テストモードで実行')
    
    args = parser.parse_args()
    
    # 設定値検証
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("Error: --min-confidence must be between 0.0 and 1.0")
        return 1
    
    if args.update_interval < 1:
        print("Error: --update-interval must be at least 1")
        return 1
    
    if args.point_size <= 0:
        print("Error: --point-size must be positive")
        return 1
    
    # 情報表示
    print("=" * 70)
    print("Geocussion-SP 手検出フェーズ統合デモ（DualViewer版）")
    print("=" * 70)
    print(f"深度フィルタ: {'無効' if args.no_filter else '有効'}")
    print(f"手検出: {'無効' if args.no_hand_detection else '有効'}")
    if not args.no_hand_detection:
        print(f"  - MediaPipe GPU: {'有効' if args.gpu_mediapipe else '無効'}")
        print(f"  - トラッキング: {'無効' if args.no_tracking else '有効'}")
        print(f"  - 最小検出信頼度: {args.min_confidence}")
    print(f"点群更新間隔: {args.update_interval} フレーム")
    print(f"点サイズ: {args.point_size}")
    if args.high_resolution:
        print(f"表示解像度: 1280x720 (高解像度)")
    else:
        print(f"RGBウィンドウサイズ: {args.window_width}x{args.window_height}")
    print("=" * 70)
    
    # テストモード
    if args.test:
        print("テストモードで実行中...")
        try:
            import unittest
            
            # テストディレクトリをパスに追加
            test_dir = os.path.join(PROJECT_ROOT, 'tests')
            sys.path.insert(0, test_dir)
            
            # 検出フェーズのテストのみ実行
            loader = unittest.TestLoader()
            suite = loader.discover(test_dir, pattern='*detection_test.py')
            
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            if result.wasSuccessful():
                print("手検出フェーズのテストが正常に完了しました！")
                return 0
            else:
                print(f"テスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
                return 1
        except Exception as e:
            print(f"テスト実行エラー: {e}")
            return 1
    
    # 依存関係確認
    try:
        import pyorbbecsdk
        print("✓ pyorbbecsdk インポート成功")
    except ImportError:
        print("✗ エラー: pyorbbecsdk がインストールされていません")
        print("以下のコマンドでインストールしてください:")
        print("pip install pyorbbecsdk")
        return 1
    
    try:
        import mediapipe
        print("✓ mediapipe インポート成功")
    except ImportError:
        print("✗ エラー: mediapipe がインストールされていません")
        print("以下のコマンドでインストールしてください:")
        print("pip install mediapipe")
        return 1
    
    try:
        import open3d
        print("✓ open3d インポート成功")
    except ImportError:
        print("✗ エラー: open3d がインストールされていません")
        print("以下のコマンドでインストールしてください:")
        print("pip install open3d")
        return 1
    
    # DualViewer実行
    try:
        viewer = create_hand_detection_viewer(args)
        print("\n手検出機能付きDualViewerを開始します...")
        print("=" * 70)
        
        viewer.run()
        
        print("\nDualViewerが正常に終了しました")
        return 0
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました")
        return 0
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 