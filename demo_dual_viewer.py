#!/usr/bin/env python3
"""
Geocussion-SP デュアルウィンドウビューワー デモ（互換性エイリアス）

注意: このスクリプトは互換性のために残されています。
新しい統一デモシステムを使用することを推奨します:
    python demo_unified.py --mode basic

使用方法:
    python demo_dual_viewer.py
"""

import sys
import os

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("【互換性モード】demo_dual_viewer.py")
print("新しい統一デモシステムにリダイレクトしています...")
print("今後は 'python demo_unified.py --mode basic' をご使用ください。")
print()

# 引数を統一システム用に変換
unified_args = ['--mode', 'basic']

# 既存の引数を統一システムの引数に変換
original_args = sys.argv[1:]
for arg in original_args:
    if arg == '--test':
        unified_args.append('--test')
    elif arg == '--no-filter':
        unified_args.append('--no-filter')
    elif arg.startswith('--filter-type='):
        filter_type = arg.split('=')[1]
        unified_args.extend(['--filter-type', filter_type])
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


def create_demo_viewer(args):
    """デモ用ビューワーを作成"""
    
    # フィルタタイプの選択
    filter_types = []
    if args.enable_filter:
        if args.filter_type == "median":
            filter_types = [FilterType.MEDIAN]
        elif args.filter_type == "bilateral":
            filter_types = [FilterType.BILATERAL]
        elif args.filter_type == "temporal":
            filter_types = [FilterType.TEMPORAL]
        else:  # combined
            filter_types = [FilterType.COMBINED]
    
    # ビューワー作成
    viewer = DualViewer(
        enable_filter=args.enable_filter,
        update_interval=args.update_interval,
        point_size=args.point_size,
        rgb_window_size=(args.window_width, args.window_height)
    )
    
    return viewer


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Geocussion-SP デュアルウィンドウビューワー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python demo_dual_viewer.py                    # デフォルト設定で実行
    python demo_dual_viewer.py --no-filter        # フィルタ無効
    python demo_dual_viewer.py --filter-type median  # メディアンフィルタのみ
    python demo_dual_viewer.py --point-size 3.0   # 点サイズを大きく
    python demo_dual_viewer.py --update-interval 1 # 毎フレーム点群更新

操作方法:
    RGB ウィンドウ:
        Q/ESC: 終了
        F: フィルタ ON/OFF 切り替え
        R: フィルタ履歴リセット
    
    3D ビューワー:
        マウス: 回転/パン/ズーム
        R: 視点リセット
        +/-: 点サイズ変更
        S: 点群保存
        """
    )
    
    # 引数定義
    parser.add_argument(
        '--enable-filter', '--filter',
        action='store_true',
        default=True,
        help='深度フィルタを有効にする（デフォルト: 有効）'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='深度フィルタを無効にする'
    )
    
    parser.add_argument(
        '--filter-type',
        choices=['median', 'bilateral', 'temporal', 'combined'],
        default='combined',
        help='フィルタタイプを選択（デフォルト: combined）'
    )
    
    parser.add_argument(
        '--update-interval',
        type=int,
        default=3,
        help='点群更新間隔（フレーム数、デフォルト: 3）'
    )
    
    parser.add_argument(
        '--point-size',
        type=float,
        default=2.0,
        help='点群の点サイズ（デフォルト: 2.0）'
    )
    
    parser.add_argument(
        '--window-width',
        type=int,
        default=640,
        help='RGBウィンドウの幅（デフォルト: 640）'
    )
    
    parser.add_argument(
        '--window-height',
        type=int,
        default=480,
        help='RGBウィンドウの高さ（デフォルト: 480）'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='テストモードで実行（カメラ無しでモジュールテスト）'
    )
    
    args = parser.parse_args()
    
    # no-filter フラグの処理
    if args.no_filter:
        args.enable_filter = False
    
    # 情報表示
    print("=" * 60)
    print("Geocussion-SP デュアルウィンドウビューワー")
    print("=" * 60)
    print(f"フィルタ: {'有効' if args.enable_filter else '無効'}")
    if args.enable_filter:
        print(f"フィルタタイプ: {args.filter_type}")
    print(f"点群更新間隔: {args.update_interval} フレーム")
    print(f"点サイズ: {args.point_size}")
    print(f"RGBウィンドウサイズ: {args.window_width}x{args.window_height}")
    print("=" * 60)
    
    # テストモード
    if args.test:
        print("テストモードで実行中...")
        try:
            import unittest
            import sys
            import os
            
            # テストディレクトリをパスに追加
            test_dir = os.path.join(PROJECT_ROOT, 'tests')
            sys.path.insert(0, test_dir)
            
            # テストスイートを作成
            loader = unittest.TestLoader()
            suite = loader.discover(test_dir, pattern='*_test.py')
            
            # テスト実行
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            if result.wasSuccessful():
                print("全テストが正常に完了しました！")
                return 0
            else:
                print(f"テスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
                return 1
        except Exception as e:
            print(f"テスト実行エラー: {e}")
            return 1
    
    # カメラ検証
    try:
        import pyorbbecsdk
        print("pyorbbecsdk インポート成功")
    except ImportError:
        print("エラー: pyorbbecsdk がインストールされていません")
        print("以下のコマンドでインストールしてください:")
        print("pip install pyorbbecsdk")
        return 1
    
    # ビューワー実行
    try:
        viewer = create_demo_viewer(args)
        print("\nビューワーを開始します...")
        print("終了するには RGB ウィンドウで Q キーまたは ESC キーを押してください")
        
        viewer.run()
        
        print("ビューワーが正常に終了しました")
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