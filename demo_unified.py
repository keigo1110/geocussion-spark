#!/usr/bin/env python3
"""
Geocussion-SP 統一デモシステム エントリーポイント

全デモスクリプトを統一したエントリーポイント。
既存のdemo_*.pyスクリプトの機能を全て包含し、Clean Architectureパターンで実装。

使用方法:
    # 基本ビューワー（旧demo_dual_viewer.py相当）
    python demo_unified.py --mode basic
    
    # 手検出デモ（旧demo_hand_detection.py相当）
    python demo_unified.py --mode hands
    
    # 衝突検出デモ（旧demo_collision_detection.py相当）
    python demo_unified.py --mode collision
    
    # Clean Architectureデモ（旧demo_clean_architecture.py相当）
    python demo_unified.py --mode clean
    
    # テストモード
    python demo_unified.py --test

機能:
    - 引数解析の統一化（重複排除）
    - 初期化処理の共通化
    - エラーハンドリングの統一
    - ログシステムの統一
    - Clean Architectureパターンによる保守性向上
"""

import sys
import os

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.demo.runner import main
    
except ImportError as e:
    print(f"エラー: 必要なモジュールをインポートできません")
    print(f"詳細: {e}")
    print("\n以下を確認してください:")
    print("1. 仮想環境がアクティベートされているか")
    print("2. requirements.txtのパッケージがインストールされているか")
    print("3. pyorbbecsdk が正しくインストールされているか")
    print("\nインストール手順:")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 