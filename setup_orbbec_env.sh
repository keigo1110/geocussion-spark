#!/bin/bash
# OrbbecSDK環境変数設定スクリプト
# Cursor内ターミナルでOrbbecSDKを使用するために必要

echo "🔧 Setting up OrbbecSDK environment for Cursor terminal..."

# プロジェクトルートディレクトリを取得
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# OrbbecSDKのライブラリパスを設定
export LD_LIBRARY_PATH="${PROJECT_ROOT}/pyorbbecsdk:${LD_LIBRARY_PATH}"

# PythonパスにOrbbecSDKを追加
export PYTHONPATH="${PROJECT_ROOT}/pyorbbecsdk:${PYTHONPATH}"

# 仮想環境がアクティブでない場合はアクティベート
if [[ "$VIRTUAL_ENV" != *"geocussion-spark/venv" ]]; then
    echo "🔧 Activating virtual environment..."
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

echo "✅ OrbbecSDK environment configured successfully!"
echo "   LD_LIBRARY_PATH: ${PROJECT_ROOT}/pyorbbecsdk"
echo "   PYTHONPATH: ${PROJECT_ROOT}/pyorbbecsdk"
echo "   Virtual environment: $(basename ${VIRTUAL_ENV})"

# 動作確認（オプション）
if [[ "$1" == "--test" ]]; then
    echo ""
    echo "🧪 Testing OrbbecSDK import..."
    python3 -c "
try:
    from pyorbbecsdk import Pipeline, FrameSet, Config
    print('✅ OrbbecSDK import successful in Cursor terminal!')
except Exception as e:
    print('❌ OrbbecSDK import failed:', e)
    "
fi

echo ""
echo "💡 Usage:"
echo "   source setup_orbbec_env.sh          # Setup environment"
echo "   source setup_orbbec_env.sh --test   # Setup + test import"
echo "   python3 demo_collision_detection.py # Run demo with camera support" 