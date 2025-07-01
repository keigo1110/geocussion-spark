#!/bin/bash
# OrbbecSDK環境変数設定スクリプト
# Cursor内ターミナルでOrbbecSDKを使用するために必要

echo "🔧 Setting up OrbbecSDK environment for Cursor terminal..."

# プロジェクトルートディレクトリを取得
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# OrbbecSDKのライブラリパスを設定（vendor配下に移動済み）
export LD_LIBRARY_PATH="${PROJECT_ROOT}/vendor/pyorbbecsdk:${LD_LIBRARY_PATH}"

# PythonパスにOrbbecSDKを追加（vendor配下を追加）
export PYTHONPATH="${PROJECT_ROOT}/vendor/pyorbbecsdk:${PROJECT_ROOT}:${PYTHONPATH}"

# 仮想環境がアクティブでない場合はアクティベート
if [[ "$VIRTUAL_ENV" != *"geocussion-spark/venv" ]]; then
    echo "🔧 Activating virtual environment..."
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

echo "✅ OrbbecSDK environment configured successfully!"
echo "   LD_LIBRARY_PATH: ${PROJECT_ROOT}/vendor/pyorbbecsdk"
echo "   PYTHONPATH: ${PROJECT_ROOT}/vendor/pyorbbecsdk:${PROJECT_ROOT}"
echo "   Virtual environment: $(basename ${VIRTUAL_ENV})"

# 動作確認（オプション）
if [[ "$1" == "--test" ]]; then
    echo ""
    echo "🧪 Testing OrbbecSDK import..."
    python3 -c "
import sys
try:
    # vendor版の直接import
    sys.path.insert(0, '${PROJECT_ROOT}/vendor/pyorbbecsdk')
    from pyorbbecsdk import Pipeline, FrameSet, Config
    print('✅ OrbbecSDK import successful via vendor path!')
except Exception as e:
    print('❌ OrbbecSDK import failed:', e)
    print('Python path:', sys.path[:3])
    "
fi

echo ""
echo "💡 Usage:"
echo "   source setup_orbbec_env.sh          # Setup environment"
echo "   source setup_orbbec_env.sh --test   # Setup + test import"
echo "   python3 demo_collision_detection.py # Run demo with camera support" 