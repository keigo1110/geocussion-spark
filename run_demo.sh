#!/bin/bash
# Geocussion-Spark Demo Launcher
# PyOrbbecSDKのパスを設定してデモを実行

cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib/

echo "Starting Geocussion-Spark Demo..."
echo "PYTHONPATH set to: $PYTHONPATH"

# デモを実行
python3 demo_collision_detection.py "$@" 