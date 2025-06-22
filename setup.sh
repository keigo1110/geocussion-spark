#!/bin/bash

# ===================================================================
# Geocussion-Spark 自動環境構築スクリプト
# ===================================================================

set -e  # エラー時にスクリプトを停止

# カラー出力設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ロゴ表示
echo -e "${BLUE}"
echo "=================================================="
echo "  Geocussion-Spark Environment Setup"
echo "  Hand-Terrain Collision Detection System"
echo "=================================================="
echo -e "${NC}"

# 関数定義
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        print_status "$1 is already installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Ubuntu/Debian)
        sudo apt-get update
        sudo apt-get install -y \
            python3 python3-pip python3-venv \
            libasound2-dev portaudio19-dev \
            libportaudio2 libportaudiocpp0 \
            libopencv-dev python3-opencv \
            libegl1-mesa-dev \
            git curl wget
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if check_command brew; then
            brew install python3 portaudio opencv
        else
            print_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
    else
        print_warning "Unsupported OS. Please install dependencies manually."
    fi
}

setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    # Python バージョンチェック
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python version: $python_version"
    
    # 仮想環境の作成
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    # 仮想環境のアクティベート
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # pipのアップグレード
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
}

install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # requirements.txtからのインストール
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Python dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

setup_orbbec_sdk() {
    print_status "Setting up Orbbec SDK..."
    
    if [ -d "pyorbbecsdk/install/lib" ]; then
        # 環境変数の設定
        export PYTHONPATH="$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib"
        
        # .bashrcに環境変数を追加（オプション）
        if [ "$1" = "--persistent" ]; then
            echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib" >> ~/.bashrc
            print_status "Added PYTHONPATH to ~/.bashrc"
        fi
        
        print_status "Orbbec SDK configured"
    else
        print_warning "Orbbec SDK directory not found. Please ensure pyorbbecsdk is properly extracted."
    fi
}

run_tests() {
    print_status "Running system tests..."
    
    # 基本的なインポートテスト
    python3 -c "
import sys
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
print('✓ All basic imports successful')
"
    
    # デモテストの実行
    if [ -f "demo_collision_detection.py" ]; then
        print_status "Running collision detection test..."
        PYTHONPATH="$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib" python3 demo_collision_detection.py --test
    fi
}

create_run_script() {
    print_status "Creating run script..."
    
    cat > run_demo.sh << 'EOF'
#!/bin/bash
# Geocussion-Spark デモ実行スクリプト

# 仮想環境のアクティベート
source venv/bin/activate

# 環境変数の設定
export PYTHONPATH="$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib"

# デモの実行
echo "Starting Geocussion-Spark Demo..."
python demo_collision_detection.py "$@"
EOF
    
    chmod +x run_demo.sh
    print_status "Created run_demo.sh script"
}

cleanup() {
    print_status "Cleaning up temporary files..."
    # 必要に応じてクリーンアップ処理を追加
}

main() {
    print_status "Starting Geocussion-Spark setup..."
    
    # オプション解析
    INSTALL_SYSTEM_DEPS=true
    RUN_TESTS=true
    PERSISTENT_ENV=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-system-deps)
                INSTALL_SYSTEM_DEPS=false
                shift
                ;;
            --no-tests)
                RUN_TESTS=false
                shift
                ;;
            --persistent)
                PERSISTENT_ENV=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --no-system-deps   Skip system dependency installation"
                echo "  --no-tests        Skip running tests"
                echo "  --persistent      Add environment variables to ~/.bashrc"
                echo "  -h, --help        Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # セットアップ実行
    if [ "$INSTALL_SYSTEM_DEPS" = true ]; then
        install_system_deps
    fi
    
    setup_python_env
    install_python_deps
    
    if [ "$PERSISTENT_ENV" = true ]; then
        setup_orbbec_sdk --persistent
    else
        setup_orbbec_sdk
    fi
    
    create_run_script
    
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi
    
    # 完了メッセージ
    echo -e "${GREEN}"
    echo "=================================================="
    echo "  Setup Complete! 🎉"
    echo "=================================================="
    echo "To run the demo:"
    echo "  ./run_demo.sh"
    echo ""
    echo "To activate the environment manually:"
    echo "  source venv/bin/activate"
    echo "  export PYTHONPATH=\$PYTHONPATH:\$(pwd)/pyorbbecsdk/install/lib"
    echo ""
    echo "For more information, see SETUP.md"
    echo -e "${NC}"
}

# トラップでクリーンアップ
trap cleanup EXIT

# メイン実行
main "$@" 