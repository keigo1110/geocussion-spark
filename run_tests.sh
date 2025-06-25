#!/bin/bash
# =====================================================================
# Geocussion-SP 統合テスト実行スクリプト
# 
# プロフェッショナル品質のテスト実行とCI/CD自動化を提供します。
# pytest統一基盤による包括的品質保証システムです。
# =====================================================================

set -e  # エラー時に即座に終了

# ===================================================================
# 設定・環境変数
# ===================================================================

export PYTHONPATH="${PYTHONPATH}:$(pwd)/pyorbbecsdk/install/lib/"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===================================================================
# ヘルパー関数
# ===================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
使用方法: $0 [OPTIONS] [TEST_CATEGORY]

テストカテゴリ:
  unit            ユニットテスト（高速）
  integration     統合テスト（中速）
  performance     パフォーマンステスト（低速）
  e2e             E2Eテスト（最低速）
  all             全テスト実行（デフォルト）
  
オプション:
  -h, --help      このヘルプを表示
  -v, --verbose   詳細出力モード
  -q, --quiet     静寂モード（エラーのみ）
  -x, --exitfirst 最初の失敗で停止
  -p, --parallel  並列実行（推奨）
  --cov           カバレッジレポート生成
  --html          HTMLレポート生成
  --benchmark     ベンチマーク実行
  --ci            CI モード（レポート生成 + 厳格）
  --quick         クイックテスト（unit のみ）
  
例:
  $0                           # 全テスト実行
  $0 unit                      # ユニットテストのみ
  $0 --ci                      # CI モード
  $0 performance --benchmark   # パフォーマンステスト（詳細）
  $0 --quick                   # クイックテスト
EOF
}

check_environment() {
    log_info "環境チェック中..."
    
    # Python環境確認
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 が見つかりません"
        exit 1
    fi
    
    # 仮想環境確認
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        log_warning "仮想環境が有効化されていません"
        if [[ -d "venv" ]]; then
            log_info "venv を有効化中..."
            source venv/bin/activate
        fi
    fi
    
    # 依存関係確認
    if ! python3 -c "import pytest" &> /dev/null; then
        log_error "pytest がインストールされていません"
        log_info "pip install -r requirements.txt を実行してください"
        exit 1
    fi
    
    # srcモジュール確認
    if ! python3 -c "import src" &> /dev/null; then
        log_error "srcモジュールが見つかりません"
        exit 1
    fi
    
    log_success "環境チェック完了"
}

# ===================================================================
# テスト実行関数
# ===================================================================

run_unit_tests() {
    log_info "ユニットテスト実行中..."
    pytest tests/ -m "unit" \
        --tb=short \
        --durations=10 \
        ${PYTEST_ARGS[@]}
}

run_integration_tests() {
    log_info "統合テスト実行中..."
    pytest tests/ -m "integration" \
        --tb=short \
        --durations=10 \
        ${PYTEST_ARGS[@]}
}

run_performance_tests() {
    log_info "パフォーマンステスト実行中..."
    pytest tests/ -m "performance" \
        --tb=short \
        --durations=20 \
        --disable-warnings \
        ${PYTEST_ARGS[@]}
}

run_e2e_tests() {
    log_info "E2Eテスト実行中..."
    pytest tests/ -m "e2e" \
        --tb=short \
        --durations=20 \
        --maxfail=1 \
        ${PYTEST_ARGS[@]}
}

run_all_tests() {
    log_info "全テストスイート実行中..."
    pytest tests/ \
        --tb=short \
        --durations=20 \
        ${PYTEST_ARGS[@]}
}

run_quick_tests() {
    log_info "クイックテスト実行中（ユニットテストのみ）..."
    pytest tests/ -m "unit and not slow" \
        --tb=line \
        --durations=5 \
        -x \
        ${PYTEST_ARGS[@]}
}

# ===================================================================
# メイン処理
# ===================================================================

main() {
    # デフォルト値
    TEST_CATEGORY="all"
    PYTEST_ARGS=()
    VERBOSE=false
    QUIET=false
    PARALLEL=false
    COVERAGE=false
    HTML_REPORT=false
    BENCHMARK=false
    CI_MODE=false
    QUICK_MODE=false
    
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                PYTEST_ARGS+=("-v")
                shift
                ;;
            -q|--quiet)
                QUIET=true
                PYTEST_ARGS+=("-q")
                shift
                ;;
            -x|--exitfirst)
                PYTEST_ARGS+=("-x")
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                PYTEST_ARGS+=("-n" "auto")
                shift
                ;;
            --cov)
                COVERAGE=true
                PYTEST_ARGS+=("--cov=src" "--cov-report=term-missing")
                shift
                ;;
            --html)
                HTML_REPORT=true
                PYTEST_ARGS+=("--cov-report=html:htmlcov")
                shift
                ;;
            --benchmark)
                BENCHMARK=true
                PYTEST_ARGS+=("--benchmark-only")
                shift
                ;;
            --ci)
                CI_MODE=true
                COVERAGE=true
                HTML_REPORT=true
                PYTEST_ARGS+=("--cov=src" "--cov-report=term-missing" "--cov-report=html:htmlcov" "--cov-report=xml" "--junitxml=pytest-results.xml")
                shift
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            unit|integration|performance|e2e|all)
                TEST_CATEGORY=$1
                shift
                ;;
            *)
                log_error "不明なオプション: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # ヘッダー出力
    echo "======================================================================"
    echo "Geocussion-SP テストスイート実行"
    echo "======================================================================"
    
    # 環境チェック
    check_environment
    
    # テストディレクトリ作成
    mkdir -p htmlcov logs
    
    # ログファイル設定
    LOG_FILE="logs/test_$(date +%Y%m%d_%H%M%S).log"
    
    if [[ "$QUIET" == false ]]; then
        log_info "テストカテゴリ: $TEST_CATEGORY"
        log_info "並列実行: $PARALLEL"
        log_info "カバレッジ: $COVERAGE"
        log_info "ログファイル: $LOG_FILE"
    fi
    
    # テスト実行
    case $TEST_CATEGORY in
        unit)
            run_unit_tests | tee "$LOG_FILE"
            ;;
        integration)
            run_integration_tests | tee "$LOG_FILE"
            ;;
        performance)
            run_performance_tests | tee "$LOG_FILE"
            ;;
        e2e)
            run_e2e_tests | tee "$LOG_FILE"
            ;;
        all)
            if [[ "$QUICK_MODE" == true ]]; then
                run_quick_tests | tee "$LOG_FILE"
            else
                run_all_tests | tee "$LOG_FILE"
            fi
            ;;
        *)
            log_error "無効なテストカテゴリ: $TEST_CATEGORY"
            exit 1
            ;;
    esac
    
    # 結果の要約
    EXIT_CODE=$?
    echo "======================================================================"
    if [[ $EXIT_CODE -eq 0 ]]; then
        log_success "全テストが成功しました！"
        if [[ "$HTML_REPORT" == true ]]; then
            log_info "HTMLレポート: htmlcov/index.html"
        fi
    else
        log_error "テストに失敗しました（終了コード: $EXIT_CODE）"
        log_info "詳細ログ: $LOG_FILE"
    fi
    
    echo "======================================================================"
    exit $EXIT_CODE
}

# スクリプト実行
main "$@" 