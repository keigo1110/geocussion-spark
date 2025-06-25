"""
Numba JIT configuration and initialization (Lazy Loading)

全モジュールで統一されたNumba設定を提供し、確実な高速化を実現
遅延初期化によりImportError復旧とデバッグ機能を強化
"""

import os
import sys
import time
import importlib
import numpy as np

# グローバルなNumba状態
_NUMBA_INITIALIZED = False
NUMBA_AVAILABLE = False
jit = None
njit = None
numba = None
_LAST_INIT_ATTEMPT = 0
_INIT_RETRY_INTERVAL = 60  # 60秒間隔でリトライ

def _create_dummy_decorators():
    """Numba無効時のダミーデコレータを作成"""
    def dummy_jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # @njit without parentheses
            return args[0]
        return decorator
    
    return dummy_jit, dummy_jit

def _log_environment_info():
    """環境情報をログ出力"""
    print(f"🔍 Python executable: {sys.executable}")
    print(f"🔍 Python prefix: {sys.prefix}")
    print(f"🔍 Working directory: {os.getcwd()}")
    print(f"🔍 sys.path entries: {len(sys.path)}")
    
    # Numba関連パッケージの検索
    for path in sys.path:
        if 'numba' in str(path).lower():
            print(f"🔍 Potential numba path: {path}")

def initialize_numba(force_retry=False, verbose=False):
    """
    Numbaを遅延初期化（複数回安全実行可能）
    
    Args:
        force_retry: 強制的に再初期化を試行
        verbose: 詳細ログを出力
    """
    global _NUMBA_INITIALIZED, NUMBA_AVAILABLE, jit, njit, numba, _LAST_INIT_ATTEMPT
    
    current_time = time.time()
    
    # 初期化済みかつ強制リトライでない場合はスキップ
    if _NUMBA_INITIALIZED and not force_retry:
        if verbose:
            print(f"🔄 Numba already initialized: {NUMBA_AVAILABLE}")
        return NUMBA_AVAILABLE
        
    # リトライ間隔チェック
    if not force_retry and (current_time - _LAST_INIT_ATTEMPT) < _INIT_RETRY_INTERVAL:
        if verbose:
            print(f"🕒 Numba retry cooldown: {_INIT_RETRY_INTERVAL - (current_time - _LAST_INIT_ATTEMPT):.1f}s remaining")
        return NUMBA_AVAILABLE
    
    _LAST_INIT_ATTEMPT = current_time
    
    if verbose:
        print("🔧 Initializing Numba (lazy loading)...")
        _log_environment_info()
    
    try:
        # sys.modulesからNumba関連エントリをクリア（ImportError復旧）
        if force_retry:
            numba_modules = [k for k in sys.modules.keys() if k.startswith('numba')]
            for mod in numba_modules:
                if verbose:
                    print(f"🧹 Clearing cached module: {mod}")
                del sys.modules[mod]
            
            # importlib キャッシュクリア
            importlib.invalidate_caches()
        
        # Numbaインポート試行
        import numba as _numba
        from numba import jit as _jit, njit as _njit, config
        
        # Numba設定の最適化（cache=Falseでキャッシュ問題を回避）
        config.THREADING_LAYER = 'threadsafe'
        config.NUMBA_ENABLE_CUDASIM = False
        config.NUMBA_DISABLE_PERFORMANCE_WARNINGS = 1
        
        # グローバル変数に代入
        numba = _numba
        jit = _jit
        njit = _njit
        NUMBA_AVAILABLE = True
        _NUMBA_INITIALIZED = True
        
        print(f"🚀 Numba JIT acceleration enabled (v{numba.__version__})")
        if verbose:
            print(f"🚀 Numba location: {numba.__file__}")
        
        return True
        
    except ImportError as e:
        if verbose:
            print(f"⚠️ Numba import failed: {e}")
            print(f"⚠️ Current sys.path:")
            for i, path in enumerate(sys.path[:5]):  # 最初の5つだけ表示
                print(f"   {i}: {path}")
        
        # ダミーデコレータを設定
        jit, njit = _create_dummy_decorators()
        NUMBA_AVAILABLE = False
        _NUMBA_INITIALIZED = True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Numba initialization error: {e}")
        
        # ダミーデコレータを設定
        jit, njit = _create_dummy_decorators()
        NUMBA_AVAILABLE = False
        _NUMBA_INITIALIZED = True
        
        return False

def get_numba():
    """
    Numbaオブジェクトを取得（遅延初期化）
    
    Returns:
        tuple: (jit, njit, available)
    """
    if not _NUMBA_INITIALIZED:
        initialize_numba()
    
    return jit, njit, NUMBA_AVAILABLE

def get_numba_status():
    """Numba状態を取得"""
    if not _NUMBA_INITIALIZED:
        initialize_numba()
        
    return {
        'available': NUMBA_AVAILABLE,
        'version': numba.__version__ if NUMBA_AVAILABLE else None,
        'jit_function': jit,
        'njit_function': njit,
        'initialized': _NUMBA_INITIALIZED,
        'last_attempt': _LAST_INIT_ATTEMPT
    }

def get_optimized_jit_config():
    """
    最適化されたJIT設定を取得（cache問題を回避）
    
    Returns:
        dict: JIT設定辞書
    """
    # キャッシュを無効化してランタイムエラーを回避
    return {
        'cache': False,  # cache=Falseで<string>問題を回避
        'fastmath': True,
        'nogil': True,
        'inline': 'always'
    }

def create_optimized_jit(parallel=False):
    """
    最適化されたJITデコレータを作成
    
    Args:
        parallel: 並列処理を有効化
        
    Returns:
        decorator: JITデコレータ
    """
    jit_func, njit_func, available = get_numba()
    
    if not available:
        return jit_func  # ダミーデコレータ
    
    config = get_optimized_jit_config()
    if parallel:
        config['parallel'] = True
    
    return njit_func(**config)

def warmup_basic_functions():
    """基本的なJIT関数をウォームアップ"""
    jit_func, njit_func, available = get_numba()
    
    if not available:
        print("⚠️ Numba not available, skipping warmup")
        return False
        
    try:
        # キャッシュ無効でウォームアップ関数を作成
        @njit_func(**get_optimized_jit_config())
        def test_add(a, b):
            return a + b
            
        @njit_func(**get_optimized_jit_config())
        def test_distance(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2))
        
        # ウォームアップ実行
        _ = test_add(1.0, 2.0)
        p1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        p2 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        _ = test_distance(p1, p2)
        
        print("🔥 JIT functions warmed up - maximum performance ready")
        return True
        
    except Exception as e:
        print(f"⚠️ Numba warmup failed: {e}")
        return False

def retry_numba_initialization(verbose=True):
    """
    Numba初期化を手動リトライ
    
    Args:
        verbose: 詳細ログを出力
        
    Returns:
        bool: 成功/失敗
    """
    print("🔄 Retrying Numba initialization...")
    success = initialize_numba(force_retry=True, verbose=verbose)
    
    if success:
        warmup_basic_functions()
        print("✅ Numba retry successful")
    else:
        print("❌ Numba retry failed")
    
    return success

# モジュール読み込み時は何もしない（遅延初期化）
# 使用時に自動的に初期化される 