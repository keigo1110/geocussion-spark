"""
Numba JIT configuration and initialization (Lazy Loading)

全モジュールで統一されたNumba設定を提供し、確実な高速化を実現
遅延初期化によりImportError復旧とデバッグ機能を強化
Numbaキャッシュ自動クリア機能による長期安定化
"""

import os
import sys
import time
import importlib
import threading
import shutil
import tempfile
import atexit
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# グローバルなNumba状態
_NUMBA_INITIALIZED = False
NUMBA_AVAILABLE = False
jit = None
njit = None
numba = None
_LAST_INIT_ATTEMPT = 0
_INIT_RETRY_INTERVAL = 60  # 60秒間隔でリトライ

# Numbaキャッシュ管理
_CACHE_CLEANUP_THREAD_ACTIVE = False
_CACHE_CLEANUP_LOCK = threading.Lock()

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

def _find_numba_cache_directories() -> list[Path]:
    """Numbaキャッシュディレクトリを検索"""
    cache_dirs = []
    
    # 一般的なキャッシュディレクトリ
    possible_locations = [
        Path.home() / '.numba_cache',
        Path.home() / '.cache' / 'numba',
        Path(tempfile.gettempdir()) / 'numba_cache',
        Path('/tmp') / 'numba_cache',
        Path('/var/tmp') / 'numba_cache',
    ]
    
    # 環境変数から取得
    if 'NUMBA_CACHE_DIR' in os.environ:
        possible_locations.append(Path(os.environ['NUMBA_CACHE_DIR']))
    
    # 実際に存在するディレクトリを検索
    for location in possible_locations:
        if location.exists() and location.is_dir():
            cache_dirs.append(location)
    
    # __pycache__内のNumbaキャッシュファイルも検索
    current_dir = Path.cwd()
    for pycache_dir in current_dir.rglob('__pycache__'):
        if any(f.name.endswith('.nbc') or f.name.endswith('.nbi') for f in pycache_dir.iterdir()):
            cache_dirs.append(pycache_dir)
    
    return cache_dirs

def _get_cache_size_mb(cache_dir: Path) -> float:
    """キャッシュディレクトリのサイズを取得（MB）"""
    try:
        total_size = 0
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    except Exception:
        return 0.0

def _cleanup_numba_cache_directory(cache_dir: Path, max_age_hours: float = 24.0) -> Dict[str, Any]:
    """Numbaキャッシュディレクトリをクリーンアップ"""
    cleanup_stats = {
        'directory': str(cache_dir),
        'files_removed': 0,
        'size_freed_mb': 0.0,
        'errors': []
    }
    
    try:
        if not cache_dir.exists():
            return cleanup_stats
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                try:
                    # ファイルの最終変更時刻をチェック
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleanup_stats['files_removed'] += 1
                        cleanup_stats['size_freed_mb'] += file_size / (1024 * 1024)
                        
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error removing {file_path}: {e}")
        
        # 空のディレクトリを削除
        for dir_path in sorted(cache_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error removing empty dir {dir_path}: {e}")
                    
    except Exception as e:
        cleanup_stats['errors'].append(f"General cleanup error: {e}")
    
    return cleanup_stats

def cleanup_numba_cache(max_age_hours: float = 24.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Numbaキャッシュを自動クリーンアップ
    
    Args:
        max_age_hours: この時間より古いキャッシュファイルを削除
        verbose: 詳細ログを出力
    
    Returns:
        クリーンアップ統計
    """
    if verbose:
        print(f"🧹 Starting Numba cache cleanup (max_age: {max_age_hours}h)")
    
    cache_dirs = _find_numba_cache_directories()
    total_stats = {
        'directories_processed': 0,
        'total_files_removed': 0,
        'total_size_freed_mb': 0.0,
        'directories': [],
        'errors': []
    }
    
    for cache_dir in cache_dirs:
        if verbose:
            size_before = _get_cache_size_mb(cache_dir)
            print(f"🧹 Processing cache directory: {cache_dir} ({size_before:.1f}MB)")
        
        cleanup_result = _cleanup_numba_cache_directory(cache_dir, max_age_hours)
        total_stats['directories'].append(cleanup_result)
        total_stats['total_files_removed'] += cleanup_result['files_removed']
        total_stats['total_size_freed_mb'] += cleanup_result['size_freed_mb']
        total_stats['errors'].extend(cleanup_result['errors'])
        
        if verbose and cleanup_result['files_removed'] > 0:
            print(f"🧹 Cleaned {cleanup_result['files_removed']} files, "
                  f"freed {cleanup_result['size_freed_mb']:.1f}MB from {cache_dir}")
    
    total_stats['directories_processed'] = len(cache_dirs)
    
    if verbose:
        print(f"🧹 Numba cache cleanup completed: "
              f"{total_stats['total_files_removed']} files removed, "
              f"{total_stats['total_size_freed_mb']:.1f}MB freed")
        
        if total_stats['errors']:
            print(f"⚠️ {len(total_stats['errors'])} errors occurred during cleanup")
    
    return total_stats

def _background_cache_cleanup():
    """バックグラウンドキャッシュクリーンアップ"""
    global _CACHE_CLEANUP_THREAD_ACTIVE
    
    while _CACHE_CLEANUP_THREAD_ACTIVE:
        try:
            # 6時間間隔でクリーンアップ（60秒間隔でチェック）
            for _ in range(360):  # 6時間 = 360 * 60秒
                if not _CACHE_CLEANUP_THREAD_ACTIVE:
                    return
                time.sleep(60)  # 1分間隔でチェック
            
            if _CACHE_CLEANUP_THREAD_ACTIVE:
                cleanup_stats = cleanup_numba_cache(max_age_hours=24.0, verbose=False)
                if cleanup_stats['total_files_removed'] > 0:
                    print(f"🧹 Background Numba cache cleanup: "
                          f"{cleanup_stats['total_files_removed']} files removed, "
                          f"{cleanup_stats['total_size_freed_mb']:.1f}MB freed")
                    
        except Exception as e:
            print(f"⚠️ Background cache cleanup error: {e}")
            # エラー時は1時間後にリトライ（60秒間隔でチェック）
            for _ in range(60):
                if not _CACHE_CLEANUP_THREAD_ACTIVE:
                    return
                time.sleep(60)

def start_automatic_cache_cleanup():
    """自動キャッシュクリーンアップを開始"""
    global _CACHE_CLEANUP_THREAD_ACTIVE
    
    with _CACHE_CLEANUP_LOCK:
        if not _CACHE_CLEANUP_THREAD_ACTIVE:
            import threading
            _CACHE_CLEANUP_THREAD_ACTIVE = True
            # デーモンスレッドとして起動
            cleanup_thread = threading.Thread(
                target=_background_cache_cleanup,
                daemon=True,
                name="numba_cache_cleanup"
            )
            cleanup_thread.start()
            print("🧹 Automatic Numba cache cleanup started")

def stop_automatic_cache_cleanup():
    """自動キャッシュクリーンアップを停止"""
    global _CACHE_CLEANUP_THREAD_ACTIVE
    
    with _CACHE_CLEANUP_LOCK:
        if _CACHE_CLEANUP_THREAD_ACTIVE:
            _CACHE_CLEANUP_THREAD_ACTIVE = False
            print("🧹 Automatic Numba cache cleanup stopped")

def initialize_numba(force_retry=False, verbose=False, enable_auto_cleanup=True):
    """
    Numbaを遅延初期化（複数回安全実行可能）
    
    Args:
        force_retry: 強制的に再初期化を試行
        verbose: 詳細ログを出力
        enable_auto_cleanup: 自動キャッシュクリーンアップを有効化
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
        
        # 自動キャッシュクリーンアップを開始
        if enable_auto_cleanup:
            start_automatic_cache_cleanup()
            # プロセス終了時にクリーンアップを停止
            atexit.register(stop_automatic_cache_cleanup)
        
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

def get_numba_cache_info() -> Dict[str, Any]:
    """Numbaキャッシュ情報を取得"""
    cache_dirs = _find_numba_cache_directories()
    
    cache_info = {
        'cache_directories': [],
        'total_size_mb': 0.0,
        'total_files': 0,
        'cleanup_active': _CACHE_CLEANUP_THREAD_ACTIVE
    }
    
    for cache_dir in cache_dirs:
        size_mb = _get_cache_size_mb(cache_dir)
        file_count = len(list(cache_dir.rglob('*'))) if cache_dir.exists() else 0
        
        cache_info['cache_directories'].append({
            'path': str(cache_dir),
            'size_mb': size_mb,
            'file_count': file_count,
            'exists': cache_dir.exists()
        })
        
        cache_info['total_size_mb'] += size_mb
        cache_info['total_files'] += file_count
    
    return cache_info

def force_numba_cache_cleanup(max_age_hours: float = 1.0) -> Dict[str, Any]:
    """
    Numbaキャッシュを強制クリーンアップ
    
    Args:
        max_age_hours: この時間より古いキャッシュファイルを削除
    
    Returns:
        クリーンアップ統計
    """
    print(f"🧹 Forcing Numba cache cleanup (max_age: {max_age_hours}h)")
    return cleanup_numba_cache(max_age_hours=max_age_hours, verbose=True) 