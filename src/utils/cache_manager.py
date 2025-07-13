"""
統一キャッシュマネージャー

キャッシュ拡張機能（推定FPS+10%）を実現する統一キャッシュ管理システム
- 自動キャッシュクリーンアップ
- メモリ使用量監視
- 統計収集とレポート
- 設定ベースのキャッシュ有効期間延長
"""

import time
import threading
import weakref
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import logging

from ..config import get_config
from .. import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def touch(self):
        """アクセス記録を更新"""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """キャッシュ統計"""
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_cleanups: int = 0
    memory_usage_mb: float = 0.0
    cache_size: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    
    def update_hit_rate(self):
        """ヒット率を更新"""
        total_requests = self.total_hits + self.total_misses
        self.hit_rate = self.total_hits / total_requests if total_requests > 0 else 0.0


class ExtendedCacheManager:
    """拡張キャッシュマネージャー"""
    
    def __init__(self, name: str, validity_time: float, size_limit: int):
        self.name = name
        self.validity_time = validity_time
        self.size_limit = size_limit
        
        # キャッシュストレージ
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # 統計
        self.stats = CacheStats()
        
        # 自動クリーンアップ
        self._last_cleanup = time.time()
        
        # 設定取得
        self.config = get_config()
        
        logger.info(f"ExtendedCacheManager '{name}' initialized: validity={validity_time}s, size_limit={size_limit}")
    
    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        start_time = time.perf_counter()
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats.total_misses += 1
                return None
            
            current_time = time.time()
            
            # 有効期限チェック
            if current_time - entry.timestamp > self.validity_time:
                del self._cache[key]
                self.stats.total_misses += 1
                self.stats.total_evictions += 1
                return None
            
            # ヒット記録
            entry.touch()
            self.stats.total_hits += 1
            self.stats.update_hit_rate()
            
            # アクセス時間統計更新
            access_time_ms = (time.perf_counter() - start_time) * 1000
            self.stats.avg_access_time_ms = (
                (self.stats.avg_access_time_ms * (self.stats.total_hits - 1) + access_time_ms) / 
                self.stats.total_hits
            )
            
            return entry.value
    
    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        """キャッシュに値を保存"""
        with self._lock:
            current_time = time.time()
            
            # サイズ制限チェック
            if len(self._cache) >= self.size_limit:
                self._evict_lru()
            
            # エントリ作成
            entry = CacheEntry(
                value=value,
                timestamp=current_time,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._update_memory_usage()
            
            # 定期クリーンアップ
            if self.config.cache.auto_cache_cleanup_interval > 0:
                if current_time - self._last_cleanup > self.config.cache.auto_cache_cleanup_interval:
                    cleanup_thread = threading.Thread(
                        target=self._background_cleanup,
                        daemon=True,
                        name=f"cache_cleanup_{self.name}"
                    )
                    cleanup_thread.start()
    
    def _evict_lru(self) -> None:
        """LRU方式で最も古いエントリを削除"""
        if not self._cache:
            return
        
        # 最も古いアクセス時間のエントリを見つける
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_access)
        del self._cache[oldest_key]
        self.stats.total_evictions += 1
    
    def _background_cleanup(self) -> None:
        """バックグラウンドクリーンアップ"""
        try:
            current_time = time.time()
            cleanup_count = 0
            
            with self._lock:
                # 期限切れエントリを削除
                expired_keys = [
                    key for key, entry in self._cache.items()
                    if current_time - entry.timestamp > self.validity_time
                ]
                
                for key in expired_keys:
                    del self._cache[key]
                    cleanup_count += 1
                
                self.stats.total_cleanups += 1
                self._last_cleanup = current_time
                self._update_memory_usage()
            
            if cleanup_count > 0:
                logger.debug(f"Cache '{self.name}' cleaned up {cleanup_count} expired entries")
            
            # メモリ使用量チェック
            if self.stats.memory_usage_mb > self.config.cache.cache_memory_threshold_mb:
                self._force_cleanup()
                
        except Exception as e:
            logger.warning(f"Cache cleanup error for '{self.name}': {e}")
    
    def _force_cleanup(self) -> None:
        """強制クリーンアップ（メモリ使用量が閾値を超えた場合）"""
        with self._lock:
            # アクセス回数が少ないエントリを優先的に削除
            if len(self._cache) > self.size_limit // 2:
                sorted_keys = sorted(
                    self._cache.keys(), 
                    key=lambda k: self._cache[k].access_count
                )
                
                # 半分のエントリを削除
                keys_to_remove = sorted_keys[:len(sorted_keys) // 2]
                for key in keys_to_remove:
                    del self._cache[key]
                    self.stats.total_evictions += 1
                
                self._update_memory_usage()
                logger.info(f"Cache '{self.name}' force cleanup: removed {len(keys_to_remove)} entries")
    
    def _update_memory_usage(self) -> None:
        """メモリ使用量を更新"""
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        self.stats.memory_usage_mb = total_size / (1024 * 1024)
        self.stats.cache_size = len(self._cache)
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats()
            logger.info(f"Cache '{self.name}' cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self._lock:
            return {
                'name': self.name,
                'total_hits': self.stats.total_hits,
                'total_misses': self.stats.total_misses,
                'total_evictions': self.stats.total_evictions,
                'total_cleanups': self.stats.total_cleanups,
                'memory_usage_mb': self.stats.memory_usage_mb,
                'cache_size': self.stats.cache_size,
                'hit_rate': self.stats.hit_rate,
                'avg_access_time_ms': self.stats.avg_access_time_ms,
                'validity_time': self.validity_time,
                'size_limit': self.size_limit
            }
    
    def shutdown(self) -> None:
        """シャットダウン処理"""
        self.clear()


class GlobalCacheManager:
    """グローバルキャッシュマネージャー"""
    
    _instance: Optional['GlobalCacheManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.managers: Dict[str, ExtendedCacheManager] = {}
        self.config = get_config()
        
        # 設定に基づいてキャッシュマネージャーを初期化
        self._initialize_managers()
        
        # 統計レポート用のスレッド
        if self.config.cache.enable_cache_statistics:
            self._start_stats_reporting()
    
    @classmethod
    def get_instance(cls) -> 'GlobalCacheManager':
        """シングルトンインスタンスを取得"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _initialize_managers(self) -> None:
        """設定に基づいてキャッシュマネージャーを初期化"""
        config = self.config.cache
        
        # LODメッシュキャッシュ
        self.managers['lod_mesh'] = ExtendedCacheManager(
            'lod_mesh',
            config.lod_cache_validity_time,
            config.lod_cache_size_limit
        )
        
        # 衝突検索キャッシュ
        self.managers['collision_search'] = ExtendedCacheManager(
            'collision_search',
            config.collision_cache_validity_time,
            config.collision_cache_size_limit
        )
        
        # 曲率計算キャッシュ
        self.managers['curvature'] = ExtendedCacheManager(
            'curvature',
            config.curvature_cache_validity_time,
            config.curvature_cache_size_limit
        )
        
        # メッシュパイプラインキャッシュ
        self.managers['pipeline'] = ExtendedCacheManager(
            'pipeline',
            config.pipeline_cache_validity_time,
            config.pipeline_cache_size_limit
        )
        
        logger.info(f"Initialized {len(self.managers)} cache managers")
    
    def get_manager(self, name: str) -> Optional[ExtendedCacheManager]:
        """指定名のキャッシュマネージャーを取得"""
        return self.managers.get(name)
    
    def _start_stats_reporting(self) -> None:
        """統計レポートを開始"""
        def report_stats():
            while True:
                try:
                    time.sleep(60)  # 1分間隔でレポート
                    self._report_cache_stats()
                except Exception as e:
                    logger.warning(f"Cache stats reporting error: {e}")
        
        # デーモンスレッドとして起動
        stats_thread = threading.Thread(target=report_stats, daemon=True, name="cache_stats_reporter")
        stats_thread.start()
    
    def _report_cache_stats(self) -> None:
        """キャッシュ統計をレポート"""
        total_hits = 0
        total_misses = 0
        total_memory_mb = 0.0
        
        logger.info("=== Cache Statistics Report ===")
        
        for name, manager in self.managers.items():
            stats = manager.get_stats()
            total_hits += stats['total_hits']
            total_misses += stats['total_misses']
            total_memory_mb += stats['memory_usage_mb']
            
            logger.info(f"Cache '{name}': "
                       f"hits={stats['total_hits']}, "
                       f"misses={stats['total_misses']}, "
                       f"hit_rate={stats['hit_rate']:.1%}, "
                       f"size={stats['cache_size']}, "
                       f"memory={stats['memory_usage_mb']:.1f}MB")
        
        overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        logger.info(f"Overall: hit_rate={overall_hit_rate:.1%}, total_memory={total_memory_mb:.1f}MB")
        logger.info("=== End Cache Report ===")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """全キャッシュマネージャーの統計を取得"""
        return {name: manager.get_stats() for name, manager in self.managers.items()}
    
    def clear_all(self) -> None:
        """全キャッシュをクリア"""
        for manager in self.managers.values():
            manager.clear()
        logger.info("All caches cleared")
    
    def shutdown(self) -> None:
        """シャットダウン処理"""
        for manager in self.managers.values():
            manager.shutdown()


def get_cache_manager(name: str) -> Optional[ExtendedCacheManager]:
    """指定名のキャッシュマネージャーを取得"""
    return GlobalCacheManager.get_instance().get_manager(name)


def get_global_cache_stats() -> Dict[str, Any]:
    """グローバルキャッシュ統計を取得"""
    return GlobalCacheManager.get_instance().get_all_stats()


def clear_all_caches() -> None:
    """全キャッシュをクリア"""
    GlobalCacheManager.get_instance().clear_all() 