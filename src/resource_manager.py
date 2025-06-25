#!/usr/bin/env python3
"""
Geocussion-SP リソース管理システム

メモリリーク防止と適切なリソース解放を統一管理するシステム。
音響エンジン、入力ストリーム、その他のリソースの自動クリーンアップを提供します。
"""

import atexit
import threading
import time
import weakref
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, TypeVar, Generic
from contextlib import contextmanager
from enum import Enum, auto

from . import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ResourceState(Enum):
    """リソース状態"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    ACTIVE = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class ResourceInfo:
    """リソース情報"""
    resource_id: str
    resource_type: str
    state: ResourceState = ResourceState.UNINITIALIZED
    created_at: float = field(default_factory=time.perf_counter)
    last_accessed: float = field(default_factory=time.perf_counter)
    cleanup_handlers: List[Callable[[], None]] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    memory_estimate_bytes: int = 0
    
    @property
    def age_seconds(self) -> float:
        """リソースの生存時間"""
        return time.perf_counter() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """最終アクセスからの経過時間"""
        return time.perf_counter() - self.last_accessed
    
    def touch(self):
        """最終アクセス時刻を更新"""
        self.last_accessed = time.perf_counter()


class ManagedResource(ABC):
    """管理対象リソースの抽象基底クラス"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        self._resource_manager: Optional['ResourceManager'] = None
        self._is_registered = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """リソースを初期化"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """リソースを解放"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """メモリ使用量を取得（概算、バイト単位）"""
        pass
    
    @property
    @abstractmethod
    def resource_type(self) -> str:
        """リソースタイプを取得"""
        pass
    
    def register_with_manager(self, manager: 'ResourceManager'):
        """リソースマネージャーに登録"""
        self._resource_manager = manager
        self._is_registered = True
    
    def __enter__(self):
        """コンテキストマネージャー開始"""
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize resource: {self.resource_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        self.cleanup()
    
    def __del__(self):
        """デストラクタ"""
        try:
            if self._is_registered and self._resource_manager:
                self._resource_manager.unregister_resource(self.resource_id)
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in destructor for {self.resource_id}: {e}")


class ResourceManager:
    """統一リソース管理システム"""
    
    def __init__(self, 
                 enable_auto_cleanup: bool = True,
                 cleanup_interval_seconds: float = 30.0,
                 max_idle_time_seconds: float = 300.0):
        """
        初期化
        
        Args:
            enable_auto_cleanup: 自動クリーンアップを有効にするか
            cleanup_interval_seconds: クリーンアップ実行間隔
            max_idle_time_seconds: アイドルリソースの最大保持時間
        """
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.max_idle_time_seconds = max_idle_time_seconds
        
        # リソース管理
        self._resources: Dict[str, ResourceInfo] = {}
        self._resource_instances: Dict[str, Any] = {}  # WeakValueDictionary的な管理
        self._lock = threading.RLock()
        
        # 自動クリーンアップスレッド
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # 統計情報
        self.stats = {
            'total_resources_created': 0,
            'total_resources_cleaned': 0,
            'current_active_resources': 0,
            'memory_usage_estimate_bytes': 0,
            'cleanup_runs': 0,
            'cleanup_errors': 0
        }
        
        if self.enable_auto_cleanup:
            self._start_cleanup_thread()
        
        # アプリケーション終了時のクリーンアップ登録
        atexit.register(self.shutdown_all)
        
        logger.info(f"ResourceManager initialized: auto_cleanup={enable_auto_cleanup}")
    
    def register_resource(self, 
                         resource: ManagedResource,
                         cleanup_handlers: Optional[List[Callable[[], None]]] = None,
                         dependencies: Optional[Set[str]] = None,
                         memory_estimate: int = 0) -> bool:
        """
        リソースを登録
        
        Args:
            resource: 管理対象リソース
            cleanup_handlers: 追加のクリーンアップ処理
            dependencies: 依存関係（先に初期化されるべきリソースID）
            memory_estimate: メモリ使用量推定値
            
        Returns:
            登録成功したかどうか
        """
        with self._lock:
            try:
                resource_id = resource.resource_id
                
                if resource_id in self._resources:
                    logger.warning(f"Resource {resource_id} is already registered")
                    return False
                
                # リソース情報作成
                resource_info = ResourceInfo(
                    resource_id=resource_id,
                    resource_type=resource.resource_type,
                    cleanup_handlers=cleanup_handlers or [],
                    dependencies=dependencies or set(),
                    memory_estimate_bytes=memory_estimate
                )
                
                # 依存関係の逆方向登録
                for dep_id in resource_info.dependencies:
                    if dep_id in self._resources:
                        self._resources[dep_id].dependents.add(resource_id)
                
                self._resources[resource_id] = resource_info
                self._resource_instances[resource_id] = resource
                
                # リソースにマネージャーを登録
                resource.register_with_manager(self)
                
                # 統計更新
                self.stats['total_resources_created'] += 1
                self.stats['current_active_resources'] += 1
                self.stats['memory_usage_estimate_bytes'] += memory_estimate
                
                logger.info(f"Resource registered: {resource_id} ({resource.resource_type})")
                return True
                
            except Exception as e:
                logger.error(f"Error registering resource {resource.resource_id}: {e}")
                return False
    
    def unregister_resource(self, resource_id: str) -> bool:
        """
        リソースを登録解除
        
        Args:
            resource_id: リソースID
            
        Returns:
            解除成功したかどうか
        """
        with self._lock:
            try:
                if resource_id not in self._resources:
                    return False
                
                resource_info = self._resources[resource_id]
                
                # 依存関係の削除
                for dep_id in resource_info.dependencies:
                    if dep_id in self._resources:
                        self._resources[dep_id].dependents.discard(resource_id)
                
                for dependent_id in resource_info.dependents:
                    if dependent_id in self._resources:
                        self._resources[dependent_id].dependencies.discard(resource_id)
                
                # クリーンアップ実行
                self._cleanup_resource(resource_id)
                
                # 削除
                del self._resources[resource_id]
                if resource_id in self._resource_instances:
                    del self._resource_instances[resource_id]
                
                # 統計更新
                self.stats['current_active_resources'] -= 1
                self.stats['memory_usage_estimate_bytes'] -= resource_info.memory_estimate_bytes
                
                logger.info(f"Resource unregistered: {resource_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error unregistering resource {resource_id}: {e}")
                return False
    
    def get_resource(self, resource_id: str) -> Optional[Any]:
        """リソースインスタンスを取得"""
        with self._lock:
            if resource_id in self._resource_instances:
                # アクセス時刻更新
                if resource_id in self._resources:
                    self._resources[resource_id].touch()
                return self._resource_instances[resource_id]
            return None
    
    def cleanup_resource(self, resource_id: str) -> bool:
        """指定されたリソースをクリーンアップ"""
        with self._lock:
            return self._cleanup_resource(resource_id)
    
    def _cleanup_resource(self, resource_id: str) -> bool:
        """内部クリーンアップ処理"""
        try:
            if resource_id not in self._resources:
                return False
            
            resource_info = self._resources[resource_id]
            resource_instance = self._resource_instances.get(resource_id)
            
            # 依存するリソースを先にクリーンアップ
            for dependent_id in list(resource_info.dependents):
                if dependent_id in self._resources:
                    logger.info(f"Cleaning up dependent resource: {dependent_id}")
                    self._cleanup_resource(dependent_id)
            
            # カスタムクリーンアップハンドラー実行
            for handler in resource_info.cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Error in cleanup handler for {resource_id}: {e}")
            
            # リソース自体のクリーンアップ
            if resource_instance and hasattr(resource_instance, 'cleanup'):
                try:
                    resource_instance.cleanup()
                    logger.info(f"Resource cleaned up: {resource_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_id}: {e}")
            
            # 状態更新
            resource_info.state = ResourceState.STOPPED
            
            # 統計更新
            self.stats['total_resources_cleaned'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error during resource cleanup {resource_id}: {e}")
            self.stats['cleanup_errors'] += 1
            return False
    
    def cleanup_idle_resources(self) -> int:
        """アイドル状態のリソースをクリーンアップ"""
        cleaned_count = 0
        
        with self._lock:
            idle_resources = []
            
            for resource_id, resource_info in self._resources.items():
                if (resource_info.idle_seconds > self.max_idle_time_seconds and
                    resource_info.state not in [ResourceState.STOPPING, ResourceState.STOPPED]):
                    idle_resources.append(resource_id)
            
            for resource_id in idle_resources:
                if self._cleanup_resource(resource_id):
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} idle resources")
        
        return cleaned_count
    
    def shutdown_all(self):
        """全リソースを強制シャットダウン"""
        logger.info("Shutting down all resources...")
        
        # 自動クリーンアップ停止
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
        
        with self._lock:
            # 依存関係を考慮した順序でクリーンアップ
            remaining_resources = set(self._resources.keys())
            
            while remaining_resources:
                # 依存されていないリソースから削除
                candidates = []
                for resource_id in remaining_resources:
                    resource_info = self._resources[resource_id]
                    if not any(dep in remaining_resources for dep in resource_info.dependents):
                        candidates.append(resource_id)
                
                if not candidates:
                    # 循環依存の場合は強制クリーンアップ
                    candidates = list(remaining_resources)
                
                for resource_id in candidates:
                    self._cleanup_resource(resource_id)
                    remaining_resources.discard(resource_id)
        
        # ガベージコレクション実行
        gc.collect()
        
        logger.info("All resources shutdown complete")
    
    def _start_cleanup_thread(self):
        """自動クリーンアップスレッドを開始"""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval_seconds):
                try:
                    self.cleanup_idle_resources()
                    self.stats['cleanup_runs'] += 1
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
                    self.stats['cleanup_errors'] += 1
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info("Auto cleanup thread started")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'active_resources': {
                    resource_id: {
                        'type': info.resource_type,
                        'state': info.state.name,
                        'age_seconds': info.age_seconds,
                        'idle_seconds': info.idle_seconds,
                        'memory_estimate_bytes': info.memory_estimate_bytes
                    }
                    for resource_id, info in self._resources.items()
                },
                'total_estimated_memory_mb': self.stats['memory_usage_estimate_bytes'] / (1024 * 1024)
            })
        return stats
    
    @contextmanager
    def managed_resource(self, resource: ManagedResource, **kwargs):
        """コンテキストマネージャーでリソース管理"""
        try:
            if self.register_resource(resource, **kwargs):
                if resource.initialize():
                    yield resource
                else:
                    raise RuntimeError(f"Failed to initialize resource: {resource.resource_id}")
            else:
                raise RuntimeError(f"Failed to register resource: {resource.resource_id}")
        finally:
            self.unregister_resource(resource.resource_id)


# グローバルリソースマネージャー
_global_resource_manager: Optional[ResourceManager] = None
_manager_lock = threading.Lock()

def get_resource_manager() -> ResourceManager:
    """グローバルリソースマネージャーを取得"""
    global _global_resource_manager
    if _global_resource_manager is None:
        with _manager_lock:
            if _global_resource_manager is None:
                _global_resource_manager = ResourceManager()
    return _global_resource_manager

def shutdown_resource_manager():
    """グローバルリソースマネージャーをシャットダウン"""
    global _global_resource_manager
    if _global_resource_manager is not None:
        _global_resource_manager.shutdown_all()
        _global_resource_manager = None

@contextmanager
def managed_resource(resource: ManagedResource, **kwargs):
    """グローバルマネージャーでリソース管理"""
    manager = get_resource_manager()
    with manager.managed_resource(resource, **kwargs) as managed:
        yield managed 