"""
非同期・デッドロックフリーリソース管理器

perf-006: 共有リソースクリーンアップスレッドによるデッドロック / フリーズを解決
asyncio + queue ベースのチャンクドクリーンアップ実装
"""

import asyncio
import time
import atexit
import weakref
from typing import Dict, Any, Optional, List, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import gc

from . import get_logger

logger = get_logger(__name__)


class AsyncResourceState(Enum):
    """非同期リソース状態"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    ACTIVE = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()
    CLEANUP_QUEUED = auto()


@dataclass
class CleanupTask:
    """クリーンアップタスク"""
    resource_id: str
    resource_type: str
    cleanup_function: Callable[[], None]
    priority: int = 0  # 高優先度 = 低数値
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.perf_counter)
    
    def __lt__(self, other: 'CleanupTask') -> bool:
        return self.priority < other.priority


@dataclass 
class AsyncResourceInfo:
    """非同期リソース情報"""
    resource_id: str
    resource_type: str
    state: AsyncResourceState = AsyncResourceState.UNINITIALIZED
    created_at: float = field(default_factory=time.perf_counter)
    last_accessed: float = field(default_factory=time.perf_counter)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    memory_estimate_bytes: int = 0
    cleanup_timeout_sec: float = 5.0
    
    @property
    def age_seconds(self) -> float:
        return time.perf_counter() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        return time.perf_counter() - self.last_accessed
    
    def touch(self) -> None:
        self.last_accessed = time.perf_counter()


class AsyncManagedResource(ABC):
    """非同期管理リソース基底クラス"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        self._manager: Optional['AsyncResourceManager'] = None
        self._cleanup_lock = asyncio.Lock()
        self._is_cleaned = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """リソース初期化（非同期）"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """リソースクリーンアップ（非同期）"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """メモリ使用量取得"""
        pass
    
    @property
    @abstractmethod
    def resource_type(self) -> str:
        """リソースタイプ"""
        pass
    
    def register_with_manager(self, manager: 'AsyncResourceManager') -> None:
        """マネージャーに登録"""
        self._manager = manager
    
    async def safe_cleanup(self) -> bool:
        """安全なクリーンアップ（重複呼び出し防止）"""
        async with self._cleanup_lock:
            if self._is_cleaned:
                return True
            
            try:
                result = await self.cleanup()
                self._is_cleaned = True
                return result
            except Exception as e:
                logger.error(f"Error in safe cleanup for {self.resource_id}: {e}")
                return False
    
    def __del__(self):
        """デストラクタ（ノンブロッキング）"""
        if hasattr(self, '_manager') and self._manager and not self._is_cleaned:
            # 非同期クリーンアップをキューに追加
            self._manager.queue_cleanup_nowait(self.resource_id, self.cleanup_sync)
    
    def cleanup_sync(self) -> None:
        """同期クリーンアップ（asyncio外からの呼び出し用）"""
        try:
            # 可能な場合は非同期版を実行
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既存のイベントループがある場合はタスクとして追加
                loop.create_task(self.safe_cleanup())
            else:
                # 新しいイベントループで実行
                asyncio.run(self.safe_cleanup())
        except Exception:
            # フォールバック: 基本的なクリーンアップのみ
            if hasattr(self, '_basic_cleanup'):
                self._basic_cleanup()


class AsyncResourceManager:
    """非同期リソースマネージャー（デッドロックフリー）"""
    
    def __init__(
        self,
        cleanup_interval_sec: float = 1.0,
        max_idle_time_sec: float = 300.0,
        max_concurrent_cleanups: int = 3,
        chunk_size: int = 5,
        enable_background_gc: bool = True
    ):
        self.cleanup_interval_sec = cleanup_interval_sec
        self.max_idle_time_sec = max_idle_time_sec
        self.max_concurrent_cleanups = max_concurrent_cleanups
        self.chunk_size = chunk_size
        self.enable_background_gc = enable_background_gc
        
        # リソース管理
        self._resources: Dict[str, AsyncResourceInfo] = {}
        self._resource_instances: Dict[str, weakref.ReferenceType] = {}
        
        # 非同期処理
        self._cleanup_queue = asyncio.PriorityQueue()
        self._cleanup_semaphore = asyncio.Semaphore(max_concurrent_cleanups)
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # スレッドセーフ同期クリーンアップ用
        self._sync_cleanup_queue: queue.Queue[CleanupTask] = queue.Queue()
        self._sync_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ResourceCleanup")
        
        # 統計
        self.stats = {
            'total_resources_created': 0,
            'total_resources_cleaned': 0,
            'current_active_resources': 0,
            'memory_usage_estimate_bytes': 0,
            'cleanup_queue_size': 0,
            'async_cleanups': 0,
            'sync_cleanups': 0,
            'cleanup_errors': 0,
            'deadlock_preventions': 0
        }
        
        # バックグラウンドタスク開始
        self._start_background_tasks()
        
        # 終了時処理登録
        atexit.register(self.shutdown_sync)
        
        logger.info("AsyncResourceManager initialized")
    
    async def register_resource(
        self,
        resource: AsyncManagedResource,
        dependencies: Optional[Set[str]] = None,
        memory_estimate: int = 0,
        cleanup_timeout: float = 5.0
    ) -> bool:
        """リソースを非同期登録"""
        try:
            resource_id = resource.resource_id
            
            if resource_id in self._resources:
                logger.warning(f"Resource {resource_id} already registered")
                return False
            
            # リソース情報作成
            resource_info = AsyncResourceInfo(
                resource_id=resource_id,
                resource_type=resource.resource_type,
                dependencies=dependencies or set(),
                memory_estimate_bytes=memory_estimate,
                cleanup_timeout_sec=cleanup_timeout
            )
            
            # 依存関係の逆方向登録
            for dep_id in resource_info.dependencies:
                if dep_id in self._resources:
                    self._resources[dep_id].dependents.add(resource_id)
            
            # 登録
            self._resources[resource_id] = resource_info
            self._resource_instances[resource_id] = weakref.ref(
                resource, 
                lambda ref: self.queue_cleanup_nowait(resource_id, lambda: None)
            )
            
            # マネージャー登録
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
    
    async def unregister_resource(self, resource_id: str) -> bool:
        """リソースを非同期登録解除"""
        try:
            if resource_id not in self._resources:
                return False
            
            resource_info = self._resources[resource_id]
            
            # 依存するリソースを先にクリーンアップ
            dependent_cleanups = []
            for dependent_id in list(resource_info.dependents):
                if dependent_id in self._resources:
                    dependent_cleanups.append(self.cleanup_resource(dependent_id))
            
            if dependent_cleanups:
                await asyncio.gather(*dependent_cleanups, return_exceptions=True)
            
            # 依存関係削除
            for dep_id in resource_info.dependencies:
                if dep_id in self._resources:
                    self._resources[dep_id].dependents.discard(resource_id)
            
            # メインリソースクリーンアップ
            await self.cleanup_resource(resource_id)
            
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
    
    async def cleanup_resource(self, resource_id: str) -> bool:
        """リソースを非同期クリーンアップ"""
        if resource_id not in self._resources:
            return False
        
        resource_info = self._resources[resource_id]
        
        # 既にクリーンアップ中/完了の場合はスキップ
        if resource_info.state in [AsyncResourceState.CLEANUP_QUEUED, 
                                  AsyncResourceState.STOPPING, 
                                  AsyncResourceState.STOPPED]:
            return True
        
        resource_info.state = AsyncResourceState.CLEANUP_QUEUED
        
        # 非同期クリーンアップタスクをキューに追加
        cleanup_task = CleanupTask(
            resource_id=resource_id,
            resource_type=resource_info.resource_type,
            cleanup_function=lambda: self._cleanup_resource_impl(resource_id),
            priority=len(resource_info.dependents)  # 依存が多いほど高優先度
        )
        
        await self._cleanup_queue.put(cleanup_task)
        self.stats['cleanup_queue_size'] = self._cleanup_queue.qsize()
        
        return True
    
    async def _cleanup_resource_impl(self, resource_id: str) -> bool:
        """リソースクリーンアップ実装"""
        async with self._cleanup_semaphore:
            try:
                if resource_id not in self._resources:
                    return False
                
                resource_info = self._resources[resource_id]
                resource_info.state = AsyncResourceState.STOPPING
                
                # リソースインスタンス取得
                resource_ref = self._resource_instances.get(resource_id)
                resource = resource_ref() if resource_ref else None
                
                if resource:
                    # タイムアウト付きクリーンアップ
                    try:
                        await asyncio.wait_for(
                            resource.safe_cleanup(),
                            timeout=resource_info.cleanup_timeout_sec
                        )
                        logger.info(f"Resource cleaned up: {resource_id}")
                    except asyncio.TimeoutError:
                        logger.warning(f"Cleanup timeout for resource: {resource_id}")
                        self.stats['cleanup_errors'] += 1
                        # 強制的にクリーンアップ済みとマーク
                        resource._is_cleaned = True
                
                resource_info.state = AsyncResourceState.STOPPED
                self.stats['total_resources_cleaned'] += 1
                self.stats['async_cleanups'] += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource_id}: {e}")
                self.stats['cleanup_errors'] += 1
                return False
    
    def queue_cleanup_nowait(self, resource_id: str, cleanup_func: Callable[[], None]) -> None:
        """同期的なクリーンアップタスクをキューに追加（ノンブロッキング）"""
        try:
            cleanup_task = CleanupTask(
                resource_id=resource_id,
                resource_type="sync",
                cleanup_function=cleanup_func,
                priority=100  # 低優先度
            )
            
            self._sync_cleanup_queue.put_nowait(cleanup_task)
            
        except queue.Full:
            logger.warning(f"Sync cleanup queue full, dropping task for {resource_id}")
    
    async def cleanup_idle_resources(self) -> int:
        """アイドルリソースの一括クリーンアップ"""
        idle_resources = []
        
        for resource_id, resource_info in self._resources.items():
            if (resource_info.idle_seconds > self.max_idle_time_sec and
                resource_info.state == AsyncResourceState.READY):
                idle_resources.append(resource_id)
        
        # チャンク単位でクリーンアップ
        cleaned_count = 0
        for i in range(0, len(idle_resources), self.chunk_size):
            chunk = idle_resources[i:i + self.chunk_size]
            cleanup_tasks = [self.cleanup_resource(rid) for rid in chunk]
            
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            cleaned_count += sum(1 for r in results if r is True)
            
            # CPU時間を他のタスクに譲る
            await asyncio.sleep(0.01)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} idle resources")
        
        return cleaned_count
    
    def _start_background_tasks(self) -> None:
        """バックグラウンドタスクを開始"""
        # 非同期クリーンアップワーカー
        task1 = asyncio.create_task(self._async_cleanup_worker())
        self._background_tasks.add(task1)
        task1.add_done_callback(self._background_tasks.discard)
        
        # 定期クリーンアップタスク
        task2 = asyncio.create_task(self._periodic_cleanup_worker())
        self._background_tasks.add(task2)
        task2.add_done_callback(self._background_tasks.discard)
        
        # 同期クリーンアップワーカー（別スレッド）
        self._sync_executor.submit(self._sync_cleanup_worker)
    
    async def _async_cleanup_worker(self) -> None:
        """非同期クリーンアップワーカー"""
        while not self._shutdown_event.is_set():
            try:
                # タイムアウト付きでタスク取得
                task = await asyncio.wait_for(
                    self._cleanup_queue.get(),
                    timeout=1.0
                )
                
                # クリーンアップ実行
                await task.cleanup_function()
                
                # キュー完了通知
                self._cleanup_queue.task_done()
                self.stats['cleanup_queue_size'] = self._cleanup_queue.qsize()
                
            except asyncio.TimeoutError:
                # タイムアウトは正常（定期的なチェック）
                continue
            except Exception as e:
                logger.error(f"Error in async cleanup worker: {e}")
                await asyncio.sleep(0.1)
    
    async def _periodic_cleanup_worker(self) -> None:
        """定期クリーンアップワーカー"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval_sec)
                
                # アイドルリソースクリーンアップ
                await self.cleanup_idle_resources()
                
                # ガベージコレクション（必要に応じて）
                if self.enable_background_gc:
                    collected = gc.collect()
                    if collected > 0:
                        logger.debug(f"GC collected {collected} objects")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup worker: {e}")
                await asyncio.sleep(1.0)
    
    def _sync_cleanup_worker(self) -> None:
        """同期クリーンアップワーカー（別スレッド）"""
        while not self._shutdown_event.is_set():
            try:
                # タイムアウト付きでタスク取得
                task = self._sync_cleanup_queue.get(timeout=1.0)
                
                # クリーンアップ実行
                task.cleanup_function()
                
                self.stats['sync_cleanups'] += 1
                self._sync_cleanup_queue.task_done()
                
            except queue.Empty:
                # タイムアウトは正常
                continue
            except Exception as e:
                logger.error(f"Error in sync cleanup worker: {e}")
                time.sleep(0.1)
    
    async def shutdown(self) -> None:
        """非同期シャットダウン"""
        logger.info("Shutting down AsyncResourceManager...")
        
        # シャットダウンフラグ設定
        self._shutdown_event.set()
        
        # 全リソースクリーンアップ
        cleanup_tasks = []
        for resource_id in list(self._resources.keys()):
            cleanup_tasks.append(self.cleanup_resource(resource_id))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # バックグラウンドタスク停止
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # 同期クリーンアップ停止
        self._sync_executor.shutdown(wait=True)
        
        logger.info("AsyncResourceManager shutdown complete")
    
    def shutdown_sync(self) -> None:
        """同期シャットダウン（atexit用）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既存のループがある場合はタスクとして追加
                loop.create_task(self.shutdown())
            else:
                # 新しいループで実行
                asyncio.run(self.shutdown())
        except Exception as e:
            logger.error(f"Error in sync shutdown: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        stats = self.stats.copy()
        stats['cleanup_queue_size'] = self._cleanup_queue.qsize()
        stats['background_tasks_count'] = len(self._background_tasks)
        return stats


# グローバルインスタンス
_global_async_resource_manager: Optional[AsyncResourceManager] = None


async def get_async_resource_manager() -> AsyncResourceManager:
    """グローバル非同期リソースマネージャーを取得"""
    global _global_async_resource_manager
    if _global_async_resource_manager is None:
        _global_async_resource_manager = AsyncResourceManager()
    return _global_async_resource_manager


@asynccontextmanager
async def async_managed_resource(resource: AsyncManagedResource, **kwargs) -> AsyncManagedResource:
    """非同期管理リソースコンテキストマネージャー"""
    manager = await get_async_resource_manager()
    
    try:
        await manager.register_resource(resource, **kwargs)
        await resource.initialize()
        yield resource
    finally:
        await manager.unregister_resource(resource.resource_id) 