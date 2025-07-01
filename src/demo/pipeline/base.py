#!/usr/bin/env python3
"""
パイプラインステージの基底クラス

各ステージが実装すべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ... import get_logger


@dataclass
class StageResult:
    """ステージ処理結果の基底クラス"""
    success: bool = True
    error_message: Optional[str] = None
    data: Optional[Any] = None


class PipelineStage(ABC):
    """パイプラインステージの抽象基底クラス"""
    
    def __init__(self, config: Any) -> None:
        """
        初期化
        
        Args:
            config: ステージ固有の設定
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        ステージの初期化
        
        Returns:
            初期化成功時はTrue
        """
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> StageResult:
        """
        ステージの処理を実行
        
        Args:
            **kwargs: ステージ固有の入力データ
            
        Returns:
            処理結果
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        pass
    
    def is_initialized(self) -> bool:
        """初期化済みかどうかを返す"""
        return self._initialized
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        設定を動的に更新
        
        Args:
            new_config: 更新する設定項目の辞書
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")