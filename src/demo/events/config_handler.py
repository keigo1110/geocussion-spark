#!/usr/bin/env python3
"""
設定変更イベントハンドラー

キーボードイベントに応じて設定を変更するハンドラー
"""

from typing import Optional
from .base import EventHandler, EventType, Event
from .ui_events import KeyPressedEvent, KeyCode
from .pipeline_events import ConfigChangedEvent
from ... import get_logger


class ConfigurationEventHandler(EventHandler):
    """
    設定変更イベントハンドラー
    
    キーボードイベントを受け取って、対応する設定変更を行う
    """
    
    def __init__(self, pipeline_config, pipeline) -> None:
        """
        初期化
        
        Args:
            pipeline_config: パイプライン設定オブジェクト
            pipeline: パイプラインインスタンス
        """
        self.pipeline_config = pipeline_config
        self.pipeline = pipeline
        self.logger = get_logger(__name__)
        
        # イベントディスパッチャー
        from .base import get_event_dispatcher
        self.event_dispatcher = get_event_dispatcher()
        
    def handle_event(self, event: Event) -> None:
        """
        イベントハンドラー
        
        Args:
            event: 処理するイベント
        """
        if event.event_type != EventType.KEY_PRESSED:
            return
            
        key_event = event
        key = key_event.key_code
        
        # メッシュ生成トグル
        if key == ord('m') or key == ord('M'):
            old_value = self.pipeline_config.enable_mesh_generation
            self.pipeline_config.enable_mesh_generation = not old_value
            self.pipeline.update_config(self.pipeline_config)
            status = "有効" if self.pipeline_config.enable_mesh_generation else "無効"
            print(f"メッシュ生成: {status}")
            
            # 設定変更イベント発行
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="enable_mesh_generation",
                old_value=old_value,
                new_value=self.pipeline_config.enable_mesh_generation
            ))
        
        # 衝突検出トグル
        elif key == ord('c') or key == ord('C'):
            old_value = self.pipeline_config.enable_collision_detection
            self.pipeline_config.enable_collision_detection = not old_value
            self.pipeline.update_config(self.pipeline_config)
            status = "有効" if self.pipeline_config.enable_collision_detection else "無効"
            print(f"衝突検出: {status}")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="enable_collision_detection",
                old_value=old_value,
                new_value=self.pipeline_config.enable_collision_detection
            ))
        
        # 衝突可視化トグル
        elif key == ord('v') or key == ord('V'):
            old_value = self.pipeline_config.enable_collision_visualization
            self.pipeline_config.enable_collision_visualization = not old_value
            status = "有効" if self.pipeline_config.enable_collision_visualization else "無効"
            print(f"衝突可視化: {status}")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="enable_collision_visualization",
                old_value=old_value,
                new_value=self.pipeline_config.enable_collision_visualization
            ))
        
        # メッシュ強制更新
        elif key == ord('n') or key == ord('N'):
            print("メッシュ強制更新...")
            self.pipeline.force_mesh_update()
        
        # 球半径調整
        elif key == ord('+') or key == ord('='):
            old_value = self.pipeline_config.sphere_radius
            self.pipeline_config.sphere_radius = min(old_value + 0.01, 0.2)
            self.pipeline.update_config(self.pipeline_config)
            print(f"球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="sphere_radius",
                old_value=old_value,
                new_value=self.pipeline_config.sphere_radius
            ))
        
        elif key == ord('-') or key == ord('_'):
            old_value = self.pipeline_config.sphere_radius
            self.pipeline_config.sphere_radius = max(old_value - 0.01, 0.01)
            self.pipeline.update_config(self.pipeline_config)
            print(f"球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="sphere_radius",
                old_value=old_value,
                new_value=self.pipeline_config.sphere_radius
            ))
        
        # 音響合成トグル
        elif key == ord('a') or key == ord('A'):
            old_value = self.pipeline_config.enable_audio_synthesis
            self.pipeline_config.enable_audio_synthesis = not old_value
            self.pipeline.update_config(self.pipeline_config)
            status = "有効" if self.pipeline_config.enable_audio_synthesis else "無効"
            print(f"音響合成: {status}")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="enable_audio_synthesis",
                old_value=old_value,
                new_value=self.pipeline_config.enable_audio_synthesis
            ))
        
        # 音量調整
        elif key == ord('1'):
            old_value = self.pipeline_config.audio_master_volume
            self.pipeline_config.audio_master_volume = max(0.0, old_value - 0.1)
            self.pipeline.update_config(self.pipeline_config)
            print(f"音量: {int(self.pipeline_config.audio_master_volume * 100)}%")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="audio_master_volume",
                old_value=old_value,
                new_value=self.pipeline_config.audio_master_volume
            ))
        
        elif key == ord('2'):
            old_value = self.pipeline_config.audio_master_volume
            self.pipeline_config.audio_master_volume = min(1.0, old_value + 0.1)
            self.pipeline.update_config(self.pipeline_config)
            print(f"音量: {int(self.pipeline_config.audio_master_volume * 100)}%")
            
            self.event_dispatcher.publish(ConfigChangedEvent(
                config_key="audio_master_volume",
                old_value=old_value,
                new_value=self.pipeline_config.audio_master_volume
            ))
    
    def can_handle(self, event_type: EventType) -> bool:
        """このハンドラーがキー押下イベントを処理できることを示す"""
        return event_type == EventType.KEY_PRESSED