#!/usr/bin/env python3
"""
音響ステージ: 衝突イベントからの音響合成

衝突イベントを音響パラメータにマッピングし、リアルタイムで音を生成します。
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from .base import PipelineStage, StageResult
from ...collision.events import CollisionEvent
from ...sound.mapping import AudioMapper, ScaleType, InstrumentType
from ...sound.synth import AudioSynthesizer, create_audio_synthesizer
from ...sound.voice_mgr import VoiceManager, create_voice_manager, allocate_and_play, StealStrategy


@dataclass
class AudioStageConfig:
    """音響ステージの設定"""
    enable_audio_synthesis: bool = True
    # 音響パラメータ
    audio_scale: ScaleType = ScaleType.PENTATONIC
    audio_instrument: InstrumentType = InstrumentType.MARIMBA
    audio_polyphony: int = 16
    audio_master_volume: float = 0.7
    # 音声管理
    voice_steal_strategy: StealStrategy = StealStrategy.OLDEST
    # 空間音響
    enable_spatial_audio: bool = True
    listener_position: np.ndarray = None
    
    def __post_init__(self):
        if self.listener_position is None:
            self.listener_position = np.array([0.0, 0.0, -1.0])


@dataclass
class AudioStageResult(StageResult):
    """音響ステージの処理結果"""
    active_voices: int = 0
    notes_triggered: List[int] = None
    
    def __post_init__(self):
        if self.notes_triggered is None:
            self.notes_triggered = []


class AudioStage(PipelineStage):
    """音響ステージの実装"""
    
    def __init__(self, config: AudioStageConfig) -> None:
        """
        初期化
        
        Args:
            config: 音響ステージ設定
        """
        super().__init__(config)
        self.config: AudioStageConfig = config
        self.audio_mapper: Optional[AudioMapper] = None
        self.synthesizer: Optional[AudioSynthesizer] = None
        self.voice_manager: Optional[VoiceManager] = None
        
    def initialize(self) -> bool:
        """ステージの初期化"""
        if not self.config.enable_audio_synthesis:
            self.logger.info("音響合成は無効化されています")
            self._initialized = True
            return True
            
        try:
            # 音響マッパー初期化
            self.audio_mapper = AudioMapper(
                scale=self.config.audio_scale,
                base_octave=3
            )
            self.logger.info(f"音響マッパーを初期化しました: {self.config.audio_scale}")
            
            # シンセサイザー初期化
            self.synthesizer = create_audio_synthesizer(
                max_polyphony=self.config.audio_polyphony
            )
            if self.synthesizer:
                self.logger.info(f"音響シンセサイザーを初期化しました: {self.config.audio_instrument}")
            else:
                self.logger.error("音響シンセサイザーの初期化に失敗しました")
                return False
            
            # ボイスマネージャー初期化
            self.voice_manager = create_voice_manager(
                synthesizer=self.synthesizer,
                max_polyphony=self.config.audio_polyphony,
                steal_strategy=self.config.voice_steal_strategy
            )
            self.logger.info("ボイスマネージャーを初期化しました")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"音響ステージの初期化に失敗: {e}")
            return False
    
    def process(self, collision_events: List[CollisionEvent]) -> AudioStageResult:
        """
        音響合成を実行
        
        Args:
            collision_events: 衝突イベントのリスト
            
        Returns:
            音響処理結果
        """
        if not self._initialized:
            return AudioStageResult(
                success=False,
                error_message="Stage not initialized"
            )
        
        if not self.config.enable_audio_synthesis:
            return AudioStageResult(success=True)
        
        if not collision_events:
            return AudioStageResult(
                success=True,
                active_voices=len(self.voice_manager.active_voices) if self.voice_manager else 0
            )
        
        try:
            notes_triggered = []
            
            for event in collision_events:
                # 音響パラメータへのマッピング
                audio_params = self._map_collision_to_audio(event)
                
                if audio_params:
                    # 空間音響パラメータの調整
                    if self.config.enable_spatial_audio:
                        pan, volume = self._calculate_spatial_params(event.spatial_position)
                        audio_params.pan = pan
                        audio_params.velocity = volume * audio_params.velocity
                    
                    # 音声の割り当てと再生
                    if self.voice_manager and self.synthesizer:
                        voice = allocate_and_play(
                            self.voice_manager,
                            audio_params,
                            priority=5,
                            spatial_position=event.spatial_position if self.config.enable_spatial_audio else None
                        )
                        if voice:
                            notes_triggered.append(audio_params.midi_note)
                            self.logger.debug(f"ノート {audio_params.midi_note} を再生: velocity={audio_params.velocity:.2f}")
            
            # アクティブボイス数を取得
            active_voices = len(self.voice_manager.active_voices) if self.voice_manager else 0
            
            if notes_triggered:
                self.logger.debug(f"{len(notes_triggered)} notes triggered, {active_voices} active voices")
            
            return AudioStageResult(
                success=True,
                active_voices=active_voices,
                notes_triggered=notes_triggered
            )
            
        except Exception as e:
            self.logger.error(f"音響処理エラー: {e}")
            return AudioStageResult(
                success=False,
                error_message=str(e)
            )
    
    def _map_collision_to_audio(self, event: CollisionEvent) -> Optional['AudioParameters']:
        """
        衝突イベントを音響パラメータにマッピング
        
        Args:
            event: 衝突イベント
            
        Returns:
            音響パラメータオブジェクト
        """
        try:
            if self.audio_mapper:
                # AudioMapperを使用してCollisionEventをAudioParametersに変換
                return self.audio_mapper.map_collision_event(event)
            else:
                # フォールバック：手動でAudioParametersを作成
                from ...sound.mapping import AudioParameters
                
                # 位置から音高を決定
                y_normalized = np.clip(event.contact_position[1], -1.0, 1.0)
                pitch = 60 + y_normalized * 24  # C4を中心に±2オクターブ
                
                # X座標から音色の変化
                x_normalized = np.clip(event.contact_position[0], -1.0, 1.0)
                timbre = (x_normalized + 1.0) / 2.0
                
                # 強度からベロシティ
                velocity = np.clip(event.intensity.value / 8.0, 0.1, 1.0)
                
                return AudioParameters(
                    pitch=pitch,
                    velocity=velocity,
                    duration=0.5 + velocity * 0.5,
                    instrument=self.config.audio_instrument,
                    timbre=timbre,
                    brightness=0.7,
                    pan=0.0,  # 後で空間計算で上書きされる
                    distance=0.3,
                    reverb=0.3,
                    attack=0.01,
                    decay=0.1,
                    sustain=0.7,
                    release=0.3,
                    event_id=event.event_id,
                    hand_id=event.hand_id,
                    timestamp=event.timestamp
                )
            
        except Exception as e:
            self.logger.error(f"音響マッピングエラー: {e}")
            return None
    
    def _calculate_spatial_params(self, position: np.ndarray) -> Tuple[float, float]:
        """
        3D位置から空間音響パラメータを計算
        
        Args:
            position: 3D位置
            
        Returns:
            (pan, volume)のタプル
        """
        # リスナーからの相対位置
        relative_pos = position - self.config.listener_position
        
        # パンニング（X座標ベース）
        pan = np.clip(relative_pos[0], -1.0, 1.0)
        
        # 距離減衰
        distance = np.linalg.norm(relative_pos)
        volume = 1.0 / (1.0 + distance * 0.5)
        volume = np.clip(volume, 0.0, 1.0)
        
        return pan, volume
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """設定を動的に更新"""
        super().update_config(new_config)
        
        # 音階や楽器が変更された場合の処理
        if 'audio_scale' in new_config and self.audio_mapper:
            self.audio_mapper.scale = new_config['audio_scale']
            self.logger.info(f"音階を変更: {new_config['audio_scale']}")
        
        if 'audio_instrument' in new_config and self.synthesizer:
            # 楽器変更は再初期化が必要な場合がある
            self.logger.info(f"楽器変更: {new_config['audio_instrument']} (次回初期化時に適用)")
        
        if 'audio_master_volume' in new_config and self.synthesizer:
            self.synthesizer.set_master_volume(new_config['audio_master_volume'])
            self.logger.info(f"マスター音量を変更: {new_config['audio_master_volume']}")
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.synthesizer:
            self.synthesizer.cleanup()
            self.synthesizer = None
        if self.voice_manager:
            self.voice_manager = None
        if self.audio_mapper:
            self.audio_mapper = None
        self._initialized = False
        self.logger.info("音響ステージをクリーンアップしました")