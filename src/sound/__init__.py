"""
Geocussion-SP 音響生成フェーズ

このパッケージは衝突イベントを基にリアルタイム音響合成を行う機能を提供します。
衝突パラメータから音響パラメータへのマッピング、pyo音響合成エンジン、
ボイス管理システムで構成されています。

処理フロー:
1. パラメータマッピング (mapping.py) - 衝突→音響パラメータ変換
2. 音響合成 (synth.py) - pyo音響エンジンによるリアルタイム合成
3. ボイス管理 (voice_mgr.py) - ポリフォニー制御と空間配置

予算時間: 5ms
"""

# パラメータマッピング
from .mapping import (
    InstrumentType,
    ScaleType,
    AudioParameters,
    AudioMapper,
    map_collision_to_audio,
    batch_map_collisions,
    create_scale_mapper
)

# 音響合成エンジン
from .synth import (
    EngineState,
    AudioConfig,
    AudioSynthesizer,
    create_audio_synthesizer,
    play_audio_immediately
)

# ボイス管理システム
from .voice_mgr import (
    VoiceState,
    StealStrategy,
    SpatialMode,
    VoiceInfo,
    SpatialConfig,
    VoiceManager,
    create_voice_manager,
    allocate_and_play
)

__all__ = [
    # パラメータマッピング
    'InstrumentType',
    'ScaleType',
    'AudioParameters',
    'AudioMapper',
    'map_collision_to_audio',
    'batch_map_collisions',
    'create_scale_mapper',
    
    # 音響合成エンジン
    'EngineState',
    'AudioConfig',
    'AudioSynthesizer',
    'create_audio_synthesizer',
    'play_audio_immediately',
    
    # ボイス管理システム
    'VoiceState',
    'StealStrategy',
    'SpatialMode',
    'VoiceInfo',
    'SpatialConfig',
    'VoiceManager',
    'create_voice_manager',
    'allocate_and_play'
] 