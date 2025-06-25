#!/usr/bin/env python3
"""
Geocussion-SP 設定管理システム

プロジェクト全体で使用される設定値を統一管理し、
Magic Numberのハードコーディングを解消します。
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from src import get_logger

logger = get_logger(__name__)


@dataclass
class AudioConfig:
    """音響システム設定"""
    # エンジン設定
    sample_rate: int = 44100
    buffer_size: int = 256
    channels: int = 2
    audio_driver: str = "portaudio"
    enable_duplex: bool = False
    
    # 音量・エフェクト設定
    master_volume: float = 0.7
    reverb_level: float = 0.3
    
    # ボイス管理設定
    max_polyphony: int = 16
    voice_steal_strategy: str = "oldest"
    
    # 楽器別設定
    instrument_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "sine": {
            "carrier_ratio": 1.0,
            "modulator_ratio": 4.0,
            "modulation_index": 2.5
        },
        "sawtooth": {
            "harmonics_1": 1.0,
            "harmonics_2": 0.5,
            "harmonics_3": 0.25,
            "harmonics_4": 0.125,
            "filter_cutoff": 800.0,
            "filter_resonance": 0.3
        },
        "square": {
            "carrier_ratio": 1.0,
            "modulator_ratio": 3.14,
            "modulation_index": 2.0
        },
        "noise": {
            "harmonics_1": 1.0,
            "harmonics_2": 0.8,
            "harmonics_3": 0.6,
            "harmonics_4": 0.4,
            "harmonics_5": 0.2,
            "detune": 0.02,
            "chorus_depth": 0.3
        },
        "pluck": {
            "pluck_position": 0.3,
            "damping": 0.1,
            "string_tension": 0.8
        }
    })
    
    # 空間音響設定
    room_size: float = 10.0
    reverb_decay: float = 2.0
    doppler_factor: float = 1.0
    air_absorption: float = 0.01


@dataclass  
class CollisionConfig:
    """衝突検出設定"""
    # 球体設定
    default_sphere_radius: float = 0.05  # 5cm
    min_sphere_radius: float = 0.01      # 1cm
    max_sphere_radius: float = 0.2       # 20cm
    
    # 検索設定
    default_search_radius: float = 0.05  # 5cm
    max_search_radius: float = 0.2       # 20cm
    max_cache_size: int = 100
    
    # 判定精度設定
    collision_tolerance: float = 1e-6
    enable_face_culling: bool = False
    max_contacts_per_sphere: int = 10
    
    # パフォーマンス設定
    enable_adaptive_radius: bool = True
    adaptive_radius_history_size: int = 10


@dataclass
class MeshConfig:
    """メッシュ生成設定"""
    # 点群投影設定
    projection_resolution: float = 0.01  # 1cm解像度
    projection_method: str = "median_height"
    enable_hole_filling: bool = True
    
    # Delaunay三角化設定
    enable_adaptive_sampling: bool = True
    enable_boundary_points: bool = True
    quality_threshold: float = 0.5
    
    # メッシュ簡略化設定  
    target_reduction: float = 0.7        # 70%削減
    preserve_boundary: bool = True
    simplify_quality_threshold: float = 0.1
    
    # 属性計算設定
    curvature_radius: float = 0.05       # 曲率計算半径
    
    # パフォーマンス設定
    max_triangles: int = 10000
    max_vertices: int = 5000


@dataclass
class InputConfig:
    """入力システム設定"""
    # カメラ設定
    default_depth_width: int = 640
    default_depth_height: int = 480
    default_depth_fps: int = 30
    
    # 深度処理設定
    depth_scale: float = 1000.0          # mm to m conversion
    min_depth: float = 0.1               # 10cm
    max_depth: float = 10.0              # 10m
    
    # 点群変換設定
    point_cloud_downsample: bool = True
    downsample_factor: int = 2
    
    # フィルタ設定
    enable_temporal_filter: bool = True
    temporal_alpha: float = 0.1
    enable_bilateral_filter: bool = True
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 10.0
    bilateral_sigma_space: float = 10.0


@dataclass
class DetectionConfig:
    """手検出設定"""
    # MediaPipe設定
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    
    # 3D投影設定
    enable_depth_interpolation: bool = True
    interpolation_method: str = "bilinear"
    depth_smoothing_kernel_size: int = 5
    
    # トラッキング設定
    enable_kalman_filter: bool = True
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 1e-1
    tracking_max_distance: float = 0.3   # 30cm
    tracking_history_size: int = 10


@dataclass
class VisualizationConfig:
    """可視化設定"""
    # ウィンドウ設定
    window_width: int = 1280
    window_height: int = 720
    enable_high_resolution: bool = True
    
    # 更新間隔設定
    point_cloud_update_interval: int = 1  # フレーム
    mesh_update_interval: int = 5         # フレーム
    max_mesh_skip_frames: int = 60        # 2秒@30fps
    
    # 表示設定
    point_size: float = 2.0
    enable_hand_visualization: bool = True
    enable_collision_visualization: bool = True
    enable_mesh_visualization: bool = True


@dataclass
class GeocussionConfig:
    """プロジェクト全体設定"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    input: InputConfig = field(default_factory=InputConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # ログ設定
    log_level: str = "INFO"
    log_format_style: str = "detailed"
    enable_performance_logging: bool = True


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self):
        self._config: Optional[GeocussionConfig] = None
        self._config_file_path: Optional[Path] = None
    
    def load_config(self, config_file: Optional[Path] = None) -> GeocussionConfig:
        """
        設定ファイルを読み込み
        
        Args:
            config_file: 設定ファイルパス（Noneの場合はデフォルト設定）
            
        Returns:
            読み込まれた設定
        """
        if config_file is None:
            # デフォルト設定ファイルを探す
            project_root = Path(__file__).parent.parent
            default_paths = [
                project_root / "geocussion.yaml",
                project_root / "config.yaml",
                Path.home() / ".geocussion" / "config.yaml"
            ]
            
            for path in default_paths:
                if path.exists():
                    config_file = path
                    break
        
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                
                self._config = self._dict_to_config(config_dict)
                self._config_file_path = config_file
                logger.info(f"Configuration loaded from {config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
                logger.info("Using default configuration")
                self._config = GeocussionConfig()
        else:
            logger.info("No config file found, using default configuration")
            self._config = GeocussionConfig()
        
        return self._config
    
    def save_config(self, config_file: Optional[Path] = None) -> bool:
        """
        設定をファイルに保存
        
        Args:
            config_file: 保存先ファイルパス
            
        Returns:
            保存成功したかどうか
        """
        if self._config is None:
            logger.error("No configuration to save")
            return False
        
        if config_file is None:
            config_file = self._config_file_path or Path("geocussion.yaml")
        
        try:
            config_dict = self._config_to_dict(self._config)
            
            # ディレクトリが存在しない場合は作成
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")
            return False
    
    def get_config(self) -> GeocussionConfig:
        """現在の設定を取得"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> GeocussionConfig:
        """辞書を設定オブジェクトに変換"""
        # 簡単な実装（完全な型変換は複雑なので最小限）
        config = GeocussionConfig()
        
        if 'audio' in config_dict:
            audio_dict = config_dict['audio']
            if isinstance(audio_dict, dict):
                for key, value in audio_dict.items():
                    if hasattr(config.audio, key):
                        setattr(config.audio, key, value)
        
        if 'collision' in config_dict:
            collision_dict = config_dict['collision']
            if isinstance(collision_dict, dict):
                for key, value in collision_dict.items():
                    if hasattr(config.collision, key):
                        setattr(config.collision, key, value)
        
        # 他のセクションも同様に処理...
        # （実装は長くなるため、必要に応じて拡張）
        
        return config
    
    def _config_to_dict(self, config: GeocussionConfig) -> Dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        return {
            'audio': {
                'sample_rate': config.audio.sample_rate,
                'buffer_size': config.audio.buffer_size,
                'master_volume': config.audio.master_volume,
                'reverb_level': config.audio.reverb_level,
                'max_polyphony': config.audio.max_polyphony,
                # 必要な設定のみを出力
            },
            'collision': {
                'default_sphere_radius': config.collision.default_sphere_radius,
                'default_search_radius': config.collision.default_search_radius,
                'collision_tolerance': config.collision.collision_tolerance,
                'max_contacts_per_sphere': config.collision.max_contacts_per_sphere,
            },
            'mesh': {
                'projection_resolution': config.mesh.projection_resolution,
                'quality_threshold': config.mesh.quality_threshold,
                'target_reduction': config.mesh.target_reduction,
            },
            'input': {
                'min_depth': config.input.min_depth,
                'max_depth': config.input.max_depth,
                'depth_scale': config.input.depth_scale,
            },
            'detection': {
                'min_detection_confidence': config.detection.min_detection_confidence,
                'min_tracking_confidence': config.detection.min_tracking_confidence,
                'max_num_hands': config.detection.max_num_hands,
            },
            'log_level': config.log_level,
        }


# グローバル設定マネージャー
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """グローバル設定マネージャーを取得"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> GeocussionConfig:
    """現在の設定を取得"""
    return get_config_manager().get_config()

def load_config(config_file: Optional[Path] = None) -> GeocussionConfig:
    """設定を読み込み"""
    return get_config_manager().load_config(config_file)

def save_config(config_file: Optional[Path] = None) -> bool:
    """設定を保存"""
    return get_config_manager().save_config(config_file) 