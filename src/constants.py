#!/usr/bin/env python3
"""
共通定数・設定値

アプリケーション全体で使用される定数や閾値を一元管理し、
モジュール間の循環依存を解消します。
"""

from typing import Final

# =============================================================================
# 数値精度・許容誤差
# =============================================================================

# 数値計算の許容誤差
NUMERICAL_TOLERANCE: Final[float] = 1e-6
COLLISION_TOLERANCE: Final[float] = 1e-6
DISTANCE_EPSILON: Final[float] = 1e-8

# =============================================================================
# 衝突検出関連
# =============================================================================

# デフォルト検索半径（メートル）
DEFAULT_SEARCH_RADIUS: Final[float] = 0.05  # 5cm
DEFAULT_SPHERE_RADIUS: Final[float] = 0.02  # 2cm

# 衝突検出制限
MAX_CONTACTS_PER_SPHERE: Final[int] = 10
MAX_COLLISION_CANDIDATES: Final[int] = 100

# 性能閾値
COLLISION_SEARCH_TIME_LIMIT_MS: Final[float] = 10.0
COLLISION_DETECTION_TIME_LIMIT_MS: Final[float] = 5.0

# 衝突検出改善パラメータ
COLLISION_ADAPTIVE_VELOCITY_COEFFICIENT = 0.12  # 速度係数（0.05から拡大）
COLLISION_ADAPTIVE_MAX_RADIUS_EXTENSION = 0.08  # 最大半径拡張（3cmから8cmに）
COLLISION_INTERPOLATION_MIN_SAMPLES = 3  # 最小補間サンプル数
COLLISION_INTERPOLATION_MAX_SAMPLES = 8  # 最大補間サンプル数
COLLISION_HAND_HISTORY_SIZE = 8  # 手位置履歴サイズ（3から8に拡張）

# 高速移動検出閾値
HIGH_SPEED_VELOCITY_THRESHOLD = 1.5  # m/s - 高速移動判定閾値
VERY_HIGH_SPEED_VELOCITY_THRESHOLD = 2.5  # m/s - 超高速移動判定閾値

# =============================================================================
# メッシュ処理関連
# =============================================================================

# 三角形数制限
DEFAULT_MAX_TRIANGLES: Final[int] = 10000
MAX_TRIANGLES_PER_LEAF: Final[int] = 10
ADAPTIVE_SIMPLIFY_TARGET: Final[int] = 1000

# メッシュ品質
MIN_TRIANGLE_AREA: Final[float] = 1e-8
MESH_QUALITY_THRESHOLD: Final[float] = 0.5

# 空間分割
SPATIAL_INDEX_MAX_DEPTH: Final[int] = 10

# =============================================================================
# 音響処理関連
# =============================================================================

# オーディオ設定
DEFAULT_SAMPLE_RATE: Final[int] = 44100
DEFAULT_BUFFER_SIZE: Final[int] = 256
DEFAULT_CHANNELS: Final[int] = 2

# 音響パラメータ
MIN_FREQUENCY: Final[float] = 20.0
MAX_FREQUENCY: Final[float] = 20000.0
DEFAULT_VOLUME: Final[float] = 0.5

# ボイス管理
MAX_CONCURRENT_VOICES: Final[int] = 64
VOICE_FADE_TIME: Final[float] = 0.1

# =============================================================================
# 入力処理関連
# =============================================================================

# カメラ解像度
DEFAULT_DEPTH_WIDTH: Final[int] = 640
DEFAULT_DEPTH_HEIGHT: Final[int] = 480
DEFAULT_COLOR_WIDTH: Final[int] = 1280
DEFAULT_COLOR_HEIGHT: Final[int] = 720

# フレームレート
DEFAULT_FPS: Final[int] = 30
MAX_FPS: Final[int] = 60

# 深度処理
MIN_DEPTH_VALUE: Final[float] = 0.1  # 10cm
MAX_DEPTH_VALUE: Final[float] = 10.0  # 10m
DEPTH_NOISE_THRESHOLD: Final[float] = 0.01

# =============================================================================
# 手検出関連
# =============================================================================

# 検出信頼度
MIN_DETECTION_CONFIDENCE: Final[float] = 0.5
MIN_TRACKING_CONFIDENCE: Final[float] = 0.5
MIN_PRESENCE_CONFIDENCE: Final[float] = 0.5

# 手の物理的制限
MAX_HAND_SIZE: Final[float] = 0.25  # 25cm
MIN_HAND_SIZE: Final[float] = 0.05  # 5cm

# トラッキング
MAX_TRACKING_DISTANCE: Final[float] = 0.5  # 50cm
TRACKING_SMOOTHING_FACTOR: Final[float] = 0.7

# =============================================================================
# リソース管理関連
# =============================================================================

# メモリ最適化
DEFAULT_ARRAY_POOL_SIZE: Final[int] = 100
MEMORY_CLEANUP_INTERVAL: Final[float] = 30.0  # 30秒
IDLE_RESOURCE_TIMEOUT: Final[float] = 300.0   # 5分

# 大容量配列の判定閾値（バイト）
LARGE_ARRAY_THRESHOLD: Final[int] = 1024 * 1024  # 1MB

# =============================================================================
# デバッグ・UI関連
# =============================================================================

# ウィンドウサイズ
DEFAULT_WINDOW_WIDTH: Final[int] = 1280
DEFAULT_WINDOW_HEIGHT: Final[int] = 960

# 描画設定
DEBUG_POINT_SIZE: Final[int] = 3
DEBUG_LINE_THICKNESS: Final[int] = 2
FPS_DISPLAY_INTERVAL: Final[float] = 1.0  # 1秒

# 色定数（BGR形式）
COLOR_RED: Final[tuple] = (0, 0, 255)
COLOR_GREEN: Final[tuple] = (0, 255, 0)
COLOR_BLUE: Final[tuple] = (255, 0, 0)
COLOR_YELLOW: Final[tuple] = (0, 255, 255)
COLOR_CYAN: Final[tuple] = (255, 255, 0)
COLOR_MAGENTA: Final[tuple] = (255, 0, 255)
COLOR_WHITE: Final[tuple] = (255, 255, 255)
COLOR_BLACK: Final[tuple] = (0, 0, 0)

# =============================================================================
# パフォーマンス監視
# =============================================================================

# 統計レポート間隔
STATS_REPORT_INTERVAL: Final[float] = 5.0  # 5秒
PERFORMANCE_HISTORY_SIZE: Final[int] = 100

# パフォーマンス警告閾値
HIGH_CPU_USAGE_THRESHOLD: Final[float] = 80.0  # 80%
HIGH_MEMORY_USAGE_THRESHOLD: Final[float] = 512 * 1024 * 1024  # 512MB 

# =============================================================================
# デモスクリプト（demo_collision_detection.py）固有の設定
# =============================================================================

# 衝突デモ用デフォルト値
DEMO_SPHERE_RADIUS_DEFAULT: Final[float] = 0.05  # 5 cm
DEMO_MESH_UPDATE_INTERVAL: Final[int] = 15       # フレーム
DEMO_MAX_MESH_SKIP_FRAMES: Final[int] = 60       # フレーム

# オーディオ／ビジュアルデフォルト
DEMO_AUDIO_COOLDOWN_TIME: Final[float] = 0.3     # 秒
DEMO_VOXEL_SIZE: Final[float] = 0.005            # 5 mm
DEMO_AUDIO_POLYPHONY: Final[int] = 16
DEMO_MASTER_VOLUME: Final[float] = 0.7

# 解像度プリセット
LOW_RESOLUTION: Final[tuple[int, int]] = (424, 240)
HIGH_RESOLUTION: Final[tuple[int, int]] = (848, 480)
ESTIMATED_HIGH_RES_POINTS: Final[int] = 300_000 