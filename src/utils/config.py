import yaml
from pathlib import Path
from typing import Any, Optional

# --- 定数 ---
# プロジェクトルートからの相対パスでconfig.yamlの場所を決定
CONFIG_FILE_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

class DotDict(dict):
    """
    ドット表記でアクセス可能な辞書クラス。
    例: config.general.log_level
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

class ConfigManager:
    """
    設定ファイル(config.yaml)を読み込み、管理するシングルトンクラス。
    
    Attributes:
        _instance (Optional['ConfigManager']): シングルトンのインスタンス
        _config (Optional[dict]): 読み込まれた設定データを保持する辞書
    """
    _instance: Optional['ConfigManager'] = None
    _config: Optional[dict] = None

    def __new__(cls):
        """シングルトンインスタンスを生成または返す。"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        設定ファイルを読み込み、内部変数に格納する。
        ファイルが見つからない場合はエラーを発生させる。
        """
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            # ここではloggingの準備ができていない可能性が高いので、標準エラー出力を使う
            import sys
            print(f"ERROR: Configuration file not found at {CONFIG_FILE_PATH}", file=sys.stderr)
            sys.exit(1)
        except yaml.YAMLError as e:
            import sys
            print(f"ERROR: Error parsing YAML file {CONFIG_FILE_PATH}: {e}", file=sys.stderr)
            sys.exit(1)

    def get_config(self) -> dict:
        """
        読み込まれた設定データを返す。

        Returns:
            dict: 設定データの辞書
        
        Raises:
            RuntimeError: 設定がロードされていない場合に発生。
        """
        if self._config is None:
            # この状況は、正常な初期化フローでは発生しないはず
            raise RuntimeError("Configuration has not been loaded.")
        return self._config

# --- モジュールレベルでのインスタンス化 ---
# このモジュールをインポートすると、自動的に設定が読み込まれる
config_manager = ConfigManager()

# --- グローバルアクセス用の関数 ---
def get_config() -> dict:
    """
    グローバルな設定インスタンスから設定データを取得するための便利な関数。

    使用例:
    from src.utils.config import get_config
    config = get_config()
    log_level = config["general"]["log_level"]
    """
    return config_manager.get_config()

# --- 直接アクセス用の変数 ---
# より直接的にアクセスしたい場合に使用
# from src.utils.config import settings
# log_level = settings["general"]["log_level"]
settings = get_config()

# --- グローバルな設定インスタンス ---
# アプリケーション全体でこのインスタンスをインポートして使用する
config_manager = ConfigManager()

# --- 使用例 ---
# if __name__ == '__main__':
#     try:
#         # アプリケーションの起動時に一度だけロードする
#         config_manager.load_config()
#
#         # ドットアクセスで値を取得
#         log_level = config_manager.config.general.log_level
#         print(f"Log Level: {log_level}")
#
#         # getメソッドで値を取得
#         sphere_radius = config_manager.get('collision_detection.sphere_radius_m', 0.05)
#         print(f"Sphere Radius: {sphere_radius}")
#
#         # 存在しないキー
#         non_existent = config_manager.get('foo.bar', 'default_value')
#         print(f"Non-existent key: {non_existent}")
#
#     except (FileNotFoundError, yaml.YAMLError) as e:
#         print(f"Failed to initialize configuration: {e}") 