import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.config import settings

# --- グローバルロガーインスタンス ---
# この'geocussion_spark'という名前でロガーを取得すれば、どこからでも同じ設定が使える
LOGGER_NAME = "geocussion_spark"

def setup_logger():
    """
    アプリケーション全体で使用するロガーをセットアップする。
    設定はconfig.yamlから読み込む。
    """
    # 既存のハンドラをすべて削除して、二重にログが出力されるのを防ぐ
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    logger = logging.getLogger(LOGGER_NAME)
    
    try:
        log_level_str = settings.get("general", {}).get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
    except (AttributeError, KeyError):
        logger.setLevel(logging.INFO)
        print("Warning: Log level not found in config, defaulting to INFO.", file=sys.stderr)

    # --- フォーマッターの定義 ---
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s"
    )

    # --- コンソールハンドラの作成 ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # --- ファイルハンドラの作成 (設定が有効な場合) ---
    try:
        if settings.get("general", {}).get("log_to_file", False):
            log_dir = Path(__file__).resolve().parents[2] / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "app.log"
            
            # ログが5MBに達したらローテーションし、バックアップは3つまで保持
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
    except (AttributeError, KeyError, OSError) as e:
        print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)
        
    # 他のライブラリ（例: pyo）のログレベルも合わせる
    logging.getLogger("pyo").setLevel(logging.WARNING)

    return logger

# --- グローバルロガーの初期化 ---
# このモジュールがインポートされた時点でロガーがセットアップされる
log = setup_logger()

# --- 使用例 ---
# from src.utils.logger import log
# log.debug("This is a debug message.")
# log.info("Application starting.")
# log.warning("Something might be wrong.")
# log.error("A critical error occurred.")

# --- 利便性のためのエイリアス ---
# これらをインポートして利用する
debug = log.debug
info = log.info
warning = log.warning
error = log.error
critical = log.critical

# デフォルトでの初期化を防ぐため、明示的な呼び出しを推奨
# if __name__ == '__main__':
#     # 使用例
#     setup_logger(level=logging.DEBUG, log_to_file=True)
#
#     info("これは情報ログです。")
#     warning("これは警告ログです。")
#     debug("これはデバッグログです。")
#
#     try:
#         1 / 0
#     except ZeroDivisionError:
#         error("ゼロ除算エラーが発生しました。", exc_info=True) 