"""
OAK-D S2入力フェーズパッケージ
OAK-D S2カメラからの深度・RGB画像取得
"""

from .stream import OakCamera

__all__ = [
    'OakCamera',
] 