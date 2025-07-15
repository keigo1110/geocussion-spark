#!/usr/bin/env python3
"""
カメラファクトリー - Orbbec と OAK-D カメラの統一インターフェース
"""

import argparse
from typing import Any

from src import get_logger

logger = get_logger(__name__)


def create_camera(args: argparse.Namespace) -> Any:
    """
    引数に基づいてカメラインスタンスを作成
    
    Args:
        args: コマンドライン引数
        
    Returns:
        OrbbecCamera または OakCamera インスタンス
    """
    # OAK-D を使用する場合
    if hasattr(args, 'oak') and args.oak:
        logger.info("OAK-D カメラを使用します")
        from src.input_oak.stream import OakCamera
        
        return OakCamera(
            enable_color=not getattr(args, 'no_color', False)
        )
    
    # デフォルトは Orbbec カメラ
    else:
        logger.info("Orbbec カメラを使用します")
        from src.input.stream import OrbbecCamera
        
        return OrbbecCamera(
            enable_color=not getattr(args, 'no_color', False),
            depth_width=getattr(args, 'depth_w', None),
            depth_height=getattr(args, 'depth_h', None)
        )


def add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    """
    カメラ関連の引数をparserに追加
    
    Args:
        parser: ArgumentParser インスタンス
    """
    camera_group = parser.add_argument_group('camera', 'カメラ設定')
    
    camera_group.add_argument(
        '--oak',
        action='store_true',
        help='OAK-D カメラを使用する（デフォルト: Orbbec）'
    )
    
    camera_group.add_argument(
        '--no-color',
        action='store_true',
        help='カラーストリームを無効にする'
    )
    
    camera_group.add_argument(
        '--depth-w',
        type=int,
        help='深度ストリーム幅（Orbbecのみ）'
    )
    
    camera_group.add_argument(
        '--depth-h', 
        type=int,
        help='深度ストリーム高さ（Orbbecのみ）'
    )


def get_camera_info(camera) -> dict:
    """
    カメラの情報を取得
    
    Args:
        camera: カメラインスタンス
        
    Returns:
        カメラ情報の辞書
    """
    info = {
        'type': type(camera).__name__,
        'has_color': getattr(camera, 'has_color', False),
        'is_started': getattr(camera, 'is_started', False),
        'frame_count': getattr(camera, 'frame_count', 0)
    }
    
    # 内部パラメータ情報
    if hasattr(camera, 'depth_intrinsics') and camera.depth_intrinsics:
        info['depth_intrinsics'] = {
            'fx': camera.depth_intrinsics.fx,
            'fy': camera.depth_intrinsics.fy,
            'cx': camera.depth_intrinsics.cx,
            'cy': camera.depth_intrinsics.cy,
            'width': camera.depth_intrinsics.width,
            'height': camera.depth_intrinsics.height
        }
    
    if hasattr(camera, 'color_intrinsics') and camera.color_intrinsics:
        info['color_intrinsics'] = {
            'fx': camera.color_intrinsics.fx,
            'fy': camera.color_intrinsics.fy,
            'cx': camera.color_intrinsics.cx,
            'cy': camera.color_intrinsics.cy,
            'width': camera.color_intrinsics.width,
            'height': camera.color_intrinsics.height
        }
    
    return info 