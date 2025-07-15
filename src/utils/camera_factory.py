#!/usr/bin/env python3
"""
カメラファクトリーモジュール
OrbbecCameraとOAK-D S2を統一的に作成
"""

import argparse
from typing import Optional, Union
from src import get_logger

logger = get_logger(__name__)


def create_camera(args: argparse.Namespace) -> Union['OrbbecCamera', 'OakCamera']:
    """
    引数に基づいてカメラインスタンスを作成
    
    Args:
        args: コマンドライン引数（--oakフラグを含む）
        
    Returns:
        カメラインスタンス（OrbbecCameraまたはOakCamera）
        
    Raises:
        ImportError: 指定されたカメラタイプが利用できない場合
        RuntimeError: カメラの初期化に失敗した場合
    """
    if args.oak:
        # OAK-D S2を使用
        try:
            from src.input_oak.stream import OakCamera as Cam
            logger.info("OAK-D S2カメラを使用します")
            
            camera = Cam(
                enable_color=not args.no_color,
                downsample_factor=getattr(args, 'oak_downsample', 3)
            )
            
            return camera
            
        except ImportError as e:
            logger.error(f"OAK-D S2が利用できません: {e}")
            raise ImportError(
                "OAK-D S2を使用するには、'source oakenv/bin/activate' で "
                "OAK-D S2用仮想環境をアクティベートし、depthai>=2.24がインストールされている必要があります"
            ) from e
            
    else:
        # Orbbecを使用（デフォルト）
        try:
            from src.input.stream import OrbbecCamera as Cam
            logger.info("Orbbecカメラを使用します")
            
            camera = Cam(
                enable_color=not args.no_color,
                depth_width=args.depth_w, 
                depth_height=args.depth_h
            )
            
            return camera
            
        except ImportError as e:
            logger.error(f"Orbbecカメラが利用できません: {e}")
            raise ImportError(
                "Orbbecカメラを使用するには、OrbbecSDKがインストールされている必要があります"
            ) from e


def add_camera_arguments(parser):
    """カメラ選択用の引数を追加"""
    camera_group = parser.add_argument_group('カメラ選択')
    
    camera_group.add_argument(
        '--oak',
        action='store_true',
        help='OAK-D S2カメラを使用（デフォルト: Orbbec）'
    )
    
    camera_group.add_argument(
        '--no-color',
        action='store_true',
        help='カラーストリームを無効にする'
    )
    
    camera_group.add_argument(
        '--depth-w',
        type=int,
        default=424,
        help='深度ストリーム幅 (Orbbecのみ)'
    )
    
    camera_group.add_argument(
        '--depth-h',
        type=int,
        default=240,
        help='深度ストリーム高さ (Orbbecのみ)'
    )

    camera_group.add_argument(
        '--oak-downsample',
        type=int,
        default=3,
        help='OAK-D S2のパフォーマンス向上のためのダウンサンプリング係数'
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