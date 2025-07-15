#!/usr/bin/env python3
"""
OAK-D S2 統合テストスクリプト
カメラファクトリーの動作確認と基本的な機能テスト
"""

import sys
import time
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.camera_factory import create_camera, add_camera_arguments, get_camera_info
from src import setup_logging, get_logger

# ロガー設定
setup_logging()
logger = get_logger(__name__)


def test_camera_factory():
    """カメラファクトリーの基本動作テスト"""
    logger.info("=" * 60)
    logger.info("OAK-D S2 カメラファクトリー統合テスト")
    logger.info("=" * 60)
    
    # 引数解析
    parser = argparse.ArgumentParser(description="OAK-D S2統合テスト")
    add_camera_arguments(parser)
    args = parser.parse_args()
    
    # カメラ情報表示
    camera_type = "OAK-D S2" if args.oak else "Orbbec"
    logger.info(f"使用するカメラ: {camera_type}")
    
    # カメラ作成
    try:
        camera = create_camera(args)
        logger.info("✅ カメラインスタンスの作成に成功しました")
        
        # カメラ情報の取得
        info = get_camera_info(camera)
        logger.info(f"カメラタイプ: {info['type']}")
        logger.info(f"カラーサポート: {info.get('has_color', 'N/A')}")
        
        # 内部パラメータの表示
        if 'depth_intrinsics' in info:
            depth_intrinsics = info['depth_intrinsics']
            logger.info(f"深度内部パラメータ: {depth_intrinsics['width']}x{depth_intrinsics['height']}")
            logger.info(f"  fx={depth_intrinsics['fx']:.1f}, fy={depth_intrinsics['fy']:.1f}")
            logger.info(f"  cx={depth_intrinsics['cx']:.1f}, cy={depth_intrinsics['cy']:.1f}")
        
        return test_camera_operations(camera)
        
    except ImportError as e:
        logger.error(f"❌ カメラの作成に失敗しました（ImportError）: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ カメラの作成に失敗しました: {e}")
        return False


def test_camera_operations(camera):
    """カメラの基本操作テスト"""
    logger.info("\n" + "=" * 40)
    logger.info("カメラ操作テスト")
    logger.info("=" * 40)
    
    try:
        # 初期化テスト
        logger.info("カメラを初期化中...")
        if not camera.initialize():
            logger.error("❌ カメラの初期化に失敗しました")
            return False
        logger.info("✅ カメラの初期化に成功しました")
        
        # 開始テスト
        logger.info("カメラを開始中...")
        if not camera.start():
            logger.error("❌ カメラの開始に失敗しました")
            return False
        logger.info("✅ カメラの開始に成功しました")
        
        # フレーム取得テスト
        logger.info("フレーム取得テスト（5秒間）...")
        frame_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            frame = camera.get_frame(timeout_ms=100)
            if frame is not None:
                frame_count += 1
                
                # 最初のフレームで詳細情報を表示
                if frame_count == 1:
                    logger.info(f"フレーム番号: {frame.frame_number}")
                    logger.info(f"タイムスタンプ: {frame.timestamp_ms:.1f}ms")
                    logger.info(f"深度フレーム: {'有効' if frame.depth_frame else '無効'}")
                    logger.info(f"カラーフレーム: {'有効' if frame.color_frame else '無効'}")
                    
                    # 深度データの確認
                    if frame.depth_frame:
                        depth_data = frame.depth_frame.get_data()
                        logger.info(f"深度データサイズ: {len(depth_data)} bytes")
                        
                    # カラーデータの確認
                    if frame.color_frame:
                        color_data = frame.color_frame.get_data()
                        logger.info(f"カラーデータサイズ: {len(color_data)} bytes")
                        logger.info(f"カラーフォーマット: {frame.color_frame.get_format()}")
                
                # 進捗表示
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"フレーム取得: {frame_count} frames, {fps:.1f} FPS")
            else:
                time.sleep(0.001)  # 短い待機
        
        # 結果表示
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        logger.info(f"✅ フレーム取得テスト完了: {frame_count} frames, 平均 {avg_fps:.1f} FPS")
        
        # 統計情報の取得
        stats = camera.get_stats()
        logger.info(f"カメラ統計情報: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ カメラ操作中にエラーが発生しました: {e}")
        return False
    finally:
        # クリーンアップ
        try:
            camera.stop()
            camera.cleanup()
            logger.info("✅ カメラのクリーンアップが完了しました")
        except Exception as e:
            logger.warning(f"⚠️ クリーンアップ中にエラーが発生しました: {e}")


def main():
    """メイン関数"""
    success = test_camera_factory()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 OAK-D S2 統合テストが成功しました！")
        logger.info("カメラファクトリーは正常に動作しています")
    else:
        logger.error("❌ OAK-D S2 統合テストが失敗しました")
        logger.error("設定やハードウェアを確認してください")
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 