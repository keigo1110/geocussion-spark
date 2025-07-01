#!/usr/bin/env python3
"""
統合ビューワー（パイプライン処理 + UI表示）
責務: HandledPipelineと組み合わせた統合表示システム
"""

import time
import threading
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

# パイプライン処理
from .pipeline_wrapper import HandledPipeline, HandledPipelineConfig, PipelineResults

# UI表示（Open3D）
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None  # type: ignore

# 音響設定
from ..sound.mapping import ScaleType, InstrumentType
from ..input.stream import OrbbecCamera
from ..types import FrameData, OBFormat, CameraIntrinsics

# イベントシステム
from .events import (
    get_event_dispatcher,
    EventHandler,
    EventType,
    KeyPressedEvent,
    WindowResizedEvent,
    ViewportChangedEvent,
    FrameProcessedEvent,
    MeshUpdatedEvent,
    CollisionDetectedEvent,
    StageCompletedEvent,
    ErrorEvent
)
from .events.pipeline_events import MeshUpdatedEvent as MeshEvent
from .events.pipeline_events import CollisionDetectedEvent as CollisionEvent
from .events.config_handler import ConfigurationEventHandler


class IntegratedGeocussionViewer(EventHandler):
    """
    統合Geocussionビューワー
    HandledPipelineとUI表示を組み合わせた統合システム
    """
    
    def __init__(self, config: Optional[HandledPipelineConfig] = None, **kwargs: Any) -> None:
        """
        初期化
        
        Args:
            config: パイプライン設定（指定されない場合はkwargsから構築）
            **kwargs: 設定パラメータ（config未指定時に使用）
        """
        print("統合ビューワー初期化中...")
        
        # パイプライン設定構築
        if config is not None:
            self.pipeline_config = config
        else:
            self.pipeline_config = HandledPipelineConfig(
                enable_filter=kwargs.get('enable_filter', True),
                enable_hand_detection=kwargs.get('enable_hand_detection', True),
                enable_tracking=kwargs.get('enable_tracking', True),
                min_detection_confidence=kwargs.get('min_detection_confidence', 0.7),
                use_gpu_mediapipe=kwargs.get('use_gpu_mediapipe', False),
                
                enable_mesh_generation=kwargs.get('enable_mesh_generation', True),
                mesh_update_interval=kwargs.get('mesh_update_interval', 10),
                max_mesh_skip_frames=kwargs.get('max_mesh_skip_frames', 60),
                mesh_resolution=kwargs.get('mesh_resolution', 0.01),
                mesh_quality_threshold=kwargs.get('mesh_quality_threshold', 0.3),
                mesh_reduction=kwargs.get('mesh_reduction', 0.7),
                
                enable_collision_detection=kwargs.get('enable_collision_detection', True),
                enable_collision_visualization=kwargs.get('enable_collision_visualization', True),
                sphere_radius=kwargs.get('sphere_radius', 0.05),
                
                enable_audio_synthesis=kwargs.get('enable_audio_synthesis', True),
                audio_scale=kwargs.get('audio_scale', ScaleType.PENTATONIC),
                audio_instrument=kwargs.get('audio_instrument', InstrumentType.MARIMBA),
                audio_polyphony=kwargs.get('audio_polyphony', 16),
                audio_master_volume=kwargs.get('audio_master_volume', 0.7),
                
                enable_voxel_downsampling=kwargs.get('enable_voxel_downsampling', True),
                voxel_size=kwargs.get('voxel_size', 0.005),
                enable_gpu_acceleration=kwargs.get('enable_gpu_acceleration', True)
            )
        
        # パイプライン初期化
        self.pipeline = HandledPipeline(self.pipeline_config)
        
        # カメラ設定
        self.camera: Optional[OrbbecCamera] = None
        self.depth_width = kwargs.get('depth_width')
        self.depth_height = kwargs.get('depth_height')
        
        # UI設定
        self.rgb_window_size = kwargs.get('rgb_window_size', (640, 480))
        self.point_size = kwargs.get('point_size', 2.0)
        
        # ヘッドレスモード設定
        self.headless_mode = kwargs.get('headless_mode', False)
        self.headless_duration = kwargs.get('headless_duration', 30)
        self.pure_headless_mode = kwargs.get('pure_headless_mode', False)
        
        # Open3D関連
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        
        # 可視化オブジェクト
        self.mesh_geometries = []
        self.collision_geometries = []
        self.hand_markers = []
        
        # 状態管理
        self.is_running = False
        self.is_initialized = False
        self.current_results: Optional[PipelineResults] = None
        
        # パフォーマンス表示
        self.show_performance = False
        
        # ヘルプテキスト
        self.help_text = self._build_help_text()
        
        # イベントシステム初期化
        self.event_dispatcher = get_event_dispatcher()
        self._subscribe_to_events()
        
        # 設定変更ハンドラー初期化
        self.config_handler = ConfigurationEventHandler(self.pipeline_config, self.pipeline)
        self.event_dispatcher.subscribe(EventType.KEY_PRESSED, self.config_handler)
        
        print("統合ビューワー初期化完了")
        print(f"  - メッシュ生成: {'有効' if self.pipeline_config.enable_mesh_generation else '無効'}")
        print(f"  - 衝突検出: {'有効' if self.pipeline_config.enable_collision_detection else '無効'}")
        print(f"  - 音響合成: {'有効' if self.pipeline_config.enable_audio_synthesis else '無効'}")
        print(f"  - ヘッドレスモード: {'有効' if self.headless_mode else '無効'}")
    
    def _build_help_text(self) -> str:
        """ヘルプテキスト構築"""
        help_text = "=== Geocussion 統合ビューワー ===\n"
        help_text += "ESC/Q: 終了\n"
        help_text += "H: ヘルプ表示\n"
        help_text += "\n=== メッシュ・衝突制御 ===\n"
        help_text += "M: メッシュ生成 ON/OFF\n"
        help_text += "C: 衝突検出 ON/OFF\n"
        help_text += "V: 衝突可視化 ON/OFF\n"
        help_text += "N: メッシュ強制更新\n"
        help_text += "+/-: 球半径調整\n"
        help_text += "P: パフォーマンス統計表示\n"
        
        if self.pipeline_config.enable_audio_synthesis:
            help_text += "\n=== 音響制御 ===\n"
            help_text += "A: 音響合成 ON/OFF\n"
            help_text += "S: 音階切り替え\n"
            help_text += "I: 楽器切り替え\n"
            help_text += "1/2: 音量調整\n"
        
        return help_text
    
    def _subscribe_to_events(self) -> None:
        """イベントサブスクリプション設定"""
        # パイプラインイベント
        self.event_dispatcher.subscribe(EventType.FRAME_PROCESSED, self)
        self.event_dispatcher.subscribe(EventType.MESH_UPDATED, self)
        self.event_dispatcher.subscribe(EventType.COLLISION_DETECTED, self)
        self.event_dispatcher.subscribe(EventType.STAGE_COMPLETED, self)
        self.event_dispatcher.subscribe(EventType.PIPELINE_ERROR, self)
    
    def handle_event(self, event) -> None:
        """
        イベントハンドラー実装
        
        Args:
            event: 処理するイベント
        """
        try:
            if event.event_type == EventType.FRAME_PROCESSED:
                # フレーム処理完了イベント
                # 現在は_process_frameで直接処理しているので、必要に応じて実装
                pass
                
            elif event.event_type == EventType.MESH_UPDATED:
                # メッシュ更新イベント
                if self.vis and not self.headless_mode:
                    # メッシュ可視化を更新
                    self._update_mesh_from_event(event)
                    
            elif event.event_type == EventType.COLLISION_DETECTED:
                # 衝突検出イベント
                if self.vis and not self.headless_mode:
                    # 衝突可視化を更新
                    self._update_collision_from_event(event)
                    
            elif event.event_type == EventType.STAGE_COMPLETED:
                # ステージ完了イベント（パフォーマンス統計など）
                if self.show_performance:
                    print(f"Stage {event.stage_name} completed in {event.processing_time_ms:.1f}ms")
                    
            elif event.event_type == EventType.PIPELINE_ERROR:
                # エラーイベント
                print(f"Pipeline error in {event.stage_name}: {event.error_message}")
                
        except Exception as e:
            print(f"Event handling error: {e}")
    
    def _update_mesh_from_event(self, event: MeshEvent) -> None:
        """メッシュ更新イベントからメッシュを更新"""
        try:
            # 既存メッシュ削除
            for geom in self.mesh_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.mesh_geometries.clear()
            
            # 新しいメッシュ作成
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(event.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(event.triangles)
            
            # メッシュ色設定
            if event.colors is not None:
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(event.colors)
            else:
                o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
            o3d_mesh.compute_vertex_normals()
            
            # ビューワーに追加
            self.vis.add_geometry(o3d_mesh, reset_bounding_box=False)
            self.mesh_geometries.append(o3d_mesh)
            
        except Exception as e:
            print(f"Mesh update from event error: {e}")
    
    def _update_collision_from_event(self, event: CollisionEvent) -> None:
        """衝突検出イベントから衝突可視化を更新"""
        try:
            # 既存衝突ジオメトリ削除
            for geom in self.collision_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.collision_geometries.clear()
            
            # 新しい衝突点を表示
            for collision in event.collision_events:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(collision.position)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
                
                self.vis.add_geometry(sphere, reset_bounding_box=False)
                self.collision_geometries.append(sphere)
                
        except Exception as e:
            print(f"Collision update from event error: {e}")
    
    def initialize(self) -> bool:
        """
        ビューワー初期化
        
        Returns:
            初期化成功時True
        """
        try:
            print("統合ビューワー初期化開始...")
            
            # カメラ初期化
            if not self.headless_mode:
                self.camera = OrbbecCamera(
                    enable_color=True,
                    depth_width=self.depth_width,
                    depth_height=self.depth_height
                )
                if not self.camera.initialize():
                    print("Failed to initialize camera")
                    return False
                
                # カメラストリーミング開始
                if not self.camera.start():
                    print("Failed to start camera")
                    return False
            
            # パイプライン初期化
            if not self.pipeline.initialize(self.camera):
                print("Failed to initialize pipeline")
                return False
            
            # Open3Dビューワー初期化（ヘッドレスでない場合）
            if not self.headless_mode and HAS_OPEN3D:
                if not self._initialize_3d_viewer():
                    print("Warning: 3D viewer initialization failed")
            
            print("統合ビューワー初期化完了")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Integrated viewer initialization error: {e}")
            return False
    
    def _initialize_3d_viewer(self) -> bool:
        """Open3Dビューワー初期化"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("Geocussion 3D Viewer", width=1280, height=720)
            
            # 空の点群作成
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            
            # レンダリングオプション
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = [0.1, 0.1, 0.1]
            render_option.show_coordinate_frame = True
            
            return True
            
        except Exception as e:
            print(f"3D viewer initialization error: {e}")
            return False
    
    def run(self) -> bool:
        """メインループ実行"""
        # ヘッドレスモードの場合は初期化をスキップ
        if self.headless_mode:
            print("\n🖥️  ヘッドレスモード: カメラ初期化をスキップ")
            # パイプラインのみ初期化（カメラなし）
            if not self.pipeline.initialize(None):
                print("Failed to initialize pipeline for headless mode")
                return False
            self.is_running = True
            self._run_headless_mode()
            return True
        
        # 通常モード（GUIモード）では既に初期化済みのはず
        if not self.is_initialized:
            print("⚠️  ビューワーが初期化されていません - 再初期化を試行")
            if not self.initialize():
                print("Failed to initialize integrated viewer")
                return False
        
        self.is_running = True
        print("\n統合Geocussionビューワー開始!")
        
        # 通常モード実行
        try:
            import cv2
            
            while self.is_running:
                if not self._process_frame():
                    break
                
                # キーボード入力チェック（OpenCV）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' または ESC
                    print("終了キーが押されました")
                    break
                elif key != 255:  # 何かキーが押された
                    self._handle_key_event(key)
                
                # Open3Dビューワーのイベント処理
                if HAS_OPEN3D and self.vis:
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    
            return True
                    
        except KeyboardInterrupt:
            print("\nユーザーによる中断")
            return True
        except Exception as e:
            print(f"\nメインループエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()
    
    def _run_headless_mode(self) -> None:
        """ヘッドレスモード実行"""
        import time
        
        print(f"\n🖥️  ヘッドレスモード開始 - GUI無効化によるFPS最適化")
        print(f"⏱️  実行時間: {self.headless_duration}秒")
        print("=" * 50)
        
        start_time = time.time()
        frame_count = 0
        fps_samples = []
        
        try:
            while True:
                frame_start = time.time()
                
                # フレーム処理（GUI無し）
                success = self._process_frame_headless()
                
                frame_end = time.time()
                frame_time = frame_end - frame_start
                
                if success:
                    frame_count += 1
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_samples.append(current_fps)
                
                    # 5秒間隔で統計表示
                    elapsed = frame_end - start_time
                    if frame_count % 150 == 0:  # 約5秒間隔
                        avg_fps = sum(fps_samples[-100:]) / len(fps_samples[-100:]) if fps_samples else 0
                        print(f"📊 [{elapsed:.1f}s] フレーム: {frame_count}, 平均FPS: {avg_fps:.1f}, 現在FPS: {current_fps:.1f}")
                
                # 実行時間チェック
                if time.time() - start_time >= self.headless_duration:
                    break
            
        except KeyboardInterrupt:
            print("\n⏹️  ユーザーによる中断")
        except Exception as e:
            print(f"\n❌ ヘッドレスモード実行エラー: {e}")
        
        # 統計計算
        execution_time = time.time() - start_time
        avg_fps = frame_count / execution_time if execution_time > 0 else 0
        max_fps = max(fps_samples) if fps_samples else 0
        min_fps = min(fps_samples) if fps_samples else 0
        
        # 結果表示
        print("\n" + "=" * 50)
        print("🏁 ヘッドレスモード 実行結果")
        print("=" * 50)
        print(f"⏱️  実行時間: {execution_time:.1f}秒")
        print(f"🎬 総フレーム数: {frame_count}")
        print(f"🚀 平均FPS: {avg_fps:.1f}")
        print(f"📈 最大FPS: {max_fps:.1f}")
        print(f"📉 最小FPS: {min_fps:.1f}")
        print()
    
    def _process_frame_headless(self) -> bool:
        """ヘッドレス専用フレーム処理（GUI描画なし）"""
        import time
        try:
            # ヘッドレスモード用モックデータ生成
            if not self.camera:
                # モック深度・カラー画像生成
                import numpy as np
                from ..types import FrameData, CameraIntrinsics
                
                depth_image = np.random.randint(500, 2000, (240, 424), dtype=np.uint16)
                color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # モックIntrinsics
                intrinsics = CameraIntrinsics(
                    fx=209.2152099609375,
                    fy=209.2152099609375,
                    cx=212.3312530517578,
                    cy=119.83750915527344,
                    width=424,
                    height=240
                )
                
                # モックフレームオブジェクト
                class MockFrame:
                    def __init__(self, data):
                        self.data = data
                    def get_data(self):
                        return self.data
                
                # FrameData作成（正しいフィールド名を使用）
                frame_data = FrameData(
                    depth_frame=MockFrame(depth_image),
                    color_frame=MockFrame(color_image),
                    timestamp_ms=time.time() * 1000,
                    frame_number=getattr(self, '_mock_frame_number', 0)
                )
                
                # モックカメラのintrinsicsを設定（InputStageで使用）
                if not hasattr(self, '_mock_camera_initialized'):
                    # InputStageにモックカメラを設定
                    class MockCamera:
                        def __init__(self, intrinsics):
                            self.depth_intrinsics = intrinsics
                    
                    mock_camera = MockCamera(intrinsics)
                    self.pipeline._orchestrator.input_stage.camera = mock_camera
                    
                    # DetectionStageにもintrinsicsを設定
                    self.pipeline._orchestrator.detection_stage.camera_intrinsics = intrinsics
                    self._mock_camera_initialized = True
                
                # フレーム番号をインクリメント
                self._mock_frame_number = getattr(self, '_mock_frame_number', 0) + 1
                
                # パイプライン処理実行（モックデータ付き）
                results = self.pipeline.process_frame(frame_data)
                
                # モックデータの場合は処理遅延をシミュレート
                time.sleep(0.015)  # 15ms
            else:
                # 実カメラからフレーム取得
                results = self.pipeline.process_frame()
            
            if not results:
                return True  # フレーム取得失敗は継続
            
            self.current_results = results
            
            # ヘッドレスでは表示処理をスキップ
            # パフォーマンス統計のみ更新
            if hasattr(results, 'performance_stats') and results.performance_stats:
                # 統計情報は内部で管理
                pass
            
            return True
            
        except Exception as e:
            # ヘッドレスではエラーでも継続
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            self._error_count += 1
            
            if self._error_count <= 3:  # 最初の3回のみエラー表示
                print(f"⚠️  フレーム処理警告: {e}")
            return True  # エラーでも継続
    
    def _process_frame(self) -> bool:
        """フレーム処理（通常モード）"""
        try:
            # パイプライン処理実行（ヘッドレスと同じロジック）
            results = self.pipeline.process_frame()
            if not results:
                return True  # フレーム取得失敗は継続
            
            self.current_results = results
        
            # カラー画像抽出（修正版メソッドを使用）
            color_image = self._extract_color_image(results.frame_data if hasattr(results, 'frame_data') and results.frame_data else None)
            
            # 可視化更新
            self._update_visualization(results, color_image)
            
            return True
        
        except Exception as e:
            print(f"Frame processing error: {e}")
            return False
        
    def _update_visualization(self, results, color_image) -> None:
        """可視化更新"""
        try:
            # 結果を保存
            self.current_results = results
            
            # RGB表示処理
            if color_image is not None:
                self._process_rgb_display_with_image(color_image)
            
            # 点群表示処理
            if HAS_OPEN3D and self.vis:
                self._process_pointcloud_display()
                
        except Exception as e:
            print(f"Visualization update error: {e}")
    
    def _process_rgb_display_with_image(self, color_image) -> None:
        """RGB表示処理（画像付き）"""
        try:
            if color_image is None:
                return
            
            import cv2
            
            # 手検出結果の描画
            if hasattr(self.current_results, 'hands_2d') and self.current_results.hands_2d:
                for hand in self.current_results.hands_2d:
                    if hasattr(hand, 'landmarks'):
                        for landmark in hand.landmarks:
                            x = int(landmark.x * color_image.shape[1])
                            y = int(landmark.y * color_image.shape[0])
                            cv2.circle(color_image, (x, y), 3, (0, 255, 0), -1)
            
            # 衝突結果の描画
            if hasattr(self.current_results, 'collision_events') and self.current_results.collision_events:
                cv2.putText(color_image, f"Collisions: {len(self.current_results.collision_events)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 画像表示
            cv2.imshow('Geocussion-SP Color', color_image)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"RGB display error: {e}")
    
    def _process_pointcloud_display(self) -> bool:
        """点群表示処理"""
        if not self.current_results:
            return True
        
        # メッシュ可視化更新
        if self.current_results.mesh:
            self._update_mesh_visualization(self.current_results.mesh)
        
        # 衝突可視化更新
        if self.pipeline_config.enable_collision_visualization:
            self._update_collision_visualization()
        
        # 手マーカー更新
        self._update_hand_markers()
        
        # Open3Dビューワー更新
        if self.vis:
            self.vis.poll_events()
            self.vis.update_renderer()
        
        return True
    
    def _draw_hand_detections(self, image: np.ndarray) -> None:
        """手検出結果描画"""
        height, width = image.shape[:2]
        
        # 2D手検出結果描画
        for hand_2d in self.current_results.hands_2d:
            # バウンディングボックス
            bbox = hand_2d.bounding_box
            scale_x = width / 640
            scale_y = height / 480
            
            bbox_scaled = (
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            )
            
            cv2.rectangle(image, 
                         (bbox_scaled[0], bbox_scaled[1]), 
                         (bbox_scaled[0] + bbox_scaled[2], bbox_scaled[1] + bbox_scaled[3]), 
                         (0, 255, 255), 2)
            
            # 手の情報
            cv2.putText(image, f"{hand_2d.handedness.value} ({hand_2d.confidence:.2f})",
                       (bbox_scaled[0], bbox_scaled[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 3D情報表示
        info_y = 60
        if self.current_results.hands_3d:
            cv2.putText(image, f"3D Hands: {len(self.current_results.hands_3d)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # トラッキング情報表示
        if self.current_results.tracked_hands:
            track_y = info_y + 50
            cv2.putText(image, f"Tracked: {len(self.current_results.tracked_hands)}", 
                       (10, track_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_collision_info(self, image: np.ndarray) -> None:
        """衝突情報描画"""
        y_offset = 30
        for i, event in enumerate(self.current_results.collision_events):
            text = f"Collision {i+1}: Hand {event.get('hand_id', 'Unknown')[:8]}"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
    
    def _draw_performance_overlay(self, image: np.ndarray) -> None:
        """パフォーマンス情報描画"""
        stats = self.current_results.performance_stats
        y_offset = image.shape[0] - 150
        
        texts = [
            f"Frame: {stats.get('frame_count', 0)}",
            f"Mesh: {stats.get('mesh_generation_time', 0)*1000:.1f}ms",
            f"Collision: {stats.get('collision_detection_time', 0)*1000:.1f}ms",
            f"Audio: {stats.get('audio_synthesis_time', 0)*1000:.1f}ms",
            f"Total: {stats.get('total_pipeline_time', 0)*1000:.1f}ms",
            f"Events: {stats.get('collision_events_count', 0)}",
            f"Notes: {stats.get('audio_notes_played', 0)}"
        ]
        
        for text in texts:
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def _update_mesh_visualization(self, mesh) -> None:
        """メッシュ可視化更新"""
        try:
            # 既存メッシュ削除
            for geom in self.mesh_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.mesh_geometries.clear()
            
            if mesh and hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                # Open3Dメッシュ作成
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
                
                # メッシュ色設定
                o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                o3d_mesh.compute_vertex_normals()
                
                # ビューワーに追加
                self.vis.add_geometry(o3d_mesh, reset_bounding_box=False)
                self.mesh_geometries.append(o3d_mesh)
                
        except Exception as e:
            print(f"Mesh visualization error: {e}")
    
    def _update_collision_visualization(self) -> None:
        """衝突可視化更新"""
        try:
            # 既存衝突ジオメトリ削除
            for geom in self.collision_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.collision_geometries.clear()
            
            if self.current_results and self.current_results.collision_events:
                for event in self.current_results.collision_events:
                    if 'position' in event:
                        # 衝突点に球表示
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                        sphere.translate(event['position'])
                        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
                        
                        self.vis.add_geometry(sphere, reset_bounding_box=False)
                        self.collision_geometries.append(sphere)
                        
        except Exception as e:
            print(f"Collision visualization error: {e}")
    
    def _update_hand_markers(self) -> None:
        """手マーカー更新"""
        try:
            # 既存手マーカー削除
            for marker in self.hand_markers:
                self.vis.remove_geometry(marker, reset_bounding_box=False)
            self.hand_markers.clear()
            
            if self.current_results and self.current_results.tracked_hands:
                for hand in self.current_results.tracked_hands:
                    if hasattr(hand, 'palm_center') and hand.palm_center is not None:
                        # 手の中心に球表示
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                        sphere.translate(hand.palm_center)
                        sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 緑色
                        
                        self.vis.add_geometry(sphere, reset_bounding_box=False)
                        self.hand_markers.append(sphere)
                        
        except Exception as e:
            print(f"Hand marker update error: {e}")
    
    def _handle_key_event(self, key: int) -> None:
        """キーイベント処理"""
        # キー押下イベントを発行
        self.event_dispatcher.publish(KeyPressedEvent(
            key_code=key,
            shift=False,  # TODO: 修飾キーの検出
            ctrl=False,
            alt=False
        ))
        if key == ord('h') or key == ord('H'):
            print(self.help_text)
        
        elif key == ord('m') or key == ord('M'):
            self.pipeline_config.enable_mesh_generation = not self.pipeline_config.enable_mesh_generation
            self.pipeline.update_config(self.pipeline_config)
            status = "有効" if self.pipeline_config.enable_mesh_generation else "無効"
            print(f"メッシュ生成: {status}")
        
        elif key == ord('c') or key == ord('C'):
            self.pipeline_config.enable_collision_detection = not self.pipeline_config.enable_collision_detection
            self.pipeline.update_config(self.pipeline_config)
            status = "有効" if self.pipeline_config.enable_collision_detection else "無効"
            print(f"衝突検出: {status}")
        
        elif key == ord('v') or key == ord('V'):
            self.pipeline_config.enable_collision_visualization = not self.pipeline_config.enable_collision_visualization
            status = "有効" if self.pipeline_config.enable_collision_visualization else "無効"
            print(f"衝突可視化: {status}")
        
        elif key == ord('n') or key == ord('N'):
            print("メッシュ強制更新...")
            self.pipeline.force_mesh_update()
        
        elif key == ord('+') or key == ord('='):
            self.pipeline_config.sphere_radius = min(self.pipeline_config.sphere_radius + 0.01, 0.2)
            self.pipeline.update_config(self.pipeline_config)
            print(f"球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
        
        elif key == ord('-') or key == ord('_'):
            self.pipeline_config.sphere_radius = max(self.pipeline_config.sphere_radius - 0.01, 0.01)
            self.pipeline.update_config(self.pipeline_config)
            print(f"球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
        
        elif key == ord('p') or key == ord('P'):
            self.show_performance = not self.show_performance
            status = "表示" if self.show_performance else "非表示"
            print(f"パフォーマンス統計: {status}")
        
        elif key == ord('a') or key == ord('A'):
            self.pipeline_config.enable_audio_synthesis = not self.pipeline_config.enable_audio_synthesis
            self.pipeline.update_config(self.pipeline_config)
            status = "有効" if self.pipeline_config.enable_audio_synthesis else "無効"
            print(f"音響合成: {status}")
        
        elif key == ord('s') or key == ord('S'):
            if self.pipeline_config.enable_audio_synthesis:
                self._cycle_audio_scale()
        
        elif key == ord('i') or key == ord('I'):
            if self.pipeline_config.enable_audio_synthesis:
                self._cycle_audio_instrument()
        
        elif key == ord('1'):
            self.pipeline_config.audio_master_volume = max(0.0, self.pipeline_config.audio_master_volume - 0.1)
            self.pipeline.update_config(self.pipeline_config)
            print(f"音量: {self.pipeline_config.audio_master_volume:.1f}")
        
        elif key == ord('2'):
            self.pipeline_config.audio_master_volume = min(1.0, self.pipeline_config.audio_master_volume + 0.1)
            self.pipeline.update_config(self.pipeline_config)
            print(f"音量: {self.pipeline_config.audio_master_volume:.1f}")
    
    def _cycle_audio_scale(self) -> None:
        """音階切り替え"""
        scales = list(ScaleType)
        current_index = scales.index(self.pipeline_config.audio_scale)
        next_index = (current_index + 1) % len(scales)
        self.pipeline_config.audio_scale = scales[next_index]
        self.pipeline.update_config({'audio_scale': self.pipeline_config.audio_scale})
        print(f"音階: {self.pipeline_config.audio_scale.value}")
    
    def _cycle_audio_instrument(self) -> None:
        """楽器切り替え"""
        instruments = list(InstrumentType)
        current_index = instruments.index(self.pipeline_config.audio_instrument)
        next_index = (current_index + 1) % len(instruments)
        self.pipeline_config.audio_instrument = instruments[next_index]
        self.pipeline.update_config({'audio_instrument': self.pipeline_config.audio_instrument})
        print(f"楽器: {self.pipeline_config.audio_instrument.value}")
    
    def _extract_color_image(self, frame_data):
        """カラー画像抽出（MJPG対応・C-contiguous完全対応版）"""
        try:
            if frame_data is None:
                return self._create_fallback_image()
                
            if not hasattr(frame_data, 'color_frame') or frame_data.color_frame is None:
                return self._create_fallback_image()
                
            if not hasattr(self.camera, 'has_color') or not self.camera.has_color:
                return self._create_fallback_image()
            
            import cv2
            from ..types import OBFormat
            
            try:
                color_frame = frame_data.color_frame
                frame_format = getattr(color_frame, 'get_format', lambda: OBFormat.RGB)()
                

                
                color_image = None
                
                if str(frame_format) == "OBFormat.MJPG" or str(frame_format) == "MJPG":
                    # MJPG形式の場合：JPEGデコードが必要
                    try:
                        # get_data()でJPEGバイナリデータを取得
                        jpeg_data = color_frame.get_data()
                        if jpeg_data is None:
                            if not hasattr(self, '_mjpg_viewer_no_data_error_shown'):
                                print("MJPG viewer: No data from color frame")
                                self._mjpg_viewer_no_data_error_shown = True
                            return self._create_fallback_image()
                        
                        # データサイズを確認
                        data_size = 0
                        jpeg_bytes = None
                        
                        # バイナリデータをnumpy配列に変換（複数の方法で試行）
                        try:
                            if hasattr(jpeg_data, 'tobytes'):
                                jpeg_bytes = jpeg_data.tobytes()
                            elif hasattr(jpeg_data, '__bytes__'):
                                jpeg_bytes = bytes(jpeg_data)
                            elif hasattr(jpeg_data, '__array__'):
                                jpeg_array_raw = np.array(jpeg_data, dtype=np.uint8)
                                jpeg_bytes = jpeg_array_raw.tobytes()
                            else:
                                # 最後の手段：直接bytes()を試行
                                jpeg_bytes = bytes(jpeg_data)
                            
                            data_size = len(jpeg_bytes)
                            
                        except Exception as data_convert_error:
                            if not hasattr(self, '_mjpg_viewer_data_convert_error_shown'):
                                print(f"MJPG viewer data conversion failed: {data_convert_error}")
                                self._mjpg_viewer_data_convert_error_shown = True
                            return self._create_fallback_image()
                        
                        # データサイズが妥当かチェック
                        if data_size < 100:  # JPEGは最低でも100バイト以上必要
                            if not hasattr(self, '_mjpg_viewer_small_data_error_shown'):
                                print(f"MJPG viewer data too small: {data_size} bytes")
                                self._mjpg_viewer_small_data_error_shown = True
                            return self._create_fallback_image()
                        
                        # OpenCVでJPEGデコード
                        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        bgr_image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
                        
                        if bgr_image is None:
                            if not hasattr(self, '_mjpg_viewer_opencv_decode_error_shown'):
                                print(f"OpenCV MJPG decode failed in viewer (data size: {data_size} bytes)")
                                self._mjpg_viewer_opencv_decode_error_shown = True
                            return self._create_fallback_image()
                        
                        # 画像サイズを確認
                        if bgr_image.shape[0] == 0 or bgr_image.shape[1] == 0:
                            if not hasattr(self, '_mjpg_viewer_zero_size_error_shown'):
                                print(f"MJPG viewer decoded to zero size image: {bgr_image.shape}")
                                self._mjpg_viewer_zero_size_error_shown = True
                            return self._create_fallback_image()
                        
                        # BGRからRGBに変換
                        color_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                        # C-contiguous配列として確実に作成
                        color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
                        
                        # 成功メッセージは初回のみ表示
                        if not hasattr(self, '_mjpg_viewer_success_shown'):
                            print(f"✅ MJPG viewer processing successful: {color_image.shape} ({data_size} bytes)")
                            self._mjpg_viewer_success_shown = True
                        
                    except Exception as e:
                        # エラーメッセージは初回のみ表示
                        if not hasattr(self, '_mjpg_viewer_error_shown'):
                            print(f"MJPG processing error in viewer: {e}")
                            self._mjpg_viewer_error_shown = True
                        return self._create_fallback_image()
                
                elif str(frame_format) in ["OBFormat.RGB", "RGB"]:
                    # RGB形式の場合
                    try:
                        raw_data = color_frame.get_data()
                        if raw_data is None:
                            return self._create_fallback_image()
                        
                        # カメラの解像度情報を取得
                        width = getattr(color_frame, 'get_width', lambda: 1280)()
                        height = getattr(color_frame, 'get_height', lambda: 720)()
                        
                        # numpy配列に変換
                        if hasattr(raw_data, 'tobytes'):
                            data_bytes = raw_data.tobytes()
                        else:
                            data_bytes = bytes(raw_data)
                        
                        color_array = np.frombuffer(data_bytes, dtype=np.uint8)
                        color_image = color_array.reshape((height, width, 3))
                        # C-contiguous配列として確実に作成
                        color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
                        
                    except Exception as e:
                        # エラーメッセージは初回のみ表示
                        if not hasattr(self, '_rgb_viewer_error_shown'):
                            print(f"RGB processing error in viewer: {e}")
                            self._rgb_viewer_error_shown = True
                        return self._create_fallback_image()
                
                elif str(frame_format) in ["OBFormat.BGR", "BGR"]:
                    # BGR形式の場合
                    try:
                        raw_data = color_frame.get_data()
                        if raw_data is None:
                            return self._create_fallback_image()
                        
                        # カメラの解像度情報を取得
                        width = getattr(color_frame, 'get_width', lambda: 1280)()
                        height = getattr(color_frame, 'get_height', lambda: 720)()
                        
                        # numpy配列に変換
                        if hasattr(raw_data, 'tobytes'):
                            data_bytes = raw_data.tobytes()
                        else:
                            data_bytes = bytes(raw_data)
                        
                        color_array = np.frombuffer(data_bytes, dtype=np.uint8)
                        color_image = color_array.reshape((height, width, 3))
                        # C-contiguous配列として確実に作成
                        color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
                        
                        # BGR→RGB変換
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        color_image = np.ascontiguousarray(color_image, dtype=np.uint8)
                        
                    except Exception as e:
                        # エラーメッセージは初回のみ表示
                        if not hasattr(self, '_bgr_viewer_error_shown'):
                            print(f"BGR processing error in viewer: {e}")
                            self._bgr_viewer_error_shown = True
                        return self._create_fallback_image()
                
                else:
                    # サポートされていない形式（初回のみ表示）
                    if not hasattr(self, '_unsupported_format_viewer_error_shown'):
                        print(f"Unsupported color format in viewer: {frame_format}")
                        self._unsupported_format_viewer_error_shown = True
                    return self._create_fallback_image()
                
                # 表示用にリサイズ（C-contiguous維持）
                if color_image is not None:
                    # C-contiguousを確認してからリサイズ
                    if not color_image.flags['C_CONTIGUOUS']:
                        color_image = np.ascontiguousarray(color_image)
                    
                    color_resized = cv2.resize(color_image, self.rgb_window_size)
                    # リサイズ後もC-contiguousにする
                    return np.ascontiguousarray(color_resized, dtype=np.uint8)
                
                return self._create_fallback_image()
                
            except Exception as e:
                # エラーメッセージは初回のみ表示
                if not hasattr(self, '_color_extraction_viewer_error_shown'):
                    print(f"Color frame access error in viewer: {e}")
                    self._color_extraction_viewer_error_shown = True
                return self._create_fallback_image()
            
        except Exception as e:
            # エラーメッセージは初回のみ表示
            if not hasattr(self, '_color_image_general_error_shown'):
                print(f"Color image extraction general error: {e}")
                self._color_image_general_error_shown = True
            return self._create_fallback_image()
    
    def _create_fallback_image(self):
        """フォールバック用の黒画像を作成（C-contiguous保証）"""
        try:
            fallback_image = np.zeros((self.rgb_window_size[1], self.rgb_window_size[0], 3), dtype=np.uint8)
            return np.ascontiguousarray(fallback_image)
        except Exception:
            # 最悪の場合、小さなフォールバック画像
            return np.ascontiguousarray(np.zeros((240, 320, 3), dtype=np.uint8))
    
    def cleanup(self) -> None:
        """クリーンアップ処理"""
        try:
            self.is_running = False
            self.is_initialized = False
            
            # パイプラインクリーンアップ
            if self.pipeline:
                self.pipeline.cleanup()
            
            # Open3Dビューワークリーンアップ
            if self.vis:
                self.vis.destroy_window()
            
            # OpenCVウィンドウクリーンアップ
            cv2.destroyAllWindows()
            
            print("統合ビューワークリーンアップ完了")
            
        except Exception as e:
            print(f"Cleanup error: {e}")