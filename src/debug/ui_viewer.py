#!/usr/bin/env python3
"""
UI専用ビューワー（Clean Architecture適用）
責務: プレゼンテーション・ユーザーインタラクション・可視化のみ
"""

import time
import threading
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
import open3d as o3d

# パイプラインコントローラーとインターフェース
from .pipeline_controller import (
    GeocussionPipelineController, 
    PipelineConfiguration, 
    PipelineState,
    IPipelineObserver
)
# DualViewerから必要な部分のみ移植

# 音響設定用
from ..sound.mapping import ScaleType, InstrumentType


class GeocussionUIViewer(IPipelineObserver):
    """
    Geocussion UI専用ビューワー
    Clean Architecture適用: UI/プレゼンテーション層のみの責務
    """
    
    def __init__(self, **kwargs):
        """
        初期化
        
        Args:
            **kwargs: ビューワー設定（DualViewerから継承）
        """
        print("初期化中...")
        
        # パイプライン設定の構築
        self.pipeline_config = PipelineConfiguration(
            enable_filter=kwargs.pop('enable_filter', True),
            enable_hand_detection=kwargs.pop('enable_hand_detection', True),
            enable_tracking=kwargs.pop('enable_tracking', True),
            min_detection_confidence=kwargs.pop('min_detection_confidence', 0.1),
            use_gpu_mediapipe=kwargs.pop('use_gpu_mediapipe', False),
            
            enable_mesh_generation=kwargs.pop('enable_mesh_generation', True),
            mesh_update_interval=kwargs.pop('mesh_update_interval', 5),
            max_mesh_skip_frames=kwargs.pop('max_mesh_skip_frames', 60),
            
            enable_collision_detection=kwargs.pop('enable_collision_detection', True),
            enable_collision_visualization=kwargs.pop('enable_collision_visualization', True),
            sphere_radius=kwargs.pop('sphere_radius', 0.05),
            
            enable_audio_synthesis=kwargs.pop('enable_audio_synthesis', False),
            audio_scale=kwargs.pop('audio_scale', ScaleType.PENTATONIC),
            audio_instrument=kwargs.pop('audio_instrument', InstrumentType.MARIMBA),
            audio_polyphony=kwargs.pop('audio_polyphony', 4),
            audio_master_volume=kwargs.pop('audio_master_volume', 0.7)
        )
        
        # パイプラインコントローラー初期化（ビジネスロジック）
        self.pipeline_controller = GeocussionPipelineController(self.pipeline_config)
        self.pipeline_controller.add_observer(self)
        
        # UI状態管理
        self.current_frame_data: Optional[Any] = None
        self.current_results: Dict[str, Any] = {}
        self.current_performance_stats: Dict[str, Any] = {}
        
        # 可視化オブジェクト
        self.mesh_geometries = []
        self.collision_geometries = []
        
        # パフォーマンス表示フラグ
        self.show_performance = False
        
        # UI専用ビューワーの基本属性初期化
        self.rgb_window_size = kwargs.get('rgb_window_size', (640, 480))
        self.point_size = kwargs.get('point_size', 2.0)
        
        # Open3D関連
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        
        # 手マーカー（DualViewerから移植）
        self.hand_markers = []
        
        # 状態管理
        self.is_running = False
        
        # ヘルプテキスト初期化
        self.help_text = ""
        self.update_help_text()
        
        print("UI専用ビューワーが初期化されました")
        print(f"  - メッシュ生成: {'有効' if self.pipeline_config.enable_mesh_generation else '無効'}")
        print(f"  - 衝突検出: {'有効' if self.pipeline_config.enable_collision_detection else '無効'}")
        print(f"  - 接触点可視化: {'有効' if self.pipeline_config.enable_collision_visualization else '無効'}")
        print(f"  - 球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
        print(f"  - 音響合成: {'有効' if self.pipeline_config.enable_audio_synthesis else '無効'}")
        if self.pipeline_config.enable_audio_synthesis:
            print(f"    - 音階: {self.pipeline_config.audio_scale.value}")
            print(f"    - 楽器: {self.pipeline_config.audio_instrument.value}")
            print(f"    - ポリフォニー: {self.pipeline_config.audio_polyphony}")
            print(f"    - 音量: {self.pipeline_config.audio_master_volume:.1f}")
    
    def initialize(self) -> bool:
        """
        ビューワー初期化
        
        Returns:
            成功した場合True
        """
        try:
            # パイプラインコントローラー初期化
            if not self.pipeline_controller.initialize():
                print("Failed to initialize pipeline controller")
                return False
            
            # Open3Dビューワー初期化（UI部分のみ）
            if not self._initialize_3d_viewer():
                return False
            
            print("UI viewer initialized successfully")
            return True
            
        except Exception as e:
            print(f"UI viewer initialization error: {e}")
            return False
    
    def _initialize_3d_viewer(self) -> bool:
        """Open3Dビューワーを初期化"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("3D Point Cloud Viewer", width=1280, height=720)
            
            # 空の点群を作成
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
    
    def run(self) -> None:
        """メインループ実行"""
        if not self.initialize():
            print("Failed to initialize UI viewer")
            return
        
        self.is_running = True
        print("\nGeocussion UI Viewer Started!")
        
        try:
            while self.is_running:
                if not self._process_frame():
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def update_help_text(self):
        """ヘルプテキストを更新（衝突検出機能を追加）"""
        self.help_text = "=== 基本操作 ===\n"
        self.help_text += "ESC: 終了\n"
        self.help_text += "H: このヘルプ表示\n"
        
        # 衝突検出関連のキーバインドを追加
        self.help_text += "\n=== 衝突検出制御 ===\n"
        self.help_text += "M: メッシュ生成 ON/OFF\n"
        self.help_text += "C: 衝突検出 ON/OFF\n"
        self.help_text += "V: 衝突可視化 ON/OFF\n"
        self.help_text += "N: メッシュ強制更新\n"
        self.help_text += "+/-: 球半径調整\n"
        self.help_text += "P: パフォーマンス統計表示\n"
        
        # 音響生成関連のキーバインドを追加
        self.help_text += "\n=== 音響生成制御 ===\n"
        self.help_text += "A: 音響合成 ON/OFF\n"
        self.help_text += "S: 音階切り替え\n"
        self.help_text += "I: 楽器切り替え\n"
        self.help_text += "1/2: 音量調整\n"
        self.help_text += "R: 音響エンジン再起動\n"
        self.help_text += "Q: 全音声停止\n"
    
    def handle_key_event(self, key):
        """キーイベント処理（設定変更のみ）"""
        # 基本キーイベント処理
        if key == ord('h') or key == ord('H'):
            print(self.help_text)
            return True
        
        # 衝突検出関連のキーイベント
        if key == ord('m') or key == ord('M'):
            self.pipeline_config.enable_mesh_generation = not self.pipeline_config.enable_mesh_generation
            self._update_pipeline_configuration()
            status = "有効" if self.pipeline_config.enable_mesh_generation else "無効"
            print(f"メッシュ生成: {status}")
            return True
            
        elif key == ord('c') or key == ord('C'):
            self.pipeline_config.enable_collision_detection = not self.pipeline_config.enable_collision_detection
            self._update_pipeline_configuration()
            status = "有効" if self.pipeline_config.enable_collision_detection else "無効"
            print(f"衝突検出: {status}")
            return True
            
        elif key == ord('v') or key == ord('V'):
            self.pipeline_config.enable_collision_visualization = not self.pipeline_config.enable_collision_visualization
            status = "有効" if self.pipeline_config.enable_collision_visualization else "無効"
            print(f"衝突可視化: {status}")
            self._update_visualization()
            return True
            
        elif key == ord('n') or key == ord('N'):
            print("メッシュを強制更新中...")
            self.pipeline_controller.force_mesh_update()
            return True
            
        elif key == ord('+') or key == ord('='):
            self.pipeline_config.sphere_radius = min(self.pipeline_config.sphere_radius + 0.01, 0.2)
            self._update_pipeline_configuration()
            print(f"球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
            return True
            
        elif key == ord('-') or key == ord('_'):
            self.pipeline_config.sphere_radius = max(self.pipeline_config.sphere_radius - 0.01, 0.01)
            self._update_pipeline_configuration()
            print(f"球半径: {self.pipeline_config.sphere_radius*100:.1f}cm")
            return True
            
        elif key == ord('p') or key == ord('P'):
            self.show_performance = not self.show_performance
            status = "表示" if self.show_performance else "非表示"
            print(f"パフォーマンス統計: {status}")
            return True
        
        # 音響生成関連のキーイベント
        elif key == ord('a') or key == ord('A'):
            self.pipeline_config.enable_audio_synthesis = not self.pipeline_config.enable_audio_synthesis
            self._update_pipeline_configuration()
            status = "有効" if self.pipeline_config.enable_audio_synthesis else "無効"
            print(f"音響合成: {status}")
            return True
            
        elif key == ord('s') or key == ord('S'):
            if self.pipeline_config.enable_audio_synthesis:
                self._cycle_audio_scale()
            return True
            
        elif key == ord('i') or key == ord('I'):
            if self.pipeline_config.enable_audio_synthesis:
                self._cycle_audio_instrument()
            return True
            
        elif key == ord('1'):
            self.pipeline_config.audio_master_volume = max(0.0, self.pipeline_config.audio_master_volume - 0.1)
            self._update_pipeline_configuration()
            print(f"音量: {self.pipeline_config.audio_master_volume:.1f}")
            return True
            
        elif key == ord('2'):
            self.pipeline_config.audio_master_volume = min(1.0, self.pipeline_config.audio_master_volume + 0.1)
            self._update_pipeline_configuration()
            print(f"音量: {self.pipeline_config.audio_master_volume:.1f}")
            return True
        
        return False
    
    def _update_pipeline_configuration(self):
        """パイプライン設定更新"""
        self.pipeline_controller.update_configuration(self.pipeline_config)
    
    def _cycle_audio_scale(self):
        """音階切り替え"""
        scales = list(ScaleType)
        current_index = scales.index(self.pipeline_config.audio_scale)
        next_index = (current_index + 1) % len(scales)
        self.pipeline_config.audio_scale = scales[next_index]
        self._update_pipeline_configuration()
        print(f"音階: {self.pipeline_config.audio_scale.value}")
    
    def _cycle_audio_instrument(self):
        """楽器切り替え"""
        instruments = list(InstrumentType)
        current_index = instruments.index(self.pipeline_config.audio_instrument)
        next_index = (current_index + 1) % len(instruments)
        self.pipeline_config.audio_instrument = instruments[next_index]
        self._update_pipeline_configuration()
        print(f"楽器: {self.pipeline_config.audio_instrument.value}")
    
    def _process_frame(self) -> bool:
        """
        フレーム処理（UIビューワーは処理結果の表示のみ）
        
        Returns:
            継続する場合True
        """
        # パイプラインコントローラーでフレーム処理実行
        results = self.pipeline_controller.process_frame()
        
        if not results:
            return True  # フレーム取得失敗は継続
        
        # 結果をUI用に保存
        self.current_results = results
        self.current_frame_data = results.get('frame_data')
        
        # RGB表示処理
        if not self._process_rgb_display(self.current_frame_data, results.get('collision_events', [])):
            return False
        
        # 点群表示処理
        if not self._process_pointcloud_display(self.current_frame_data):
            return False
        
        return True
    
    def _process_rgb_display(self, frame_data, collision_events=None) -> bool:
        """RGB表示処理（UI専用）"""
        if frame_data is None or frame_data.color is None:
            return True
        
        color_image = frame_data.color.copy()
        
        # 手検出結果の描画
        if 'hands_2d' in self.current_results:
            color_image = self._draw_hand_detections(
                color_image,
                self.current_results.get('hands_2d', []),
                self.current_results.get('hands_3d', []),
                self.current_results.get('tracked_hands', [])
            )
        
        # 衝突情報の描画
        if collision_events and self.pipeline_config.enable_collision_visualization:
            self._draw_collision_info(color_image, collision_events)
        
        # パフォーマンス情報の描画
        if self.show_performance:
            self._draw_performance_overlay(color_image)
        
        # ウィンドウに表示
        cv2.imshow("Geocussion - RGB View", color_image)
        
        # キーイベント処理
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        
        if key != 255:
            self.handle_key_event(key)
        
        return True
    
    def _process_pointcloud_display(self, frame_data) -> bool:
        """点群表示処理（UI専用）"""
        if frame_data is None or not self.vis:
            return True
        
        # メッシュ可視化更新
        if 'mesh' in self.current_results and self.current_results['mesh']:
            self._update_mesh_visualization(self.current_results['mesh'])
        
        # 衝突可視化更新
        if self.pipeline_config.enable_collision_visualization:
            self._update_collision_visualization()
        
        # 手マーカー更新
        self._update_hand_markers()
        
        # Open3Dビューワー更新
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return True
    
    def _update_mesh_visualization(self, mesh):
        """メッシュ可視化更新"""
        try:
            # 既存のメッシュジオメトリを削除
            for geom in self.mesh_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.mesh_geometries.clear()
            
            if mesh and hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                # Open3Dメッシュオブジェクト作成
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
                
                # メッシュの色設定
                o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # グレー
                o3d_mesh.compute_vertex_normals()
                
                # ビューワーに追加
                self.vis.add_geometry(o3d_mesh, reset_bounding_box=False)
                self.mesh_geometries.append(o3d_mesh)
                
        except Exception as e:
            print(f"Mesh visualization error: {e}")
    
    def _update_collision_visualization(self):
        """衝突可視化更新"""
        try:
            # 既存の衝突ジオメトリを削除
            for geom in self.collision_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.collision_geometries.clear()
            
            collision_events = self.current_results.get('collision_events', [])
            
            for event in collision_events:
                if 'position' in event:
                    # 衝突点に球を表示
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.translate(event['position'])
                    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
                    
                    self.vis.add_geometry(sphere, reset_bounding_box=False)
                    self.collision_geometries.append(sphere)
                    
        except Exception as e:
            print(f"Collision visualization error: {e}")
    
    def _update_visualization(self):
        """可視化全体更新"""
        if self.pipeline_config.enable_collision_visualization:
            self._update_collision_visualization()
        else:
            # 衝突可視化を非表示
            for geom in self.collision_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.collision_geometries.clear()
    
    def _update_hand_markers(self):
        """手マーカー更新"""
        try:
            # 既存の手マーカーを削除
            for marker in self.hand_markers:
                self.vis.remove_geometry(marker, reset_bounding_box=False)
            self.hand_markers.clear()
            
            tracked_hands = self.current_results.get('tracked_hands', [])
            
            for hand in tracked_hands:
                if hasattr(hand, 'palm_center') and hand.palm_center is not None:
                    # 手の中心に球を表示
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                    sphere.translate(hand.palm_center)
                    sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 緑色
                    
                    self.vis.add_geometry(sphere, reset_bounding_box=False)
                    self.hand_markers.append(sphere)
                    
        except Exception as e:
            print(f"Hand marker update error: {e}")
    
    def _draw_collision_info(self, image: np.ndarray, collision_events: list) -> None:
        """衝突情報描画"""
        if not collision_events:
            return
        
        y_offset = 30
        for i, event in enumerate(collision_events):
            text = f"Collision {i+1}: Hand {event.get('hand_id', 'Unknown')}"
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
    
    def _draw_performance_overlay(self, image: np.ndarray) -> None:
        """パフォーマンス情報描画"""
        if not self.current_performance_stats:
            return
        
        stats = self.current_performance_stats
        y_offset = image.shape[0] - 150
        
        texts = [
            f"Frame: {stats.get('frame_count', 0)}",
            f"Mesh Time: {stats.get('mesh_generation_time', 0):.3f}s",
            f"Collision Time: {stats.get('collision_detection_time', 0):.3f}s",
            f"Audio Time: {stats.get('audio_synthesis_time', 0):.3f}s",
            f"Collisions: {stats.get('collision_events_count', 0)}",
            f"Notes: {stats.get('audio_notes_played', 0)}"
        ]
        
        for text in texts:
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def cleanup(self):
        """クリーンアップ"""
        try:
            self.is_running = False
            
            if self.pipeline_controller:
                self.pipeline_controller.cleanup_resource()
            
            # Open3Dビューワーのクリーンアップ
            if self.vis:
                self.vis.destroy_window()
                
            # OpenCVウィンドウのクリーンアップ
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    # IPipelineObserver実装
    def on_frame_processed(self, frame_data, results: Dict[str, Any]) -> None:
        """フレーム処理完了時のコールバック"""
        # UIでは特別な処理は不要（結果は既に保存済み）
        pass
    
    def on_collision_detected(self, collision_events: List) -> None:
        """衝突検出時のコールバック"""
        # UIでは特別な処理は不要（可視化は_update_collision_visualizationで処理）
        pass
    
    def on_performance_update(self, stats: Dict[str, Any]) -> None:
        """パフォーマンス統計更新時のコールバック"""
        self.current_performance_stats = stats
    
    def _draw_hand_detections(self, image: np.ndarray, hands_2d: list, hands_3d: list, tracked_hands: list) -> np.ndarray:
        """
        RGB画像に手検出結果を描画
        
        Args:
            image: 描画対象画像
            hands_2d: 2D手検出結果
            hands_3d: 3D手検出結果
            tracked_hands: トラッキング結果
            
        Returns:
            描画済み画像
        """
        height, width = image.shape[:2]
        
        # 2D手検出結果描画
        for hand_2d in hands_2d:
            # バウンディングボックス
            bbox = hand_2d.bounding_box
            # 画像サイズに合わせてスケール
            scale_x = width / 640  # カメラ解像度仮定
            scale_y = height / 480
            
            bbox_scaled = (
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            )
            
            cv2.rectangle(image, (bbox_scaled[0], bbox_scaled[1]), 
                         (bbox_scaled[0] + bbox_scaled[2], bbox_scaled[1] + bbox_scaled[3]), 
                         (0, 255, 255), 2)
            
            # 手の情報
            cv2.putText(image, f"{hand_2d.handedness.value} ({hand_2d.confidence:.2f})",
                       (bbox_scaled[0], bbox_scaled[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 主要ランドマーク描画
            for i, landmark in enumerate(hand_2d.landmarks):
                if i in [0, 4, 8, 12, 16, 20]:  # 主要ランドマークのみ
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        # 3D投影結果情報表示
        info_y = 60
        if hands_3d:
            cv2.putText(image, f"3D Hands: {len(hands_3d)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            for i, hand_3d in enumerate(hands_3d):
                palm_x, palm_y, palm_z = hand_3d.palm_center_3d
                cv2.putText(image, f"  Hand {i+1}: ({palm_x:.2f}, {palm_y:.2f}, {palm_z:.2f}m)",
                           (10, info_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # トラッキング結果表示
        if tracked_hands:
            track_y = info_y + 25 * (len(hands_3d) + 2) if hands_3d else info_y + 25
            cv2.putText(image, f"Tracked: {len(tracked_hands)}", (10, track_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for i, tracked_hand in enumerate(tracked_hands):
                try:
                    speed = tracked_hand.speed
                    cv2.putText(image, 
                               f"  ID: {tracked_hand.hand_id[:8]} Speed: {speed:.2f}m/s",
                               (10, track_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except:
                    cv2.putText(image, 
                               f"  ID: {getattr(tracked_hand, 'hand_id', 'Unknown')[:8]}",
                               (10, track_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return image