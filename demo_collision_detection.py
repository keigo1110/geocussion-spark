#!/usr/bin/env python3
"""
Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）
手検出 + 地形メッシュ生成 + 衝突検出 + 音響生成の完全な処理パイプライン

使用方法:
    python demo_collision_detection.py

機能:
    - RGB画像 + 深度画像 + 3D点群の同時表示
    - 2D手検出 (MediaPipe) with RGB画像での可視化
    - 3D投影とカルマンフィルタトラッキング
    - リアルタイム地形メッシュ生成
    - 球-三角形衝突検出
    - 接触点と衝突イベントの可視化
    - リアルタイム音響合成（pyo）
    - パフォーマンス計測とボトルネック分析
"""

import sys
import os
import argparse
import time
import threading
from typing import Optional, List, Dict
import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    # 既存の実装をインポート
    from src.debug.dual_viewer import DualViewer
    from src.input.depth_filter import FilterType
    
    # 地形メッシュ生成
    from src.mesh.projection import PointCloudProjector, ProjectionMethod
    from src.mesh.delaunay import DelaunayTriangulator
    from src.mesh.simplify import MeshSimplifier
    from src.mesh.index import SpatialIndex, IndexType
    
    # 衝突検出
    from src.collision.search import CollisionSearcher
    from src.collision.sphere_tri import SphereTriangleCollision
    from src.collision.events import CollisionEventQueue
    
    # 音響生成
    from src.sound.mapping import (
        AudioMapper, AudioParameters, InstrumentType, ScaleType,
        map_collision_to_audio, batch_map_collisions
    )
    from src.sound.synth import (
        AudioSynthesizer, AudioConfig, EngineState,
        create_audio_synthesizer
    )
    from src.sound.voice_mgr import (
        VoiceManager, StealStrategy, SpatialMode, SpatialConfig,
        create_voice_manager, allocate_and_play
    )
    
    # デバッグ用
    import open3d as o3d
    import cv2
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)


class FullPipelineViewer(DualViewer):
    """全フェーズ統合拡張DualViewer（手検出+メッシュ生成+衝突検出+音響生成）"""
    
    def __init__(self, **kwargs):
        """初期化"""
        # 衝突検出関連の設定を追加
        self.enable_mesh_generation = kwargs.pop('enable_mesh_generation', True)
        self.enable_collision_detection = kwargs.pop('enable_collision_detection', True)
        self.enable_collision_visualization = kwargs.pop('enable_collision_visualization', True)
        self.mesh_update_interval = kwargs.pop('mesh_update_interval', 10)  # フレーム
        self.sphere_radius = kwargs.pop('sphere_radius', 0.05)  # 5cm
        
        # 音響生成関連の設定を追加
        self.enable_audio_synthesis = kwargs.pop('enable_audio_synthesis', True)
        self.audio_scale = kwargs.pop('audio_scale', ScaleType.PENTATONIC)
        self.audio_instrument = kwargs.pop('audio_instrument', InstrumentType.MARIMBA)
        self.audio_polyphony = kwargs.pop('audio_polyphony', 16)
        self.audio_master_volume = kwargs.pop('audio_master_volume', 0.7)
        
        # 親クラス初期化
        super().__init__(**kwargs)
        
        # 地形メッシュ生成コンポーネント
        self.projector = PointCloudProjector(
            resolution=0.01,  # 1cm解像度
            method=ProjectionMethod.MEDIAN_HEIGHT,
            fill_holes=True
        )
        
        self.triangulator = DelaunayTriangulator(
            adaptive_sampling=True,
            boundary_points=True,
            quality_threshold=0.3
        )
        
        self.simplifier = MeshSimplifier(
            target_reduction=0.7,  # 70%削減でリアルタイム用に軽量化
            preserve_boundary=True
        )
        
        # 衝突検出コンポーネント
        self.spatial_index: Optional[SpatialIndex] = None
        self.collision_searcher: Optional[CollisionSearcher] = None
        self.collision_tester: Optional[SphereTriangleCollision] = None
        self.event_queue = CollisionEventQueue()
        
        # 音響生成コンポーネント
        self.audio_mapper: Optional[AudioMapper] = None
        self.audio_synthesizer: Optional[AudioSynthesizer] = None
        self.voice_manager: Optional[VoiceManager] = None
        self.audio_enabled = False  # 音響エンジンの状態
        
        # 状態管理
        self.current_mesh = None
        self.current_collision_points = []
        self.frame_counter = 0
        self.last_mesh_update = 0
        
        # パフォーマンス統計
        self.perf_stats = {
            'mesh_generation_time': 0.0,
            'collision_detection_time': 0.0,
            'audio_synthesis_time': 0.0,
            'total_pipeline_time': 0.0,
            'collision_events_count': 0,
            'audio_notes_played': 0,
            'frame_count': 0
        }
        
        # メッシュとコリジョンの可視化オブジェクト
        self.mesh_geometries = []
        self.collision_geometries = []
        
        # 音響システム初期化
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
        
        print("全フェーズ統合ビューワーが初期化されました")
        print(f"  - メッシュ生成: {'有効' if self.enable_mesh_generation else '無効'}")
        print(f"  - 衝突検出: {'有効' if self.enable_collision_detection else '無効'}")
        print(f"  - 接触点可視化: {'有効' if self.enable_collision_visualization else '無効'}")
        print(f"  - 球半径: {self.sphere_radius*100:.1f}cm")
        print(f"  - 音響合成: {'有効' if self.enable_audio_synthesis else '無効'}")
        if self.enable_audio_synthesis:
            print(f"    - 音階: {self.audio_scale.value}")
            print(f"    - 楽器: {self.audio_instrument.value}")
            print(f"    - ポリフォニー: {self.audio_polyphony}")
            print(f"    - 音量: {self.audio_master_volume:.1f}")
            print(f"    - エンジン状態: {'動作中' if self.audio_enabled else '停止中'}")
    
    def update_help_text(self):
        """ヘルプテキストを更新（衝突検出機能を追加）"""
        super().update_help_text()
        
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
        """キーイベント処理（衝突検出機能を追加）"""
        # 親クラスのキーイベント処理
        if super().handle_key_event(key):
            return True
        
        # 衝突検出関連のキーイベント
        if key == ord('m') or key == ord('M'):
            self.enable_mesh_generation = not self.enable_mesh_generation
            status = "有効" if self.enable_mesh_generation else "無効"
            print(f"メッシュ生成: {status}")
            return True
            
        elif key == ord('c') or key == ord('C'):
            self.enable_collision_detection = not self.enable_collision_detection
            status = "有効" if self.enable_collision_detection else "無効"
            print(f"衝突検出: {status}")
            return True
            
        elif key == ord('v') or key == ord('V'):
            self.enable_collision_visualization = not self.enable_collision_visualization
            status = "有効" if self.enable_collision_visualization else "無効"
            print(f"衝突可視化: {status}")
            self._update_visualization()
            return True
            
        elif key == ord('n') or key == ord('N'):
            print("メッシュを強制更新中...")
            self._force_mesh_update()
            return True
            
        elif key == ord('+') or key == ord('='):
            self.sphere_radius = min(self.sphere_radius + 0.01, 0.2)
            print(f"球半径: {self.sphere_radius*100:.1f}cm")
            return True
            
        elif key == ord('-') or key == ord('_'):
            self.sphere_radius = max(self.sphere_radius - 0.01, 0.01)
            print(f"球半径: {self.sphere_radius*100:.1f}cm")
            return True
            
        elif key == ord('p') or key == ord('P'):
            self._print_performance_stats()
            return True
        
        # 音響生成関連のキーイベント
        elif key == ord('a') or key == ord('A'):
            self.enable_audio_synthesis = not self.enable_audio_synthesis
            if self.enable_audio_synthesis:
                self._initialize_audio_system()
            else:
                self._shutdown_audio_system()
            status = "有効" if self.enable_audio_synthesis else "無効"
            print(f"音響合成: {status}")
            return True
            
        elif key == ord('s') or key == ord('S'):
            if self.enable_audio_synthesis:
                self._cycle_audio_scale()
            return True
            
        elif key == ord('i') or key == ord('I'):
            if self.enable_audio_synthesis:
                self._cycle_audio_instrument()
            return True
            
        elif key == ord('1'):
            if self.enable_audio_synthesis and self.audio_synthesizer:
                self.audio_master_volume = max(0.0, self.audio_master_volume - 0.1)
                self.audio_synthesizer.update_master_volume(self.audio_master_volume)
                print(f"音量: {self.audio_master_volume:.1f}")
            return True
            
        elif key == ord('2'):
            if self.enable_audio_synthesis and self.audio_synthesizer:
                self.audio_master_volume = min(1.0, self.audio_master_volume + 0.1)
                self.audio_synthesizer.update_master_volume(self.audio_master_volume)
                print(f"音量: {self.audio_master_volume:.1f}")
            return True
            
        elif key == ord('r') or key == ord('R'):
            if self.enable_audio_synthesis:
                print("音響エンジンを再起動中...")
                self._restart_audio_system()
            return True
            
        elif key == ord('q') or key == ord('Q'):
            if self.enable_audio_synthesis and self.voice_manager:
                self.voice_manager.stop_all_voices()
                print("全音声を停止しました")
            return True
        
        return False
    
    def process_frame_with_collision(self, color_image, depth_image, points_3d):
        """フレーム処理（衝突検出まで含む完全パイプライン）"""
        pipeline_start = time.perf_counter()
        
        # 既存の手検出処理
        hands_2d, hands_3d = self.process_hands(color_image, depth_image)
        
        # フレームカウンタ更新
        self.frame_counter += 1
        self.perf_stats['frame_count'] += 1
        
        # 地形メッシュ生成（定期的に更新）
        if (self.enable_mesh_generation and 
            self.frame_counter - self.last_mesh_update >= self.mesh_update_interval and
            points_3d is not None and len(points_3d) > 100):
            
            mesh_start = time.perf_counter()
            try:
                self._update_terrain_mesh(points_3d)
                self.last_mesh_update = self.frame_counter
            except Exception as e:
                print(f"メッシュ生成エラー: {e}")
            
            mesh_time = (time.perf_counter() - mesh_start) * 1000
            self.perf_stats['mesh_generation_time'] = mesh_time
        
        # 衝突検出
        collision_events = []
        if (self.enable_collision_detection and 
            self.current_mesh is not None and 
            hands_3d and len(hands_3d) > 0):
            
            collision_start = time.perf_counter()
            try:
                collision_events = self._detect_collisions(hands_3d)
            except Exception as e:
                print(f"衝突検出エラー: {e}")
            
            collision_time = (time.perf_counter() - collision_start) * 1000
            self.perf_stats['collision_detection_time'] = collision_time
            self.perf_stats['collision_events_count'] += len(collision_events)
        
        # 音響生成
        if (self.enable_audio_synthesis and self.audio_enabled and 
            collision_events and len(collision_events) > 0):
            
            audio_start = time.perf_counter()
            try:
                audio_notes = self._generate_audio(collision_events)
                self.perf_stats['audio_notes_played'] += audio_notes
            except Exception as e:
                print(f"音響生成エラー: {e}")
            
            audio_time = (time.perf_counter() - audio_start) * 1000
            self.perf_stats['audio_synthesis_time'] = audio_time
        
        # 可視化更新
        if self.enable_collision_visualization:
            self._update_collision_visualization(collision_events)
        
        # パフォーマンス統計更新
        total_time = (time.perf_counter() - pipeline_start) * 1000
        self.perf_stats['total_pipeline_time'] = total_time
        
        # パフォーマンス情報をRGB画像に描画
        self._draw_performance_info(color_image, collision_events)
        
        return hands_2d, hands_3d, collision_events
    
    def _update_terrain_mesh(self, points_3d):
        """地形メッシュを更新"""
        if points_3d is None or len(points_3d) < 100:
            return
        
        try:
            # 1. 点群投影
            height_map = self.projector.project_points(points_3d)
            
            # 2. Delaunay三角分割
            triangle_mesh = self.triangulator.triangulate_heightmap(height_map)
            
            if triangle_mesh is None or triangle_mesh.num_triangles == 0:
                print("三角分割に失敗しました")
                return
            
            # 3. メッシュ簡略化
            simplified_mesh = self.simplifier.simplify_mesh(triangle_mesh)
            
            if simplified_mesh is None:
                simplified_mesh = triangle_mesh
            
            # 4. 空間インデックス構築
            self.spatial_index = SpatialIndex(simplified_mesh, index_type=IndexType.BVH)
            
            # 5. 衝突検出コンポーネント初期化
            self.collision_searcher = CollisionSearcher(self.spatial_index)
            self.collision_tester = SphereTriangleCollision(simplified_mesh)
            
            # メッシュ保存
            self.current_mesh = simplified_mesh
            
            # 可視化更新
            self._update_mesh_visualization(simplified_mesh)
            
            print(f"メッシュ更新完了: {simplified_mesh.num_triangles}三角形")
            
        except Exception as e:
            print(f"メッシュ生成中にエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_collisions(self, hands_3d):
        """衝突検出を実行"""
        if (self.collision_searcher is None or 
            self.collision_tester is None or 
            not hands_3d):
            return []
        
        collision_events = []
        self.current_collision_points = []
        
        for i, hand_pos in enumerate(hands_3d):
            if hand_pos is None:
                continue
            
            try:
                # 1. 空間検索
                search_result = self.collision_searcher._search_point(
                    hand_pos, self.sphere_radius
                )
                
                if len(search_result.triangle_indices) == 0:
                    continue
                
                # 2. 衝突判定
                collision_info = self.collision_tester.test_sphere_collision(
                    hand_pos, self.sphere_radius, search_result
                )
                
                if collision_info.has_collision:
                    # 3. イベント生成
                    hand_velocity = np.array([0.01, 0.0, 0.0])  # ダミー速度
                    event = self.event_queue.create_event(
                        collision_info, f"hand_{i}", hand_pos, hand_velocity
                    )
                    
                    if event:
                        collision_events.append(event)
                        
                        # 接触点を記録
                        for contact in collision_info.contact_points:
                            self.current_collision_points.append({
                                'position': contact.position,
                                'normal': contact.normal,
                                'depth': contact.depth,
                                'hand_id': f"hand_{i}"
                            })
            
            except Exception as e:
                print(f"手{i}の衝突検出でエラー: {e}")
        
        return collision_events
    
    def _update_mesh_visualization(self, mesh):
        """メッシュ可視化を更新"""
        if not hasattr(self, 'vis') or self.vis is None:
            return
        
        # 既存のメッシュジオメトリを削除
        for geom in self.mesh_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.mesh_geometries.clear()
        
        try:
            # Open3Dメッシュを作成
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
            
            # 法線計算
            o3d_mesh.compute_vertex_normals()
            
            # 半透明のマテリアル設定
            o3d_mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 薄青色
            
            # ワイヤーフレーム表示
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
            wireframe.paint_uniform_color([0.3, 0.3, 0.7])  # 青色
            
            # ジオメトリを追加
            self.vis.add_geometry(o3d_mesh, reset_bounding_box=False)
            self.vis.add_geometry(wireframe, reset_bounding_box=False)
            
            self.mesh_geometries.extend([o3d_mesh, wireframe])
            
        except Exception as e:
            print(f"メッシュ可視化エラー: {e}")
    
    def _update_collision_visualization(self, collision_events):
        """衝突可視化を更新"""
        if not hasattr(self, 'vis') or self.vis is None:
            return
        
        # 既存の衝突ジオメトリを削除
        for geom in self.collision_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.collision_geometries.clear()
        
        if not self.enable_collision_visualization:
            return
        
        try:
            # 接触点を可視化
            for contact in self.current_collision_points:
                # 接触点（球）
                contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                contact_sphere.translate(contact['position'])
                contact_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
                
                # 法線ベクトル（線分）
                normal_end = contact['position'] + contact['normal'] * 0.05
                normal_line = o3d.geometry.LineSet()
                normal_line.points = o3d.utility.Vector3dVector([
                    contact['position'], normal_end
                ])
                normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
                normal_line.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
                
                self.vis.add_geometry(contact_sphere, reset_bounding_box=False)
                self.vis.add_geometry(normal_line, reset_bounding_box=False)
                
                self.collision_geometries.extend([contact_sphere, normal_line])
            
            # 衝突球を可視化（手の位置）
            if hasattr(self, 'tracked_hands') and self.tracked_hands:
                for hand_id, hand_data in self.tracked_hands.items():
                    if hand_data['position_3d'] is not None:
                        hand_sphere = o3d.geometry.TriangleMesh.create_sphere(
                            radius=self.sphere_radius
                        )
                        hand_sphere.translate(hand_data['position_3d'])
                        hand_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 緑色（半透明）
                        
                        # ワイヤーフレーム表示
                        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(hand_sphere)
                        wireframe.paint_uniform_color([0.0, 0.8, 0.0])
                        
                        self.vis.add_geometry(wireframe, reset_bounding_box=False)
                        self.collision_geometries.append(wireframe)
        
        except Exception as e:
            print(f"衝突可視化エラー: {e}")
    
    def _update_visualization(self):
        """可視化全体を更新"""
        if self.current_mesh and self.enable_collision_visualization:
            self._update_mesh_visualization(self.current_mesh)
        self._update_collision_visualization([])
    
    def _force_mesh_update(self):
        """メッシュ強制更新"""
        self.last_mesh_update = 0  # 次フレームで更新されるようにリセット
    
    def _draw_performance_info(self, color_image, collision_events):
        """パフォーマンス情報をRGB画像に描画"""
        if color_image is None:
            return
        
        # 基本情報
        info_lines = [
            f"Frame: {self.frame_counter}",
            f"Pipeline: {self.perf_stats['total_pipeline_time']:.1f}ms",
            f"Mesh Gen: {self.perf_stats['mesh_generation_time']:.1f}ms",
            f"Collision: {self.perf_stats['collision_detection_time']:.1f}ms",
            f"Audio: {self.perf_stats['audio_synthesis_time']:.1f}ms",
            f"Events: {len(collision_events)}",
            f"Sphere R: {self.sphere_radius*100:.1f}cm"
        ]
        
        # メッシュ情報
        if self.current_mesh:
            info_lines.append(f"Triangles: {self.current_mesh.num_triangles}")
        
        # 接触点情報
        if self.current_collision_points:
            info_lines.append(f"Contacts: {len(self.current_collision_points)}")
        
        # 音響情報
        if self.enable_audio_synthesis:
            audio_status = "ON" if self.audio_enabled else "OFF"
            info_lines.append(f"Audio: {audio_status}")
            if self.audio_enabled and self.voice_manager:
                active_voices = len(self.voice_manager.active_voices)
                info_lines.append(f"Voices: {active_voices}/{self.audio_polyphony}")
        
        # 描画
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(color_image, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 衝突イベント情報
        if collision_events:
            cv2.putText(color_image, "COLLISION DETECTED!", 
                       (10, color_image.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 音響再生情報
        if self.enable_audio_synthesis and self.audio_enabled and collision_events:
            cv2.putText(color_image, f"PLAYING AUDIO ({self.audio_instrument.value})", 
                       (10, color_image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _print_performance_stats(self):
        """パフォーマンス統計を印刷"""
        print("\n" + "="*50)
        print("パフォーマンス統計")
        print("="*50)
        print(f"総フレーム数: {self.perf_stats['frame_count']}")
        print(f"現在のフレーム: {self.frame_counter}")
        print(f"パイプライン時間: {self.perf_stats['total_pipeline_time']:.2f}ms")
        print(f"メッシュ生成時間: {self.perf_stats['mesh_generation_time']:.2f}ms")
        print(f"衝突検出時間: {self.perf_stats['collision_detection_time']:.2f}ms")
        print(f"音響生成時間: {self.perf_stats['audio_synthesis_time']:.2f}ms")
        print(f"総衝突イベント数: {self.perf_stats['collision_events_count']}")
        print(f"総音響ノート数: {self.perf_stats['audio_notes_played']}")
        
        if self.current_mesh:
            print(f"現在のメッシュ: {self.current_mesh.num_triangles}三角形")
        
        print(f"球半径: {self.sphere_radius*100:.1f}cm")
        
        # 音響統計
        if self.enable_audio_synthesis:
            print(f"音響合成: {'有効' if self.audio_enabled else '無効'}")
            if self.audio_enabled:
                print(f"  - 音階: {self.audio_scale.value}")
                print(f"  - 楽器: {self.audio_instrument.value}")
                print(f"  - 音量: {self.audio_master_volume:.1f}")
                if self.voice_manager:
                    voice_stats = self.voice_manager.get_performance_stats()
                    print(f"  - アクティブボイス: {voice_stats['current_active_voices']}/{self.audio_polyphony}")
                    print(f"  - 総作成ボイス: {voice_stats['total_voices_created']}")
                    print(f"  - ボイススティール: {voice_stats['total_voices_stolen']}")
        
        print("="*50)
    
    def update_frame(self, color_image, depth_image, points_3d):
        """フレーム更新（衝突検出版）"""
        try:
            # 衝突検出を含む完全パイプライン処理
            hands_2d, hands_3d, collision_events = self.process_frame_with_collision(
                color_image, depth_image, points_3d
            )
            
            # RGB画像の可視化処理（手検出結果を描画）
            if color_image is not None:
                self.draw_hands_2d(color_image, hands_2d)
                self.draw_performance_overlay(color_image)
            
            # 3D可視化更新（点群 + メッシュ + 衝突）
            if self.vis is not None and points_3d is not None:
                self.update_point_cloud(points_3d)
                
                # 手の3D位置を可視化
                self.update_hand_3d_visualization()
        
        except Exception as e:
            print(f"フレーム更新エラー: {e}")
    
    def _initialize_audio_system(self):
        """音響システムを初期化"""
        try:
            print("音響システムを初期化中...")
            
            # 音響マッパー初期化
            self.audio_mapper = AudioMapper(
                scale=self.audio_scale,
                default_instrument=self.audio_instrument,
                pitch_range=(48, 84),  # C3-C6
                enable_adaptive_mapping=True
            )
            
            # 音響シンセサイザー初期化
            self.audio_synthesizer = create_audio_synthesizer(
                sample_rate=44100,
                buffer_size=256,
                max_polyphony=self.audio_polyphony
            )
            
            # 音響エンジン開始
            if self.audio_synthesizer.start_engine():
                # ボイス管理システム初期化
                self.voice_manager = create_voice_manager(
                    self.audio_synthesizer,
                    max_polyphony=self.audio_polyphony,
                    steal_strategy=StealStrategy.OLDEST
                )
                
                # マスターボリューム設定
                self.audio_synthesizer.update_master_volume(self.audio_master_volume)
                
                self.audio_enabled = True
                print("音響システム初期化完了")
            else:
                print("音響エンジンの開始に失敗しました")
                self.audio_enabled = False
        
        except Exception as e:
            print(f"音響システム初期化エラー: {e}")
            self.audio_enabled = False
    
    def _shutdown_audio_system(self):
        """音響システムを停止"""
        try:
            if self.voice_manager:
                self.voice_manager.stop_all_voices()
            
            if self.audio_synthesizer:
                self.audio_synthesizer.stop_engine()
            
            self.audio_enabled = False
            print("音響システムを停止しました")
        
        except Exception as e:
            print(f"音響システム停止エラー: {e}")
    
    def _restart_audio_system(self):
        """音響システムを再起動"""
        self._shutdown_audio_system()
        time.sleep(0.1)  # 短時間待機
        if self.enable_audio_synthesis:
            self._initialize_audio_system()
    
    def _generate_audio(self, collision_events):
        """衝突イベントから音響を生成"""
        if not self.audio_enabled or not self.audio_mapper or not self.voice_manager:
            return 0
        
        notes_played = 0
        
        for event in collision_events:
            try:
                # 衝突イベントを音響パラメータにマッピング
                audio_params = self.audio_mapper.map_collision_event(event)
                
                # 空間位置設定
                spatial_position = np.array([
                    event.contact_position[0],
                    0.0,
                    event.contact_position[2]
                ])
                
                # 音響再生
                voice_id = allocate_and_play(
                    self.voice_manager,
                    audio_params,
                    priority=7,
                    spatial_position=spatial_position
                )
                
                if voice_id:
                    notes_played += 1
            
            except Exception as e:
                print(f"音響生成エラー（イベント: {event.event_id}）: {e}")
        
        # 終了したボイスのクリーンアップ
        if self.voice_manager:
            self.voice_manager.cleanup_finished_voices()
        
        return notes_played
    
    def _cycle_audio_scale(self):
        """音階を循環切り替え"""
        scales = list(ScaleType)
        current_index = scales.index(self.audio_scale)
        next_index = (current_index + 1) % len(scales)
        self.audio_scale = scales[next_index]
        
        if self.audio_mapper:
            self.audio_mapper.set_scale(self.audio_scale)
        
        print(f"音階を切り替え: {self.audio_scale.value}")
    
    def _cycle_audio_instrument(self):
        """楽器を循環切り替え"""
        instruments = list(InstrumentType)
        current_index = instruments.index(self.audio_instrument)
        next_index = (current_index + 1) % len(instruments)
        self.audio_instrument = instruments[next_index]
        
        if self.audio_mapper:
            self.audio_mapper.default_instrument = self.audio_instrument
        
        print(f"楽器を切り替え: {self.audio_instrument.value}")
    
    def __del__(self):
        """デストラクタ - 音響システムを適切に停止"""
        try:
            if hasattr(self, 'audio_enabled') and self.audio_enabled:
                self._shutdown_audio_system()
        except Exception as e:
            print(f"デストラクタでエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python demo_collision_detection.py                    # デフォルト設定
    python demo_collision_detection.py --no-collision     # 衝突検出無効
    python demo_collision_detection.py --no-mesh          # メッシュ生成無効
    python demo_collision_detection.py --no-audio         # 音響合成無効
    python demo_collision_detection.py --sphere-radius 0.08 # 球半径8cm
    python demo_collision_detection.py --audio-instrument BELL # ベル楽器

操作方法:
    RGB Window:
        Q/ESC: 終了
        F: 深度フィルタ ON/OFF
        H: 手検出 ON/OFF
        T: トラッキング ON/OFF
        
        M: メッシュ生成 ON/OFF
        C: 衝突検出 ON/OFF
        V: 衝突可視化 ON/OFF
        N: メッシュ強制更新
        +/-: 球半径調整
        P: パフォーマンス統計表示
        
        A: 音響合成 ON/OFF
        S: 音階切り替え
        I: 楽器切り替え
        1/2: 音量調整
        R: 音響エンジン再起動
        Q: 全音声停止
    
    3D Viewer:
        マウス: 回転/パン/ズーム
        R: 視点リセット
        """
    )
    
    # 基本設定
    parser.add_argument('--no-filter', action='store_true', help='深度フィルタを無効にする')
    parser.add_argument('--no-hand-detection', action='store_true', help='手検出を無効にする')
    parser.add_argument('--no-tracking', action='store_true', help='トラッキングを無効にする')
    parser.add_argument('--gpu-mediapipe', action='store_true', help='MediaPipeでGPUを使用')
    
    # 衝突検出設定
    parser.add_argument('--no-mesh', action='store_true', help='メッシュ生成を無効にする')
    parser.add_argument('--no-collision', action='store_true', help='衝突検出を無効にする')
    parser.add_argument('--no-collision-viz', action='store_true', help='衝突可視化を無効にする')
    parser.add_argument('--mesh-interval', type=int, default=10, help='メッシュ更新間隔（フレーム数）')
    parser.add_argument('--sphere-radius', type=float, default=0.05, help='衝突検出球の半径（メートル）')
    
    # 音響生成設定
    parser.add_argument('--no-audio', action='store_true', help='音響合成を無効にする')
    parser.add_argument('--audio-scale', type=str, default='PENTATONIC', 
                       choices=['PENTATONIC', 'MAJOR', 'MINOR', 'DORIAN', 'MIXOLYDIAN', 'CHROMATIC', 'BLUES'],
                       help='音階の種類')
    parser.add_argument('--audio-instrument', type=str, default='MARIMBA',
                       choices=['MARIMBA', 'SYNTH_PAD', 'BELL', 'PLUCK', 'BASS', 'LEAD', 'PERCUSSION', 'AMBIENT'],
                       help='楽器の種類')
    parser.add_argument('--audio-polyphony', type=int, default=16, help='最大同時発音数')
    parser.add_argument('--audio-volume', type=float, default=0.7, help='マスター音量 (0.0-1.0)')
    
    # 手検出設定
    parser.add_argument('--min-confidence', type=float, default=0.7, help='最小検出信頼度 (0.0-1.0)')
    
    # 表示設定
    parser.add_argument('--update-interval', type=int, default=3, help='点群更新間隔（フレーム数）')
    parser.add_argument('--point-size', type=float, default=2.0, help='点群の点サイズ')
    parser.add_argument('--high-resolution', action='store_true', help='高解像度表示 (1280x720)')
    
    # ウィンドウサイズ
    parser.add_argument('--window-width', type=int, default=640, help='RGBウィンドウの幅')
    parser.add_argument('--window-height', type=int, default=480, help='RGBウィンドウの高さ')
    
    # テストモード
    parser.add_argument('--test', action='store_true', help='テストモードで実行')
    
    args = parser.parse_args()
    
    # 設定値検証
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("Error: --min-confidence must be between 0.0 and 1.0")
        return 1
    
    if args.sphere_radius <= 0.0 or args.sphere_radius > 0.5:
        print("Error: --sphere-radius must be between 0.0 and 0.5")
        return 1
    
    if args.audio_polyphony < 1 or args.audio_polyphony > 64:
        print("Error: --audio-polyphony must be between 1 and 64")
        return 1
    
    if args.audio_volume < 0.0 or args.audio_volume > 1.0:
        print("Error: --audio-volume must be between 0.0 and 1.0")
        return 1
    
    # 音階と楽器の列挙値変換
    try:
        audio_scale = ScaleType[args.audio_scale]
        audio_instrument = InstrumentType[args.audio_instrument]
    except KeyError as e:
        print(f"Error: Invalid audio parameter: {e}")
        return 1
    
    # 情報表示
    print("=" * 70)
    print("Geocussion-SP 全フェーズ統合デモ（Complete Pipeline）")
    print("=" * 70)
    print(f"深度フィルタ: {'無効' if args.no_filter else '有効'}")
    print(f"手検出: {'無効' if args.no_hand_detection else '有効'}")
    print(f"メッシュ生成: {'無効' if args.no_mesh else '有効'}")
    print(f"衝突検出: {'無効' if args.no_collision else '有効'}")
    if not args.no_collision:
        print(f"  - 球半径: {args.sphere_radius*100:.1f}cm")
        print(f"  - 可視化: {'無効' if args.no_collision_viz else '有効'}")
    print(f"音響合成: {'無効' if args.no_audio else '有効'}")
    if not args.no_audio:
        print(f"  - 音階: {audio_scale.value}")
        print(f"  - 楽器: {audio_instrument.value}")
        print(f"  - ポリフォニー: {args.audio_polyphony}")
        print(f"  - 音量: {args.audio_volume:.1f}")
    print("=" * 70)
    
    # テストモード
    if args.test:
        print("テストモードで実行中...")
        try:
            import unittest
            test_dir = os.path.join(PROJECT_ROOT, 'tests')
            sys.path.insert(0, test_dir)
            
            loader = unittest.TestLoader()
            suite = loader.discover(test_dir, pattern='*collision_test.py')
            
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            if result.wasSuccessful():
                print("衝突検出フェーズのテストが正常に完了しました！")
                return 0
            else:
                print(f"テスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
                return 1
        except Exception as e:
            print(f"テスト実行エラー: {e}")
            return 1
    
    # CollisionDetectionViewer実行
    try:
        viewer = FullPipelineViewer(
            enable_filter=not args.no_filter,
            enable_hand_detection=not args.no_hand_detection,
            enable_tracking=not args.no_tracking,
            enable_mesh_generation=not args.no_mesh,
            enable_collision_detection=not args.no_collision,
            enable_collision_visualization=not args.no_collision_viz,
            enable_audio_synthesis=not args.no_audio,
            update_interval=args.update_interval,
            point_size=args.point_size,
            rgb_window_size=(args.window_width, args.window_height),
            min_detection_confidence=args.min_confidence,
            use_gpu_mediapipe=args.gpu_mediapipe,
            mesh_update_interval=args.mesh_interval,
            sphere_radius=args.sphere_radius,
            audio_scale=audio_scale,
            audio_instrument=audio_instrument,
            audio_polyphony=args.audio_polyphony,
            audio_master_volume=args.audio_volume
        )
        
        print("\n全フェーズ統合ビューワーを開始します...")
        print("=" * 70)
        
        viewer.run()
        
        print("\nビューワーが正常に終了しました")
        return 0
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました")
        return 0
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 