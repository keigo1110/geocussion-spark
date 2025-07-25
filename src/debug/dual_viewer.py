#!/usr/bin/env python3
"""
デュアルウィンドウビューワー
RGB可視化ウィンドウ + 点群ビューワーの同時表示
"""

import time
import threading
import concurrent.futures
from typing import Optional, Tuple, List
import numpy as np
import cv2
import open3d as o3d

# 入力フェーズと手検出フェーズのクラスをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.input.stream import OrbbecCamera, FrameData, CameraIntrinsics
from src.input.pointcloud import PointCloudConverter
from src.input.depth_filter import DepthFilter, FilterType
from src.data_types import OBFormat

# カメラファクトリーをインポート
try:
    from src.utils.camera_factory import create_camera
    HAS_CAMERA_FACTORY = True
except ImportError:
    HAS_CAMERA_FACTORY = False

# 手検出フェーズ
from src.detection.hands2d import MediaPipeHandsWrapper, HandednessType, filter_hands_by_confidence
from src.detection.hands3d import Hand3DProjector, DepthInterpolationMethod
from src.detection.tracker import Hand3DTracker, TrackingState, filter_stable_hands


class DualViewer:
    """デュアルウィンドウビューワークラス"""
    
    def __init__(
        self,
        enable_filter: bool = True,
        enable_hand_detection: bool = True,
        enable_tracking: bool = True,
        update_interval: int = 3,
        point_size: float = 2.0,
        rgb_window_size: Tuple[int, int] = (640, 480),
        min_detection_confidence: float = 0.7,
        use_gpu_mediapipe: bool = False
    ):
        """
        初期化
        
        Args:
            enable_filter: 深度フィルタを有効にするか
            enable_hand_detection: 手検出を有効にするか
            enable_tracking: 手トラッキングを有効にするか
            update_interval: 何フレームごとに点群を更新するか
            point_size: 点群の点サイズ
            rgb_window_size: RGBウィンドウサイズ
            min_detection_confidence: 最小検出信頼度
            use_gpu_mediapipe: MediaPipeでGPUを使用するか
        """
        self.enable_filter = enable_filter
        self.enable_hand_detection = enable_hand_detection
        self.enable_tracking = enable_tracking
        self.update_interval = update_interval
        self.point_size = point_size
        self.rgb_window_size = rgb_window_size
        self.min_detection_confidence = min_detection_confidence
        self.use_gpu_mediapipe = use_gpu_mediapipe
        
        # カメラとコンバーター
        self.camera: Optional[OrbbecCamera] = None
        self.pointcloud_converter: Optional[PointCloudConverter] = None
        self.depth_filter: Optional[DepthFilter] = None
        
        # 手検出関連
        self.hands_2d: Optional[MediaPipeHandsWrapper] = None
        self.projector_3d: Optional[Hand3DProjector] = None
        self.tracker: Optional[Hand3DTracker] = None
        self.hand_markers = []  # 3D手マーカー
        
        # 現在の手検出結果（3Dマーカー用）
        self.current_hands_3d = []
        self.current_tracked_hands = []
        
        # Open3D関連
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        
        # 状態管理
        self.is_running = False
        self.frame_count = 0
        self.first_frame = True
        
        # パフォーマンス計測
        self.performance_stats = {
            'fps': 0.0,
            'frame_time': 0.0,
            'filter_time': 0.0,
            'pointcloud_time': 0.0,
            'hand_detection_time': 0.0,
            'hand_projection_time': 0.0,
            'hand_tracking_time': 0.0,
            'display_time': 0.0,
            'hands_detected': 0,
            'hands_tracked': 0
        }
        self.last_stats_time = time.perf_counter()
        
        # 非同期カラー画像デコード用スレッドプール
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._color_decode_future = None
        
        # --- RGB パイプライン高速化パラメータ ---
        # カラーフレームのデコードは毎フレーム不要。直前の画像を再利用することで
        # CPU スパイクを抑制し FPS を安定させる。
        self._color_decode_interval = 2  # decode every N frames
        self._next_color_decode_frame = 0
        self._last_color_bgr = np.zeros((self.rgb_window_size[1], self.rgb_window_size[0], 3), dtype=np.uint8)

        # -------- Depth filter interval control (GAP-05 fix) --------
        # Depth bilateral filtering is heavy; run once every N frames and
        # reuse the cached result to eliminate redundant computation.
        self._filter_interval = 2  # apply filter every 2 RGB frames
        self._next_filter_frame = 0
        self._depth_filtered_cache = None  # type: Optional[np.ndarray]
        
        # --- Hand detection timing ---
        self._hand_detection_interval = 2  # run detection every N RGB frames
        self._next_hand_detection_frame = 0
        self._last_detected_hands_2d: List = []
        self._last_detected_hands_3d: List = []
        self._last_tracked_hands: List = []
        
        # ------------- Display buffer pre-allocation -------------------
        # To avoid per-frame allocation from np.hstack, allocate a single
        # contiguous buffer that we copy depth & color images into.
        disp_h, disp_w = self.rgb_window_size[1], self.rgb_window_size[0] * 2
        self._combined_buffer = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

        # ---- Hand-marker update throttling -----------------------------
        # Open3D に毎フレーム add/remove_geometry すると内部メッセージ
        # キューが飽和し UI が固まる。フレーム N おきにまとめて更新する。
        self._marker_update_interval = 10   # 更新間隔 (frames)
        self._next_marker_update = 0        # 次回更新フレーム番号
        
    def set_camera(self, camera):
        """外部からカメラを設定（OAK-D対応）"""
        self.camera = camera
    
    def initialize(self) -> bool:
        """
        ビューワーを初期化
        
        Returns:
            成功した場合True
        """
        try:
            # 既に外部からカメラが注入されている場合は再利用する
            if self.camera is None:
                # カメラをまだ持っていない場合のみ新規生成（OrbbecCameraをデフォルトとして使用）
                self.camera = OrbbecCamera(enable_color=True)

            # カメラが未初期化・未スタートならここで起動する
            if not getattr(self.camera, "is_started", False):
                if not self.camera.initialize():
                    print("Failed to initialize camera")
                    return False

                if not self.camera.start():
                    print("Failed to start camera")
                    return False
            
            # 点群コンバーター初期化
            if self.camera.depth_intrinsics:
                self.pointcloud_converter = PointCloudConverter(self.camera.depth_intrinsics)
            else:
                print("No depth intrinsics available")
                return False
            
            # 深度フィルタ初期化
            if self.enable_filter:
                self.depth_filter = DepthFilter(
                    filter_types=[FilterType.COMBINED],
                    temporal_alpha=0.3,
                    bilateral_sigma_color=50.0
                )
            
            # 手検出コンポーネント初期化
            if self.enable_hand_detection:
                if not self._initialize_hand_detection():
                    print("Warning: Hand detection initialization failed, continuing without hand detection")
                    self.enable_hand_detection = False
            
            # Open3Dビューワー初期化
            if not self._initialize_3d_viewer():
                return False
            
            print("Dual viewer initialized successfully")
            print(f"Hand detection: {'Enabled' if self.enable_hand_detection else 'Disabled'}")
            if self.enable_hand_detection:
                print(f"  - MediaPipe: {'GPU' if self.use_gpu_mediapipe else 'CPU'} mode")
                print(f"  - Tracking: {'Enabled' if self.enable_tracking else 'Disabled'}")
                print(f"  - Min confidence: {self.min_detection_confidence}")
            return True
            
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def _initialize_hand_detection(self) -> bool:
        """手検出コンポーネントを初期化"""
        try:
            # 2D手検出初期化
            print(f"Initializing MediaPipe Hands ({'GPU' if self.use_gpu_mediapipe else 'CPU'})...")
            self.hands_2d = MediaPipeHandsWrapper(
                use_gpu=self.use_gpu_mediapipe,
                max_num_hands=2,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5
            )
            
            # 3D投影初期化
            if self.camera.depth_intrinsics:
                print("Initializing 3D projector...")
                self.projector_3d = Hand3DProjector(
                    camera_intrinsics=self.camera.depth_intrinsics,
                    interpolation_method=DepthInterpolationMethod.NEAREST,
                    min_confidence_3d=0.3
                )
            else:
                print("No depth intrinsics available for 3D projection")
                return False
            
            # トラッカー初期化
            if self.enable_tracking:
                print("Initializing hand tracker...")
                self.tracker = Hand3DTracker(
                    max_lost_frames=10,
                    min_track_length=5,
                    max_assignment_distance=0.3
                )
            
            return True
            
        except Exception as e:
            print(f"Hand detection initialization error: {e}")
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
            print("Failed to initialize dual viewer")
            return
        
        self.is_running = True
        print("\nDual Viewer Started!")
        print("\nControls:")
        print("RGB Window:")
        print("- Q/ESC: Quit")
        print("- F: Toggle filter")
        print("- R: Reset filter history")
        if self.enable_hand_detection:
            print("- H: Toggle hand detection")
            if self.enable_tracking:
                print("- T: Toggle tracking")
                print("- Y: Reset tracker")
        print("\n3D Viewer:")
        print("- Mouse: Rotate/Pan/Zoom")
        print("- R: Reset view")
        print("- +/-: Change point size")
        print("- S: Save current point cloud")
        print()
        
        try:
            while self.is_running:
                if not self._process_frame():
                    break
                    
                # パフォーマンス統計更新
                self._update_performance_stats()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _process_frame(self) -> bool:
        """
        1フレーム処理
        
        Returns:
            継続する場合True
        """
        frame_start_time = time.perf_counter()
        
        # フレーム取得
        frame_data = self.camera.get_frame(timeout_ms=100)
        if frame_data is None or frame_data.depth_frame is None:
            return True
        
        # RGB表示処理
        if not self._process_rgb_display(frame_data):
            return False
        
        # 点群表示処理（間隔制御）
        if self.frame_count % self.update_interval == 0:
            if not self._process_pointcloud_display(frame_data):
                return False
        
        self.frame_count += 1
        self.performance_stats['frame_time'] = (time.perf_counter() - frame_start_time) * 1000
        
        return True
    
    def _process_rgb_display(self, frame_data: FrameData) -> bool:
        """
        RGB表示処理
        
        Args:
            frame_data: フレームデータ
            
        Returns:
            継続する場合True
        """
        try:
            # ------------------------------------------------------------------
            # 1) 深度画像を抽出 & 必要ならフィルタを適用
            # ------------------------------------------------------------------
            depth_image = self._extract_depth_image(frame_data)

            # 深度画像が取得できなければスキップ (たとえば一時的にフレーム無し)
            if depth_image is None:
                return True

            # Depth filtering (heavy) – apply only every N frames
            if self.depth_filter is not None and self.enable_filter:
                if self.frame_count >= self._next_filter_frame or self._depth_filtered_cache is None:
                    depth_image = self.depth_filter.apply_filter(depth_image)
                    # Schedule next filtering
                    self._next_filter_frame = self.frame_count + self._filter_interval
                    self._depth_filtered_cache = depth_image
                else:
                    # Reuse cached filtered image for interim frames
                    depth_image = self._depth_filtered_cache

            # キャッシュして PointCloud パスで再利用 (不要なリシェイプを排除)
            self._last_depth_image_filtered = depth_image  # type: ignore[attr-defined]

            # 疑似カラー化
            depth_colored = self._create_depth_visualization(depth_image)
            
            # --- Decide whether to run hand detection this frame ---
            detect_now = (
                self.enable_hand_detection and self.hands_2d is not None and
                self.frame_count >= self._next_hand_detection_frame
            )

            if detect_now and self._last_color_bgr is not None and self._last_color_bgr.size > 0:
                hands_2d, hands_3d, tracked_hands = self._process_hand_detection(
                    self._last_color_bgr,
                    depth_image,
                )

                # Save for reuse and visualization
                self._last_detected_hands_2d = hands_2d
                self._last_detected_hands_3d = hands_3d
                self._last_tracked_hands = tracked_hands

                # Schedule next detection
                self._next_hand_detection_frame = self.frame_count + self._hand_detection_interval
            else:
                # Reuse previous detection results
                hands_2d = self._last_detected_hands_2d
                hands_3d = self._last_detected_hands_3d
                tracked_hands = self._last_tracked_hands
                
            # 手検出が無効化されている場合はキャッシュをクリア
            if not self.enable_hand_detection:
                hands_2d = []
                hands_3d = []
                tracked_hands = []
                self._last_detected_hands_2d = []
                self._last_detected_hands_3d = []
                self._last_tracked_hands = []

            # 3Dマーカー更新用に保存
            self.current_hands_3d = hands_3d
            self.current_tracked_hands = tracked_hands
            
            # カラー画像があれば表示
            display_images = []
            
            # 深度画像（疑似カラー）
            depth_resized = cv2.resize(depth_colored, self.rgb_window_size)
            cv2.putText(depth_resized, f"Depth (Frame: {self.frame_count})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_images.append(depth_resized)
            
            # RGB画像処理（手検出結果はキャッシュを使用）
            # --- 非同期デコード ---
            if self._color_decode_future and self._color_decode_future.done():
                try:
                    self._last_color_bgr = self._color_decode_future.result()
                except Exception:
                    pass  # keep previous image on error

            # スケジュール次カラーのデコード（一定間隔＆前ジョブが完了している場合のみ）
            if (
                frame_data.color_frame is not None
                and self.frame_count >= self._next_color_decode_frame
                and (self._color_decode_future is None or self._color_decode_future.done())
            ):
                self._next_color_decode_frame = self.frame_count + self._color_decode_interval
                self._color_decode_future = self._executor.submit(
                    self._decode_color_frame_resized, frame_data.color_frame
                )

            color_bgr = self._last_color_bgr
            if color_bgr is None or color_bgr.size == 0:
                # まだ画像が無い場合のみ同期デコード（最初の1回）
                color_bgr = self._decode_color_frame_resized(frame_data.color_frame)
                self._last_color_bgr = color_bgr

            # 手検出結果を描画したRGB画像を生成
            if color_bgr is not None and color_bgr.size > 0:
                color_with_hands = self._draw_hand_detections_on_image(
                    color_bgr.copy(), hands_2d, hands_3d, tracked_hands
                )
                display_images.append(color_with_hands)
            else:
                display_images.append(color_bgr)
            
            # ----- Efficient image stacking using preallocated buffer -----
            if len(display_images) == 2:
                w = self.rgb_window_size[0]
                self._combined_buffer[:, :w, :] = display_images[0]
                self._combined_buffer[:, w:, :] = display_images[1]
                combined_image = self._combined_buffer
            else:
                combined_image = display_images[0]
            
            # パフォーマンス情報をオーバーレイ
            self._draw_performance_overlay(combined_image)
            
            cv2.imshow("Geocussion-SP Input Viewer", combined_image)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                return False
            elif key == ord('f'):  # Toggle filter
                self.enable_filter = not self.enable_filter
                print(f"Depth filter: {'Enabled' if self.enable_filter else 'Disabled'}")
            elif key == ord('r') and self.depth_filter is not None:  # Reset filter
                self.depth_filter.reset_temporal_history()
                print("Filter history reset")
            elif key == ord('h'):  # Toggle hand detection
                self.enable_hand_detection = not self.enable_hand_detection
                print(f"Hand detection: {'Enabled' if self.enable_hand_detection else 'Disabled'}")
            elif key == ord('t') and self.enable_hand_detection:  # Toggle tracking
                self.enable_tracking = not self.enable_tracking
                print(f"Hand tracking: {'Enabled' if self.enable_tracking else 'Disabled'}")
            elif key == ord('y') and self.tracker is not None:  # Reset tracker
                self.tracker.reset()
                print("Hand tracker reset")
            
            return True
            
        except Exception as e:
            print(f"RGB display error: {e}")
            return True
    
    def _process_pointcloud_display(self, frame_data: FrameData) -> bool:
        """
        点群表示処理
        
        Args:
            frame_data: フレームデータ
            
        Returns:
            継続する場合True
        """
        try:
            pointcloud_start_time = time.perf_counter()
            
            # 深度画像 – 先に RGB パスでフィルタ済みのものを再利用
            if hasattr(self, "_last_depth_image_filtered") and self._last_depth_image_filtered is not None:
                depth_image = self._last_depth_image_filtered
            else:
                # --- Robust depth extraction (handles missing intrinsics) ---
                depth_image = self._extract_depth_image(frame_data)

                # Guard against extraction failure
                if depth_image is None:
                    # Skip this frame gracefully
                    return True

                # フィルタ適用（必要な場合）
                if self.depth_filter is not None:
                    depth_image = self.depth_filter.apply_filter(depth_image)
            
            # ---------------------------------------------------------------
            # ❶ カラー点群統一 : color_frame が欠落しているフレームでは
            #    直前に取得したカラー画像を使うことで深度カラー ↔︎ RGB の
            #    フリッカを防止する。
            # ---------------------------------------------------------------
            if hasattr(frame_data, "color_frame") and frame_data.color_frame is not None:
                color_src = frame_data.color_frame
                # 最新カラーをキャッシュ
                self._last_color_frame_for_pcd = color_src  # type: ignore[attr-defined]
            else:
                color_src = getattr(self, "_last_color_frame_for_pcd", None)

            # 点群生成（color_src が None の場合は colors=None → 単色表示）
            points, colors = self.pointcloud_converter.depth_to_pointcloud(
                frame_data.depth_frame,
                color_src,
            )
            
            self.performance_stats['pointcloud_time'] = (time.perf_counter() - pointcloud_start_time) * 1000
            
            if len(points) > 0:
                display_start_time = time.perf_counter()
                
                # Open3D点群を更新
                self.pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # 手マーカーを更新
                self._update_hand_markers()
                
                # 初回は視点をリセット
                if self.first_frame:
                    self.vis.reset_view_point(True)
                    self.first_frame = False
                
                self.performance_stats['display_time'] = (time.perf_counter() - display_start_time) * 1000
                
                hands_info = f" | Hands: {self.performance_stats['hands_detected']}D/{self.performance_stats['hands_tracked']}T" if self.enable_hand_detection else ""
                print(f"\rFrame {self.frame_count}: {len(points)} points | "
                      f"FPS: {self.performance_stats['fps']:.1f} | "
                      f"Filter: {self.performance_stats['filter_time']:.1f}ms{hands_info}", end="")
            
            # ビジュアライザーを更新
            self.vis.update_geometry(self.pcd)
            if not self.vis.poll_events():
                return False
            self.vis.update_renderer()
            
            return True
            
        except Exception as e:
            print(f"Point cloud display error: {e}")
            return True
    
    def _draw_performance_overlay(self, image: np.ndarray) -> None:
        """パフォーマンス情報をオーバーレイ表示"""
        base_texts = [
            f"FPS: {self.performance_stats['fps']:.1f}",
            f"Frame: {self.performance_stats['frame_time']:.1f}ms",
            f"Filter: {self.performance_stats['filter_time']:.1f}ms",
            f"Points: {self.performance_stats['pointcloud_time']:.1f}ms"
        ]
        
        # 手検出関連の統計を追加
        if self.enable_hand_detection:
            hand_texts = [
                f"Hand2D: {self.performance_stats['hand_detection_time']:.1f}ms",
                f"Hand3D: {self.performance_stats['hand_projection_time']:.1f}ms",
                f"Track: {self.performance_stats['hand_tracking_time']:.1f}ms",
                f"Detected: {self.performance_stats['hands_detected']} | Tracked: {self.performance_stats['hands_tracked']}"
            ]
            texts = base_texts + hand_texts
            overlay_y = image.shape[0] - 220
        else:
            texts = base_texts
            overlay_y = image.shape[0] - 120
        
        for i, text in enumerate(texts):
            color = (0, 255, 0) if i < 4 else (0, 255, 255)  # 手検出統計は水色
            cv2.putText(image, text, (10, overlay_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _update_performance_stats(self) -> None:
        """パフォーマンス統計を更新"""
        current_time = time.perf_counter()
        if current_time - self.last_stats_time >= 1.0:  # 1秒ごとに更新
            self.performance_stats['fps'] = self.frame_count / (current_time - self.last_stats_time)
            self.frame_count = 0
            self.last_stats_time = current_time
    
    def _cleanup(self) -> None:
        """クリーンアップ処理"""
        print("\nCleaning up...")
        
        self.is_running = False
        
        # 手検出コンポーネント終了
        if self.hands_2d is not None:
            self.hands_2d.close()
        
        if self.camera is not None:
            self.camera.stop()
        
        if self.vis is not None:
            self.vis.destroy_window()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")
        
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
    
    def _process_hand_detection(self, color_bgr: np.ndarray, depth_image: np.ndarray) -> Tuple[list, list, list]:
        """
        手検出処理
        
        Args:
            color_bgr: カラー画像（BGR形式）
            depth_image: 深度画像
            
        Returns:
            (hands_2d, hands_3d, tracked_hands)のタプル
        """
        hands_2d, hands_3d, tracked_hands = [], [], []
        
        try:
            if color_bgr is None or color_bgr.size == 0:
                return hands_2d, hands_3d, tracked_hands

            bgr_image = color_bgr  # Already BGR resized image
            
            # 2D手検出
            detection_start = time.perf_counter()
            hands_2d = self.hands_2d.detect_hands(bgr_image)
            hands_2d = filter_hands_by_confidence(hands_2d, min_confidence=self.min_detection_confidence)
            self.performance_stats['hand_detection_time'] = (time.perf_counter() - detection_start) * 1000
            self.performance_stats['hands_detected'] = len(hands_2d)
            
            # 3D投影
            if hands_2d and self.projector_3d is not None:
                projection_start = time.perf_counter()
                hands_3d = self.projector_3d.project_hands_batch(hands_2d, depth_image)
                self.performance_stats['hand_projection_time'] = (time.perf_counter() - projection_start) * 1000
            else:
                self.performance_stats['hand_projection_time'] = 0.0
            
            # トラッキング
            if hands_3d and self.enable_tracking and self.tracker is not None:
                tracking_start = time.perf_counter()
                tracked_hands = self.tracker.update(hands_3d)
                tracked_hands = filter_stable_hands(tracked_hands, min_confidence=0.7)
                self.performance_stats['hand_tracking_time'] = (time.perf_counter() - tracking_start) * 1000
                self.performance_stats['hands_tracked'] = len(tracked_hands)
            else:
                self.performance_stats['hand_tracking_time'] = 0.0
                self.performance_stats['hands_tracked'] = 0
                
        except Exception as e:
            print(f"\nHand detection error: {e}")
            self.performance_stats['hand_detection_time'] = 0.0
            self.performance_stats['hand_projection_time'] = 0.0
            self.performance_stats['hand_tracking_time'] = 0.0
            self.performance_stats['hands_detected'] = 0
            self.performance_stats['hands_tracked'] = 0
        
        return hands_2d, hands_3d, tracked_hands
    
    def _draw_hand_detections_on_image(self, image: np.ndarray, hands_2d: list, hands_3d: list, tracked_hands: list) -> np.ndarray:
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
            bbox = hand_2d.bounding_box
            
            # バウンディングボックス描画
            cv2.rectangle(image, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                         (0, 255, 0), 2)  # 緑色の枠
            
            # 手の情報表示
            cv2.putText(image, f"{hand_2d.handedness.value} ({hand_2d.confidence:.2f})",
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
                               f"  ID: {tracked_hand.id[:8]} Speed: {speed:.2f}m/s",
                               (10, track_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except:
                    cv2.putText(image, 
                               f"  ID: {getattr(tracked_hand, 'id', 'Unknown')[:8]}",
                               (10, track_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return image

    def _process_color_image(self, frame_data: FrameData, hands_2d: list, hands_3d: list, tracked_hands: list) -> np.ndarray:
        """
        RGB画像に手検出結果を描画
        
        Args:
            frame_data: フレームデータ
            hands_2d: 2D手検出結果
            hands_3d: 3D手検出結果
            tracked_hands: トラッキング結果
            
        Returns:
            描画済み画像
        """
        # ------------------------------
        # カラーフレームをNumPy BGR画像へ変換（OAK-D / Orbbec対応）
        # ------------------------------
        try:
            # OAK-D の OakFrameWrapper かどうかを判定
            if hasattr(frame_data.color_frame, 'frame_data') and isinstance(frame_data.color_frame.frame_data, np.ndarray):
                # OAK-D の場合: frame_data は既に BGR の NumPy 配列
                bgr_image = frame_data.color_frame.frame_data.copy()
            else:
                # Orbbec SDK の場合: 従来の処理
                color_data = np.frombuffer(frame_data.color_frame.get_data(), dtype=np.uint8)
                color_format = frame_data.color_frame.get_format()

                bgr_image: Optional[np.ndarray] = None

                try:
                    from pyorbbecsdk import OBFormat as _OBF
                except ImportError:
                    from src.data_types import OBFormat as _OBF  # type: ignore

                if color_format == _OBF.RGB:  # type: ignore[attr-defined]
                    rgb_image = color_data.reshape((self.camera.depth_intrinsics.height, self.camera.depth_intrinsics.width, 3))
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                elif color_format == _OBF.BGR:  # type: ignore[attr-defined]
                    bgr_image = color_data.reshape((self.camera.depth_intrinsics.height, self.camera.depth_intrinsics.width, 3))
                elif color_format == _OBF.MJPG:  # type: ignore[attr-defined]
                    decoded = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        bgr_image = decoded

                if bgr_image is None:
                    # Fallback: create black image
                    h = self.camera.depth_intrinsics.height if self.camera.depth_intrinsics else 480
                    w = self.camera.depth_intrinsics.width if self.camera.depth_intrinsics else 640
                    bgr_image = np.zeros((h, w, 3), dtype=np.uint8)

            height, width = bgr_image.shape[:2]

            # 2D手検出結果描画
            for hand_2d in hands_2d:
                bbox = hand_2d.bounding_box
                scale_x = width / self.camera.depth_intrinsics.width
                scale_y = height / self.camera.depth_intrinsics.height

                bbox_scaled = (
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y)
                )

                cv2.rectangle(bgr_image, (bbox_scaled[0], bbox_scaled[1]),
                              (bbox_scaled[0] + bbox_scaled[2], bbox_scaled[1] + bbox_scaled[3]),
                              (0, 255, 255), 2)

                cv2.putText(bgr_image, f"{hand_2d.handedness.value} ({hand_2d.confidence:.2f})",
                            (bbox_scaled[0], bbox_scaled[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                for i, landmark in enumerate(hand_2d.landmarks):
                    if i in [0, 4, 8, 12, 16, 20]:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(bgr_image, (x, y), 3, (0, 255, 0), -1)

            # 3D投影結果情報表示
            info_y = 60
            if hands_3d:
                cv2.putText(bgr_image, f"3D Hands: {len(hands_3d)}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                for i, hand_3d in enumerate(hands_3d):
                    palm_x, palm_y, palm_z = hand_3d.palm_center_3d
                    cv2.putText(bgr_image, f"  Hand {i+1}: ({palm_x:.2f}, {palm_y:.2f}, {palm_z:.2f}m)",
                                (10, info_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # トラッキング結果表示
            if tracked_hands:
                track_y = info_y + 25 * (len(hands_3d) + 2) if hands_3d else info_y + 25
                cv2.putText(bgr_image, f"Tracked: {len(tracked_hands)}", (10, track_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for i, tracked_hand in enumerate(tracked_hands):
                    speed = tracked_hand.speed
                    cv2.putText(bgr_image, f"  ID: {tracked_hand.id[:8]} Speed: {speed:.2f}m/s",
                                (10, track_y + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            bgr_resized = cv2.resize(bgr_image, self.rgb_window_size)
            return bgr_resized
        except Exception as e:
            # In case of any decoding error, return a black placeholder to keep pipeline running
            placeholder = np.zeros((self.rgb_window_size[1], self.rgb_window_size[0], 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Color frame error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            placeholder_resized = cv2.resize(placeholder, self.rgb_window_size)
            return placeholder_resized
    
    def _update_hand_markers(self) -> None:
        """3D点群ビューワーの手マーカーを更新"""
        try:
            # ---------------- Throttle updates -------------------
            if self.frame_count < getattr(self, "_next_marker_update", 0):
                return

            # 次回実行フレームを設定
            self._next_marker_update = self.frame_count + getattr(self, "_marker_update_interval", 10)

            # 既存マーカーを一括削除（間引きにより頻度は低い）
            for marker in self.hand_markers:
                self.vis.remove_geometry(marker, reset_bounding_box=False)
            self.hand_markers.clear()
            
            # 3D手検出結果のマーカー（黄色球体）
            for i, hand_3d in enumerate(self.current_hands_3d):
                palm_center = hand_3d.palm_center_3d
                if all(abs(coord) > 0.001 for coord in palm_center):  # 有効な座標のみ
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                    marker.translate(palm_center)
                    marker.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色: 3D検出
                    self.hand_markers.append(marker)
                    self.vis.add_geometry(marker, reset_bounding_box=False)
            
            # トラッキング結果のマーカー（状態に応じた色）
            for tracked_hand in self.current_tracked_hands:
                pos = tracked_hand.position
                if len(pos) == 3 and all(abs(coord) > 0.001 for coord in pos):
                    # 手の位置マーカー
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                    marker.translate(pos)
                    
                    # 状態に応じた色分け
                    if tracked_hand.state == TrackingState.TRACKING:
                        marker.paint_uniform_color([0.0, 1.0, 0.0])  # 緑: トラッキング中
                    elif tracked_hand.state == TrackingState.INITIALIZING:
                        marker.paint_uniform_color([0.0, 1.0, 1.0])  # シアン: 初期化中
                    else:
                        marker.paint_uniform_color([1.0, 0.5, 0.0])  # オレンジ: その他
                    
                    self.hand_markers.append(marker)
                    self.vis.add_geometry(marker, reset_bounding_box=False)
                    
                    # 速度ベクトル表示（シンプル版）
                    if tracked_hand.speed > 0.02:  # 2cm/s以上で動いている場合
                        try:
                            # 速度方向に小さな円錐を配置
                            cone = o3d.geometry.TriangleMesh.create_cone(radius=0.01, height=0.05)
                            # 速度方向への移動
                            vel_pos = pos + tracked_hand.velocity * 0.2
                            cone.translate(vel_pos)
                            cone.paint_uniform_color([1.0, 0.0, 1.0])  # マゼンタ: 速度ベクトル
                            self.hand_markers.append(cone)
                            self.vis.add_geometry(cone, reset_bounding_box=False)
                        except:
                            pass
            
            # 描画キューを確実にフラッシュ
            try:
                self.vis.update_renderer()
            except Exception:
                pass

        except Exception:
            # Marker update should never interrupt pipeline
            pass

    def _create_depth_visualization(self, depth_image: np.ndarray) -> np.ndarray:
        """深度画像の可視化を高速に疑似カラー化（uint16→BGR）"""
        # OpenCV は8bit入力を期待するためリスケール
        if depth_image.dtype != np.uint8:
            # 固定スケール (0-4 m → 0-255) で十分。動的計算はコスト高
            depth_8u = cv2.convertScaleAbs(depth_image, alpha=255.0 / 4000.0)
        else:
            depth_8u = depth_image

        return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # Depth frame helpers
    # ------------------------------------------------------------------

    def _extract_depth_image(self, frame_data: FrameData):
        """FrameData から NumPy uint16 深度画像を抽出 (カメラ intrinsics 依存)"""
        try:
            if frame_data.depth_frame is None or self.camera is None or self.camera.depth_intrinsics is None:
                return None

            intr = self.camera.depth_intrinsics
            depth_data = np.frombuffer(frame_data.depth_frame.get_data(), dtype=np.uint16)
            return depth_data.reshape((intr.height, intr.width))
        except Exception:
            # 何らかの理由で取得できなければ None を返して後段でスキップ
            return None

    # ------------------------------------------------------------------
    # Color frame helpers
    # ------------------------------------------------------------------

    def _decode_color_frame_resized(self, color_frame) -> np.ndarray:
        """カラーFrame (Orbbec SDK / OAK-D) → BGR, リサイズ済み"""
        try:
            import numpy as _np
            
            # OAK-D の OakFrameWrapper かどうかを判定
            if hasattr(color_frame, 'frame_data') and isinstance(color_frame.frame_data, _np.ndarray):
                # OAK-D の場合: frame_data は既に BGR の NumPy 配列
                bgr_img = color_frame.frame_data
                if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
                    # 既に適切なサイズの場合はそのまま返す、そうでなければリサイズ
                    if bgr_img.shape[:2] == (self.rgb_window_size[1], self.rgb_window_size[0]):
                        return bgr_img
                    else:
                        return cv2.resize(bgr_img, self.rgb_window_size)
                else:
                    # フォーマットが期待と異なる場合は黒画像を返す
                    return _np.zeros((self.rgb_window_size[1], self.rgb_window_size[0], 3), dtype=_np.uint8)
            
            # Orbbec SDK の場合: 既存の処理を維持
            color_data = _np.frombuffer(color_frame.get_data(), dtype=_np.uint8)

            try:
                from pyorbbecsdk import OBFormat as _OBF
            except ImportError:
                from src.data_types import OBFormat as _OBF  # type: ignore

            fmt = color_frame.get_format()
            h_src = self.camera.depth_intrinsics.height if self.camera and self.camera.depth_intrinsics else 720
            w_src = self.camera.depth_intrinsics.width if self.camera and self.camera.depth_intrinsics else 1280

            if fmt == _OBF.RGB:  # type: ignore[attr-defined]
                rgb_img = color_data.reshape((h_src, w_src, 3))
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            elif fmt == _OBF.BGR:  # type: ignore[attr-defined]
                bgr_img = color_data.reshape((h_src, w_src, 3))
            else:  # MJPG 等
                decoded = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                bgr_img = decoded if decoded is not None else _np.zeros((h_src, w_src, 3), dtype=_np.uint8)

            return cv2.resize(bgr_img, self.rgb_window_size)
        except Exception as e:
            # decoding error – return black image
            print(f"Color frame decoding error: {e}")
            return np.zeros((self.rgb_window_size[1], self.rgb_window_size[0], 3), dtype=np.uint8)


def main():
    """メイン関数"""
    viewer = DualViewer(
        enable_filter=True,
        enable_hand_detection=True,
        enable_tracking=True,
        update_interval=3,  # 3フレームごとに点群更新
        point_size=2.0,
        rgb_window_size=(640, 480),
        min_detection_confidence=0.7,
        use_gpu_mediapipe=False
    )
    
    viewer.run()


if __name__ == "__main__":
    main() 