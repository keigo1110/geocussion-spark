## 体験向上のための衝突検知システム改善提案

### 1. **高速移動対応の改善点**

**現状の課題**: 高速な手の動きで衝突を見逃す可能性

```python
# src/collision/detector.py の改善案
class ImprovedCollisionDetector:
    def __init__(self, ccd_enabled=True, substep_count=5):
        self.ccd_enabled = ccd_enabled
        self.substep_count = substep_count
        self.previous_positions = {}  # 手のIDごとの前フレーム位置
        
    def detect_collision_continuous(self, hand_id, current_pos, previous_pos, mesh):
        """連続衝突検知（CCD）の実装"""
        if not self.ccd_enabled:
            return self.detect_collision_discrete(current_pos, mesh)
            
        # 移動ベクトルを計算
        movement = current_pos - previous_pos
        distance = np.linalg.norm(movement)
        
        # 高速移動時は補間ステップを増やす
        adaptive_substeps = max(self.substep_count, 
                               int(distance / self.collision_radius))
        
        # 軌跡に沿って複数点で衝突チェック
        for i in range(adaptive_substeps + 1):
            t = i / adaptive_substeps
            interpolated_pos = previous_pos + t * movement
            
            collision = self.check_sphere_mesh_collision(
                interpolated_pos, self.collision_radius, mesh
            )
            
            if collision:
                # 正確な衝突点と時刻を返す
                collision['interpolation_t'] = t
                collision['impact_velocity'] = movement / self.frame_time
                return collision
                
        return None
```

### 2. **予測的衝突検知の実装**

```python
# src/detection/predictor.py (新規ファイル)
class HandMotionPredictor:
    def __init__(self, history_size=5, prediction_frames=3):
        self.history_size = history_size
        self.prediction_frames = prediction_frames
        self.velocity_history = deque(maxlen=history_size)
        self.acceleration_history = deque(maxlen=history_size-1)
        
    def predict_collision_trajectory(self, current_pos, current_velocity, mesh):
        """速度と加速度から未来の軌跡を予測"""
        # 加速度を計算
        if len(self.velocity_history) > 0:
            acceleration = (current_velocity - self.velocity_history[-1]) / self.dt
            self.acceleration_history.append(acceleration)
        
        # 平均加速度を使用して予測
        avg_acceleration = np.mean(self.acceleration_history, axis=0) if self.acceleration_history else np.zeros(3)
        
        predicted_collisions = []
        pos = current_pos.copy()
        vel = current_velocity.copy()
        
        for frame in range(self.prediction_frames):
            # 物理シミュレーション
            pos += vel * self.dt + 0.5 * avg_acceleration * self.dt**2
            vel += avg_acceleration * self.dt
            
            # 衝突チェック
            collision = self.check_collision(pos, mesh)
            if collision:
                collision['predicted_frame'] = frame
                collision['confidence'] = 1.0 - (frame / self.prediction_frames)
                predicted_collisions.append(collision)
                
        return predicted_collisions
```

### 3. **適応的サンプリングレートの実装**

```python
# src/input/adaptive_sampler.py (新規ファイル)
class AdaptiveSampler:
    def __init__(self, base_fps=30, max_fps=120):
        self.base_fps = base_fps
        self.max_fps = max_fps
        self.current_fps = base_fps
        self.velocity_threshold = 0.3  # m/s
        self.acceleration_threshold = 2.0  # m/s²
        
    def update_sampling_rate(self, hand_velocities):
        """手の速度に基づいてサンプリングレートを調整"""
        if not hand_velocities:
            self.current_fps = self.base_fps
            return
            
        max_velocity = max(np.linalg.norm(v) for v in hand_velocities.values())
        
        # 速度に基づいてFPSを調整
        if max_velocity > self.velocity_threshold * 2:
            target_fps = self.max_fps
        elif max_velocity > self.velocity_threshold:
            # 線形補間
            ratio = (max_velocity - self.velocity_threshold) / self.velocity_threshold
            target_fps = self.base_fps + ratio * (self.max_fps - self.base_fps)
        else:
            target_fps = self.base_fps
            
        # スムーズな遷移
        self.current_fps = 0.8 * self.current_fps + 0.2 * target_fps
        
        return int(self.current_fps)
```

### 4. **衝突検知の精度向上**

```python
# src/collision/detector.py の改善
class PreciseCollisionDetector:
    def __init__(self):
        self.collision_history = deque(maxlen=10)
        self.debounce_threshold = 0.05  # 50ms
        
    def detect_collision_with_validation(self, hand_pos, mesh):
        """ノイズ除去と検証を含む衝突検知"""
        # 基本的な衝突検知
        raw_collision = self.detect_collision_raw(hand_pos, mesh)
        
        if not raw_collision:
            return None
            
        # 衝突の妥当性検証
        current_time = time.time()
        
        # デバウンス処理（連続した衝突を1つにまとめる）
        if self.collision_history:
            last_collision = self.collision_history[-1]
            time_diff = current_time - last_collision['timestamp']
            distance_diff = np.linalg.norm(
                raw_collision['position'] - last_collision['position']
            )
            
            if time_diff < self.debounce_threshold and distance_diff < 0.02:
                return None  # 重複衝突として無視
                
        # 衝突の詳細情報を計算
        collision_info = {
            'position': raw_collision['position'],
            'normal': raw_collision['normal'],
            'depth': raw_collision['depth'],
            'timestamp': current_time,
            'triangle_id': raw_collision['triangle_id'],
            'impact_angle': self.calculate_impact_angle(hand_pos, raw_collision['normal']),
            'surface_properties': self.get_surface_properties(raw_collision['triangle_id'])
        }
        
        self.collision_history.append(collision_info)
        return collision_info
```

### 5. **空間分割とBVH最適化**

```python
# src/collision/bvh.py の改善
class OptimizedBVH:
    def __init__(self, leaf_size=10, dynamic_update=True):
        self.leaf_size = leaf_size
        self.dynamic_update = dynamic_update
        self.node_cache = {}
        
    def build_dynamic_bvh(self, mesh, hand_positions):
        """手の位置に基づいて動的にBVHを最適化"""
        # 手の周辺の領域を高解像度に
        hand_regions = self.identify_hand_regions(hand_positions)
        
        # 適応的な分割
        root = self.build_adaptive_node(
            mesh.triangles, 
            depth=0, 
            hand_regions=hand_regions
        )
        
        return root
        
    def query_nearest_triangles(self, position, radius, max_triangles=50):
        """高速な近傍三角形クエリ"""
        # キャッシュチェック
        cache_key = (tuple(position), radius)
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]
            
        # BVHトラバーサル with early termination
        candidates = []
        self._traverse_bvh(self.root, position, radius, candidates, max_triangles)
        
        # 結果をキャッシュ
        self.node_cache[cache_key] = candidates
        
        return candidates
```

### 6. **音生成のための衝突情報拡張**

```python
# src/collision/audio_params.py (新規ファイル)
class CollisionAudioAnalyzer:
    def analyze_collision_for_audio(self, collision_event, hand_tracker):
        """音生成に必要な詳細パラメータを抽出"""
        hand_id = collision_event['hand_id']
        
        # 衝突の強さを計算
        impact_velocity = collision_event['impact_velocity']
        impact_force = np.dot(impact_velocity, collision_event['normal'])
        
        # 接触面の特性
        surface_material = collision_event['surface_properties']['material']
        surface_tension = collision_event['surface_properties']['tension']
        
        # ジェスチャーの種類を判定
        gesture_type = self.classify_gesture(
            hand_tracker.get_hand_history(hand_id)
        )
        
        # 音響パラメータを生成
        audio_params = {
            'pitch': self.map_position_to_pitch(collision_event['position']),
            'velocity': self.map_force_to_velocity(abs(impact_force)),
            'timbre': self.get_timbre_from_gesture(gesture_type),
            'duration': self.estimate_contact_duration(collision_event),
            'reverb': self.calculate_reverb_from_surface(surface_material),
            'modulation': {
                'vibrato': self.get_vibrato_from_pressure(collision_event['depth']),
                'tremolo': self.get_tremolo_from_movement(impact_velocity)
            }
        }
        
        return audio_params
```

### 7. **パフォーマンス最適化**

```python
# src/collision/parallel_detector.py (新規ファイル)
import numba
import cupy as cp  # GPU acceleration

class ParallelCollisionDetector:
    @numba.jit(nopython=True, parallel=True)
    def check_multiple_collisions_cpu(self, hand_positions, triangles):
        """CPU並列化による複数手の同時衝突検知"""
        n_hands = hand_positions.shape[0]
        n_triangles = triangles.shape[0]
        collisions = np.zeros((n_hands, n_triangles), dtype=np.bool_)
        
        for i in numba.prange(n_hands):
            for j in range(n_triangles):
                collisions[i, j] = self._check_sphere_triangle_collision(
                    hand_positions[i], triangles[j]
                )
                
        return collisions
        
    def check_collisions_gpu(self, hand_positions, triangles):
        """GPU並列化による超高速衝突検知"""
        # データをGPUに転送
        hand_pos_gpu = cp.asarray(hand_positions)
        triangles_gpu = cp.asarray(triangles)
        
        # GPU カーネルで並列処理
        collisions = self._gpu_collision_kernel(hand_pos_gpu, triangles_gpu)
        
        return cp.asnumpy(collisions)
```

### 実装の優先順位

1. **連続衝突検知（CCD）** - 最重要、高速移動への対応
2. **適応的サンプリング** - パフォーマンスと精度のバランス
3. **予測的衝突検知** - レイテンシー削減
4. **BVH最適化** - 大規模地形での性能向上
5. **音響パラメータ拡張** - より豊かな音表現

これらの改善により、高速な手の動きでも確実に衝突を検知し、より精密で表現力豊かな音生成が可能になります。