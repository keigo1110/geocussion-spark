"""
GPU-accelerated distance calculation module using CuPy
GPU最適化距離計算モジュール (CuPy活用)

This module provides CUDA-accelerated implementations of distance calculations
between points and triangles, optimized for real-time collision detection.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any
import logging

# GPU 依存関係の安全なインポート
try:
    import cupy as cp
    from cupy.cuda import Device
    GPU_AVAILABLE = True
    CpArray = cp.ndarray
except ImportError:
    cp = None
    Device = None
    GPU_AVAILABLE = False
    CpArray = Any  # CuPy未インストール時のフォールバック

from src import get_logger
logger = get_logger(__name__)

# GPU メモリ管理設定
GPU_MEMORY_POOL_ENABLED = True
BATCH_SIZE_THRESHOLD = 1000  # この数以上でGPU処理に切り替え
MAX_GPU_MEMORY_RATIO = 0.8   # GPU メモリの80%まで使用

# CUDA カーネルコード（真の並列処理）
POINT_TRIANGLE_DISTANCE_KERNEL = """
extern "C" __global__
void point_triangle_distance_kernel(
    const double* points,      // (N, 3)
    const double* triangles,   // (M, 3, 3)
    double* distances,         // (N, M) output
    int N, int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= N || idy >= M) return;
    
    // Point coordinates
    double px = points[idx * 3 + 0];
    double py = points[idx * 3 + 1];
    double pz = points[idx * 3 + 2];
    
    // Triangle vertex coordinates
    int tri_base = idy * 9;  // 3 vertices * 3 coordinates
    double v0x = triangles[tri_base + 0];
    double v0y = triangles[tri_base + 1];
    double v0z = triangles[tri_base + 2];
    double v1x = triangles[tri_base + 3];
    double v1y = triangles[tri_base + 4];
    double v1z = triangles[tri_base + 5];
    double v2x = triangles[tri_base + 6];
    double v2y = triangles[tri_base + 7];
    double v2z = triangles[tri_base + 8];
    
    // Edge vectors
    double edge0x = v1x - v0x;
    double edge0y = v1y - v0y;
    double edge0z = v1z - v0z;
    double edge1x = v2x - v0x;
    double edge1y = v2y - v0y;
    double edge1z = v2z - v0z;
    
    // Vector from v0 to point
    double v0_to_px = px - v0x;
    double v0_to_py = py - v0y;
    double v0_to_pz = pz - v0z;
    
    // Dot products
    double a = edge0x * edge0x + edge0y * edge0y + edge0z * edge0z;
    double b = edge0x * edge1x + edge0y * edge1y + edge0z * edge1z;
    double c = edge1x * edge1x + edge1y * edge1y + edge1z * edge1z;
    double d = edge0x * v0_to_px + edge0y * v0_to_py + edge0z * v0_to_pz;
    double e = edge1x * v0_to_px + edge1y * v0_to_py + edge1z * v0_to_pz;
    
    // Barycentric coordinates
    double det = a * c - b * b;
    double s = b * e - c * d;
    double t = b * d - a * e;
    
    double final_s, final_t;
    
    // Region classification and closest point computation
    // Use higher precision epsilon for robust computations
    const double EPS = 1e-12;
    
    if (s + t <= det + EPS) {
        if (s < -EPS) {
            if (t < -EPS) {
                // Region 4
                if (d < -EPS) {
                    final_t = 0.0;
                    final_s = fmax(0.0, fmin(1.0, -d / a));
                } else {
                    final_s = 0.0;
                    final_t = fmax(0.0, fmin(1.0, -e / c));
                }
            } else {
                // Region 3
                final_s = 0.0;
                final_t = fmax(0.0, fmin(1.0, -e / c));
            }
        } else if (t < -EPS) {
            // Region 5
            final_t = 0.0;
            final_s = fmax(0.0, fmin(1.0, -d / a));
        } else {
            // Region 0 (inside triangle)
            if (fabs(det) > EPS) {
                double inv_det = 1.0 / det;
                final_s = s * inv_det;
                final_t = t * inv_det;
            } else {
                // Degenerate triangle
                final_s = 0.0;
                final_t = 0.0;
            }
        }
    } else {
        if (s < -EPS) {
            // Region 2
            double tmp0 = b + d;
            double tmp1 = c + e;
            if (tmp1 > tmp0 + EPS) {
                double numer = tmp1 - tmp0;
                double denom = a - 2.0 * b + c;
                if (fabs(denom) > EPS) {
                    final_s = fmax(0.0, fmin(1.0, numer / denom));
                } else {
                    final_s = 0.0;
                }
                final_t = 1.0 - final_s;
            } else {
                final_s = 0.0;
                final_t = fmax(0.0, fmin(1.0, -e / c));
            }
        } else if (t < -EPS) {
            // Region 6
            double tmp0 = b + e;
            double tmp1 = a + d;
            if (tmp1 > tmp0 + EPS) {
                double numer = tmp1 - tmp0;
                double denom = a - 2.0 * b + c;
                if (fabs(denom) > EPS) {
                    final_t = fmax(0.0, fmin(1.0, numer / denom));
                } else {
                    final_t = 0.0;
                }
                final_s = 1.0 - final_t;
            } else {
                final_t = 0.0;
                final_s = fmax(0.0, fmin(1.0, -d / a));
            }
        } else {
            // Region 1
            double numer = c + e - b - d;
            if (numer <= EPS) {
                final_s = 0.0;
            } else {
                double denom = a - 2.0 * b + c;
                if (fabs(denom) > EPS) {
                    final_s = fmax(0.0, fmin(1.0, numer / denom));
                } else {
                    final_s = 0.0;
                }
            }
            final_t = 1.0 - final_s;
        }
    }
    
    // Closest point computation
    double closest_x = v0x + final_s * edge0x + final_t * edge1x;
    double closest_y = v0y + final_s * edge0y + final_t * edge1y;
    double closest_z = v0z + final_s * edge0z + final_t * edge1z;
    
    // Distance computation
    double diff_x = px - closest_x;
    double diff_y = py - closest_y;
    double diff_z = pz - closest_z;
    double distance = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
    
    // Store result
    distances[idx * M + idy] = distance;
}
"""

class GPUDistanceCalculator:
    """GPU最適化距離計算器
    
    CuPy を使用して点-三角形距離計算をGPU上で並列実行する高性能実装
    """
    
    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        """
        Args:
            use_gpu: GPU使用フラグ
            device_id: 使用するGPUデバイスID
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device_id = device_id
        self.device = None
        self.memory_pool = None
        self.cuda_kernel = None
        
        # 統計情報
        self.gpu_calculations = 0
        self.cpu_fallback_calculations = 0
        self.total_gpu_time = 0.0
        self.total_cpu_time = 0.0
        
        if self.use_gpu:
            self._initialize_gpu()
    
    def _initialize_gpu(self) -> bool:
        """GPU環境を初期化"""
        try:
            if not GPU_AVAILABLE:
                logger.warning("CuPy not available, GPU initialization skipped")
                self.use_gpu = False
                return False
                
            self.device = Device(self.device_id)
            with self.device:
                # メモリプール設定
                if GPU_MEMORY_POOL_ENABLED:
                    self.memory_pool = cp.get_default_memory_pool()
                    self.memory_pool.set_limit(fraction=MAX_GPU_MEMORY_RATIO)
                
                # CUDAカーネルをコンパイル
                self.cuda_kernel = cp.RawKernel(POINT_TRIANGLE_DISTANCE_KERNEL, 'point_triangle_distance_kernel')
                
                # GPU情報ログ出力
                device_name = cp.cuda.runtime.getDeviceProperties(self.device_id)['name'].decode()
                total_mem = cp.cuda.runtime.getDeviceProperties(self.device_id)['totalGlobalMem']
                logger.info(f"GPU Distance Calculator initialized: {device_name} ({total_mem // 1024**2} MB)")
                logger.info("CUDA kernel compiled successfully")
                
                return True
                
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}, falling back to CPU")
            self.use_gpu = False
            return False
    
    def calculate_point_triangle_distance_gpu(
        self,
        points: Union[np.ndarray, CpArray],
        triangles: Union[np.ndarray, CpArray]
    ) -> Union[np.ndarray, CpArray]:
        """
        GPU最適化版: 点群と三角形群の距離を一括計算
        
        Args:
            points: 点群 (N, 3)
            triangles: 三角形頂点群 (M, 3, 3)
            
        Returns:
            距離行列 (N, M)
        """
        if not self.use_gpu or self.cuda_kernel is None:
            return self._calculate_cpu_fallback(points, triangles)
        
        try:
            with self.device:
                import time
                start_time = time.perf_counter()
                
                # データをGPUに転送（float64に変換）
                if isinstance(points, np.ndarray):
                    points_gpu = cp.asarray(points, dtype=cp.float64)
                else:
                    points_gpu = points.astype(cp.float64)
                    
                if isinstance(triangles, np.ndarray):
                    triangles_gpu = cp.asarray(triangles, dtype=cp.float64)
                else:
                    triangles_gpu = triangles.astype(cp.float64)
                
                # 入力の形状を整える
                N, _ = points_gpu.shape
                M, _, _ = triangles_gpu.shape
                
                # 三角形データを (M, 9) に平坦化
                triangles_flat = triangles_gpu.reshape(M, 9)
                
                # 出力配列
                distances_gpu = cp.zeros((N, M), dtype=cp.float64)
                
                # スレッドブロック設定
                threads_per_block = (16, 16)
                blocks_per_grid = (
                    (N + threads_per_block[0] - 1) // threads_per_block[0],
                    (M + threads_per_block[1] - 1) // threads_per_block[1]
                )
                
                # CUDAカーネル実行
                self.cuda_kernel(
                    blocks_per_grid,
                    threads_per_block,
                    (
                        points_gpu,
                        triangles_flat,
                        distances_gpu,
                        N,
                        M
                    )
                )
                
                # GPU同期
                cp.cuda.Stream.null.synchronize()
                
                # 結果をCPUに戻す（必要に応じて）
                if isinstance(points, np.ndarray):
                    result = distances_gpu.get()
                else:
                    result = distances_gpu
                
                # 統計更新
                self.gpu_calculations += N * M
                self.total_gpu_time += time.perf_counter() - start_time
                
                return result
                
        except Exception as e:
            logger.warning(f"GPU calculation failed: {e}, falling back to CPU")
            return self._calculate_cpu_fallback(points, triangles)
    
    def _calculate_cpu_fallback(
        self,
        points: np.ndarray,
        triangles: np.ndarray
    ) -> np.ndarray:
        """CPU フォールバック実装"""
        import time
        start_time = time.perf_counter()
        
        # Numba JIT版を使用
        from src.collision.distance import batch_point_triangle_distances
        result = batch_point_triangle_distances(points, triangles)
        
        # 統計更新
        self.cpu_fallback_calculations += points.shape[0] * triangles.shape[0]
        self.total_cpu_time += time.perf_counter() - start_time
        
        return result
    
    def calculate_batch_distances_adaptive(
        self,
        points: np.ndarray,
        triangles: np.ndarray,
        force_gpu: bool = False
    ) -> np.ndarray:
        """
        適応的バッチ距離計算
        データサイズに応じてGPU/CPU処理を自動選択
        
        Args:
            points: 点群 (N, 3)
            triangles: 三角形群 (M, 3, 3)
            force_gpu: GPU強制使用フラグ
            
        Returns:
            距離行列 (N, M)
        """
        total_calculations = points.shape[0] * triangles.shape[0]
        
        # 処理方法の自動選択
        use_gpu_processing = (
            self.use_gpu and (
                force_gpu or 
                total_calculations >= BATCH_SIZE_THRESHOLD
            )
        )
        
        if use_gpu_processing:
            logger.debug(f"Using GPU for {total_calculations:,} distance calculations")
            return self.calculate_point_triangle_distance_gpu(points, triangles)
        else:
            logger.debug(f"Using CPU for {total_calculations:,} distance calculations")
            return self._calculate_cpu_fallback(points, triangles)
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計を取得"""
        total_calc = self.gpu_calculations + self.cpu_fallback_calculations
        gpu_ratio = self.gpu_calculations / total_calc if total_calc > 0 else 0
        
        gpu_avg_time = self.total_gpu_time / self.gpu_calculations if self.gpu_calculations > 0 else 0
        cpu_avg_time = self.total_cpu_time / self.cpu_fallback_calculations if self.cpu_fallback_calculations > 0 else 0
        
        return {
            'total_calculations': total_calc,
            'gpu_calculations': self.gpu_calculations,
            'cpu_fallback_calculations': self.cpu_fallback_calculations,
            'gpu_usage_ratio': gpu_ratio,
            'gpu_avg_time_per_calc': gpu_avg_time,
            'cpu_avg_time_per_calc': cpu_avg_time,
            'speedup_ratio': cpu_avg_time / gpu_avg_time if gpu_avg_time > 0 else 0,
            'gpu_available': self.use_gpu
        }
    
    def cleanup(self):
        """リソースクリーンアップ"""
        if self.memory_pool:
            self.memory_pool.free_all_blocks()
        logger.info("GPU Distance Calculator cleaned up")


# 統合インターフェース関数
def create_gpu_distance_calculator(use_gpu: bool = True, device_id: int = 0) -> GPUDistanceCalculator:
    """GPU距離計算器を作成"""
    return GPUDistanceCalculator(use_gpu=use_gpu, device_id=device_id)


def batch_point_triangle_distances_gpu(
    points: np.ndarray,
    triangles: np.ndarray,
    use_gpu: bool = True
) -> np.ndarray:
    """
    GPU最適化版バッチ距離計算（統一インターフェース）
    
    Args:
        points: 点群 (N, 3)
        triangles: 三角形群 (M, 3, 3)
        use_gpu: GPU使用フラグ
        
    Returns:
        距離行列 (N, M)
    """
    calculator = create_gpu_distance_calculator(use_gpu=use_gpu)
    try:
        return calculator.calculate_batch_distances_adaptive(points, triangles)
    finally:
        calculator.cleanup()


# 既存APIとの互換性維持
def point_triangle_distance_gpu(
    point: np.ndarray,
    triangle: np.ndarray,
    use_gpu: bool = True
) -> float:
    """
    単一点-三角形距離計算（GPU対応版）
    
    Args:
        point: 点座標 (3,)
        triangle: 三角形頂点 (3, 3)
        use_gpu: GPU使用フラグ
        
    Returns:
        距離値
    """
    points = point.reshape(1, 3)
    triangles = triangle.reshape(1, 3, 3)
    
    distances = batch_point_triangle_distances_gpu(points, triangles, use_gpu=use_gpu)
    return float(distances[0, 0]) 