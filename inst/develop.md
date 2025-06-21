# 開発メモ: 実装内容の詳細言語化と問題点洗い出し

## 目次
1. 概要
2. src/detection
   1. hands2d.py
   2. hands3d.py
   3. tracker.py
3. src/mesh
   1. delaunay.py
   2. attributes.py
   3. index.py
   4. projection.py
   5. simplify.py
4. src/collision
   1. search.py
   2. sphere_tri.py
   3. events.py
5. 横断的な懸念事項

---

## 1. 概要
本ドキュメントでは、`src/detection`, `src/mesh`, `src/collision` 以下の主要モジュールを精読し、現状のアルゴリズム・クラス設計・実装品質を文章化したうえで、潜在的バグ・設計上の疑問点・パフォーマンス上の懸念を列挙する。実際の修正実装は行わず、**"何をどのように直すべきか"** を徹底的に洗い出すことを目的とする。

---

## 2. src/detection
### 2.1 hands2d.py (MediaPipe ラッパ―)
- **責務**: MediaPipe Hands を用いた2Dランドマーク検出、推論の統計収集、簡易描画。
- **主な構成**:
  - `HandLandmark`/`HandDetectionResult` dataclass
  - `MediaPipeHandsWrapper` クラス
    - GPU/CPU 自動切替、パラメータ動的更新、バッチ推論 (`detect_hands_batch`)
    - ランドマーク→結果変換 `_convert_to_hand_result`
    - 描画補助 `draw_landmarks`
- **気になる点**:
  1. **依存ライブラリ非存在時のモック**: `mp.solutions.hands.Hands` が `None` になるだけだと attribute error の可能性が高い。モッククラスをもう少し作り込む必要あり。
  2. **性能統計の保持方法**: dict に累積しているが thread-safe ではない。将来マルチスレッド化する場合はロック要。
  3. **`draw_landmarks`**: MEDIAPIPE_AVAILABLE が False でも connection 描画部で None アクセスする恐れ。

### 2.2 hands3d.py (2D→3D 投影)
- **責務**: 深度画像を使用して 2D 手ランドマークを 3D 座標へ射影。
- **主な構成**:
  - `Hand3DLandmark`, `Hand3DResult`, `DepthInterpolationMethod`, `Hand3DProjector`
- **アルゴリズム概要**:
  1. 深度前処理 (`_preprocess_depth`) で `uint16→float32` 変換、NaN マスク、ガウシアン平滑化。
  2. 各ランドマークに対して `_project_landmark_to_3d` で深度補間⇒カメラ射影。
  3. 信頼度が閾値以下なら結果棄却。
  4. ひら中心計算、性能統計更新。
- **懸念点**:
  1. **ガウシアン平滑の実装誤り**: `ndimage.gaussian_filter(depth_float[valid_mask].reshape(-1), …)` を再 reshape するが、元 shape 情報を失っているため深度画素の並びが壊れる。→ マスク領域ごとフィルタ or 画像全体にフィルタ必要。
  2. **`_get_interpolated_depth` 最近傍以外の分岐**: LINEARではバイリニア重み計算を独自実装しているが、`depth_image[i,j]` が `np.nan` の場合読み飛ばすため weight 正規化で 0 除算の可能性。
  3. **例外ハンドリング**: 多くの箇所が `except Exception as e: print…; return None` で落としており、上流で失敗理由が分かりにくい。

### 2.3 tracker.py (カルマンベース 3D 手トラッカー)
- **責務**: 3D 手結果をカルマンフィルタで時系列追跡、ID 付与。
- **主な処理フロー**: predict → assignment (Hungarian) → update → new track → lost track 処理。
- **問題点**:
  1. **加速度計算バグ**: `_predict_existing_tracks` 内で `prev_velocity = track.velocity` を更新後に再利用しているため常に 0 になる。予測前の velocity を保持すべき。
  2. **Assignment コスト**: handedness bonus 固定値が距離と同スケール (m) に混在。単位整合性が取れていない。
  3. **Covariance 初期化**: `__post_init__` と新規 Track 作成部で別々に初期化しており一貫性がない。
  4. **スレッドセーフでない dict 操作**: 将来的に非同期更新すると競合。

---

## 3. src/mesh
### 3.1 delaunay.py (メッシュ生成)
- **責務**: ハイトマップ/点群から Delaunay 分割し `TriangleMesh` を返却。
- **懸念点**:
  1. **`_is_valid_triangle` の面積計算**: 2D クロス積で面積推定しているが Z を無視。高低差が大きい場合に面積過小評価。
  2. **品質フィルタ後の頂点リマップ**で法線・色などの付随情報を失っている。MeshAfterFilter は属性欠落。
  3. **エラーハンドリング粒度**: `_perform_delaunay` で全例外を ValueError に丸めているため原因特定が困難。

### 3.2 attributes.py (ジオメトリ属性)
- **責務**: 法線・曲率・勾配などの計算。
- **気になる点**:
  1. **`calculate_vertex_normals` の面積重み**: 一般には角度重みや面積×角度が使われる。画質に影響。
  2. **曲率計算法の簡略化**: 離散ラプラシアンや角度欠損法のみで主曲率ベースの高品質曲率は未実装。
  3. **パフォーマンス**: Python ループ多数、N>10^5 でボトルネック必至。

### 3.3 index.py (BVH / KD-Tree)
- **問題点**:
  1. **`_predict_existing_tracks` と同様の統計**: 不要な処理複製がある。
  2. **BVH 分割基準**: SAH 簡略版とコメントあるが実装は center split のみで本当の SAH ではない。
  3. **Ray-Box 交差**: `inv_dir = 1/dir` で dir=0 のとき inf を使うが numpy.inf 演算の比較は注意。

### 3.4 projection.py (ハイトマップ生成)
- **懸念点**:
  1. **`_create_heightmap`** のセル統計計算を for ループで回しており大規模点群で遅い。
  2. **hole-filling**: 25px 以下の穴のみ対応。地形次第で不足。

### 3.5 simplify.py (Open3D 簡略化)
- **問題点**:
  1. **Open3D 依存**: GPU/CPU 切替やバージョン依存のエラー処理が未整備。
  2. **`_calculate_voxel_size`** で `target_voxels` = 0 になる可能性 (高い削減率)。division by zero。

---

## 4. src/collision
### 4.1 search.py (空間検索)
- **懸念点**:
  1. **キャッシュキー量子化**: 1 mm 単位だがハンドトラッキングの揺らぎでヒット率が低いかもしれない。ヒューリスティクス要調整。
  2. **`nodes_visited` 推定値**: 実測ではなく三角形数×2 で架空値。性能ログの信頼性欠如。

### 4.2 sphere_tri.py (精密判定)
- **懸念点**:
  1. **法線方向**: Face culling オプション true 時の dot 判定が "裏面" 判定になっていない可能性。
  2. **`_closest_point_on_triangle`** の barycentric 計算式が冗長でバグ混入リスク。退化三角形処理も要確認。
  3. **接触法線**: `contact_direction / (distance + 1e-8)` は距離ゼロ時に正規化誤差。

### 4.3 events.py (衝突→音イベント)
- **懸念点**:
  1. **イベントライフサイクル**: 毎フレーム `COLLISION_START` を生成し続けており `CONTINUE`/`END` 判定が手抜き。
  2. **`velocity_db` 計算**: ハンド速度を 0-100cm/s の仮定でスケールしているがハードコーディング。
  3. **Queue Overflow**: `maxlen` で自動削除されるが overflow カウンタのみ増やしている。失われたイベントが上層で分からない。

---

## 5. 横断的な懸念事項
1. **例外処理の一貫性**: ほぼ全ファイルで `except Exception as e: print(...)` が多用されている。→ ロギング基盤へ統一、スタックトレース保持。
2. **型アノテーション漏れ/誤り**: ndarray shape 情報や Optional 指定が曖昧。mypy などで静的解析したい。
3. **ユニットテスト不足**: モジュールごとのユーティリティ関数はあるが、CI 連携テストが皆無。
4. **パフォーマンス計測**: 各クラスに自己完結の stats dict があるが、統合ダッシュボードがなく個別ログ止まり。
5. **依存ライブラリ群のバージョン固定**: mediapipe, open3d, scipy など互換性チェックが必要。

---

### 改善アクション草案 (次フェーズで詳細設計予定)
- 共通エラーハンドリング + ロギングモジュール導入
- `hands3d._preprocess_depth` のフィルタ実装を NaN 対応 2D Gaussian へ置換
- Tracker の速度/加速度計算ロジック修正 + unittest
- Mesh attribute 計算を NumPy ベクトル化 or Open3D/CGAL 等 C++ バインディングで高速化
- Collision pipeline: BVH ノード訪問数正確計測、Search キャッシュ戦略再検討
- Events ライフサイクル管理: state machine 化し `CONTINUE`/`END` を適切出力

> **備考:** 上記は抜粋。各モジュールに対し詳述した修正案を別紙設計書として起案予定。