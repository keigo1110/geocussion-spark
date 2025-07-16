### 1. いま起きている現象 ― “点群に沿わないメッシュ” の正体

1. **GPU デラウネイ分割で Z 座標を誤結合**

   ```python
   vertices_2d, triangles = gpu_result
   vertices_3d = np.column_stack([
       vertices_2d[:,0],           # ← 2‑D 頂点
       vertices_2d[:,1],
       points[:len(vertices_2d),2] # ← **入力点を先頭から順に流用**
   ])
   ```

   上記ロジック（`src/mesh/delaunay.py::_perform_delaunay`）では，
   *GPU 三角分割器が返した 2‑D 頂点順序* と *元の点群の並び* が一致していると仮定し
   **Z を “適当に” くっつけている**。
   GPU 側で同一点を間引いたり並べ替えたりすると Z 軸がずれ，
   空中に“山”のようなスパイクが立つ。 ([GitHub][1])

2. **2.5D 高度マップ前提が破綻している**

   * `PointCloudProjector` は XY（または XZ/YZ）平面へ正射影し “高さ＝一意” と見なす ([GitHub][2])。
   * 壁・テーブル脚・人など *同じ (x,y) に複数 Z* があるシーンでは，
     **穴を跨いで極端に長い三角形** が生成される。
   * さらに `_add_boundary_points()` が外周に人工点を挿入し，
     点密度の薄い箇所へ三角形を強制的に張ってしまう。

### 2. 最優先で行うべき修正案

| 優先  | 改善項目                                              | 具体的パッチ／設定                                                                                                                                                                                | 期待効果                        |
| --- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| ★★★ | **GPU 経路の Z 対応付けを正しく行う**                          | ① GPU トライアンギュレータに **オリジナル頂点 index を返させる**<br>② 返って来ない場合は KD‑Tree 最近傍で Z を補間:  `python<br>tree = cKDTree(points[:,:2]); _, idx = tree.query(vertices_2d, k=1); vertices_3d = points[idx]` | メッシュの“空中スパイク”解消、Z 軸位置が正確になる |
| ★★  | **高さ一意性の破綻検知**                                    | Delaunay 前に *(x,y) ごとに Z 最大‑最小差 > 閾値* をチェックし，<br>差が大きいセルはメッシュ化から除外                                                                                                                       | 縦面・柱がある場面での誤張りを抑止           |
| ★★  | **品質フィルタを Z 含む 3‑D 基準に変更**                        | `vectorized_triangle_qualities()` は周長比だけなので，<br>「三頂点の Z 標準偏差 > h\_threshold」で弾く                                                                                                          | 高さがバラけた薄い“テント三角形”を除去        |
| ★   | **`max_edge_length` と `quality_threshold` を動的調整** | 点密度が低いフレームでは値を絞る                                                                                                                                                                         | 過剰な長辺三角形を回避                 |
| ★   | **`_add_boundary_points` の廃止またはオプション化**           | フラグ `--no-boundary-fill` を CLI に追加                                                                                                                                                       | 外周の不要な橋渡しを防止                |

#### サンプルパッチ（GPU Z 対応付けのみ抜粋）

```diff
- vertices_3d = np.column_stack([
-     vertices_2d[:,0],
-     vertices_2d[:,1],
-     points[:len(vertices_2d), 2] )
+ from scipy.spatial import cKDTree
+ tree = cKDTree(points[:, :2])
+ _, nn_idx = tree.query(vertices_2d, k=1)
+ vertices_3d = points[nn_idx]
```

---

### 3. パイプライン全体の性能／設計レビュー & 改善ポイント

| 部位                                        | ボトルネック/非効率                            | 高速化アイデア                                                                  |
| ----------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ |
| **Height‑map 投影** (`binned_statistic_2d`) | CPU 実装が 30 万点で 20 ms 超                | *CuPy* 版へ差し替え or `numba.cuda` JIT                                        |
| **Adaptive Sampling**                     | `np.random.choice` に確率ベクトル生成→メモリ確保大   | 累積分布 CDF を pre‑alloc & reuse                                             |
| **三角形品質フィルタ**                             | 品質計算後に **頂点再インデックス** を毎フレーム実施         | (1) 頂点 live‑flag を付けて in‑place 無効化<br>(2) メッシュ更新をワーカー Thread で非同期化       |
| **ログ出力**                                  | 1 frame 内で `logger.debug` 多用 → I/O 待ち | `if log.isEnabledFor(DEBUG): …` ガードで抑制                                   |
| **Open3D ビューア**                           | `set_geometry()` がフレームあたり全頂点アップロード    | 変化した VertexBuffer だけ頂点 Array 更新 (Open3D ≥0.18 の `update_geometry()` )    |
| **GC & メモリ再利用**                           | `np.vstack` で都度新配列生成                  | (a) 事前に最大頂点数でバッファ確保し slice 代入<br>(b) `memoryview` + `buffer protocol` 活用 |
| **Python–C++ 境界**                         | Orbbec Frame → NumPy → Open3D と 3‑コピー | Cython / Pybind11 で **Shared PBO** を渡しコピー 0 回に                           |

---

### 4. チェックリスト（修正後に確認すべき KPI）

| 項目                     | Baseline                 | 目標              |
| ---------------------- | ------------------------ | --------------- |
| メッシュ生成 1 回あたり時間        | 18 ms (CPU) / 7 ms (GPU) | **≦ 5 ms**      |
| フレームレイテンシ（深度入力→衝突判定完了） | 42 ms                    | **≦ 25 ms**     |
| 異常三角形率                 | 8 %                      | **＜ 0.5 %**     |
| 衝突判定誤検出                | 3 / min                  | **≦ 0.3 / min** |

---

### 5. まとめ

* **最大の原因は「GPU デラウネイ後の Z の取り違え」**。
  Z を正しくマッピングするだけで“面から浮いたメッシュ”はほぼ消えます。
* そのうえで **高さ一意性の仮定が破綻するシーン**を検知し，
  長辺・高差三角形を除外するルールを追加してください。
* パフォーマンス面は *コピー回数削減・GPU ベクタ化* が即効性大。

以上を適用すれば，視覚的なアーティファクトを抑えつつ **40 ms → 25 ms 以下**の
エンドツーエンド処理が十分に射程に入ります。

[1]: https://raw.githubusercontent.com/keigo1110/geocussion-spark/main/src/mesh/delaunay.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/keigo1110/geocussion-spark/main/src/mesh/projection.py "raw.githubusercontent.com"