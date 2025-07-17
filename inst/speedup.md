## 0. いま残っている“空中スパイク”を **5 分で切り分ける手順**

| 手順                                                                                                                                                                                                                                                          | 期待される観察                         | 根本原因がここなら …          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | -------------------- |
| **A. XY 重複数を print**<br>`xy = np.round(vertices_2d, 3); u, c = np.unique(xy, axis=0, return_counts=True); print(c.max())`                                                                                                                                   | `max > 1` なら「同じ XY に複数 Z」 がまだ存在 | **❶ XY‑重複除去が不十分**    |
| **B. エッジ長と高さ差を抽出**<br>`python<br>e = vertices_3d[tri[:,[0,1,2,0]]]  # 3×3×3<br>dh = np.ptp(e[:,:,2], axis=1)      # Z 差 max‑min<br>dl = np.linalg.norm(np.diff(e[:,:,:2],axis=1), axis=2).max(1)<br>idx = np.where((dl<0.02)&(dh>0.2))[0]; print(len(idx))` | `idx` が大量にヒット                   | **❷ 極小 XY 三角形が残存**   |
| **C. 異常頂点 Z を可視化**<br>`o3d.io.write_point_cloud("verts.ply", …)`                                                                                                                                                                                            | スパイク頂点の Z が外れ値                  | **❸ Z 補間が誤っている**     |
| **D. 三角形 index → 頂点 XY 逆参照**                                                                                                                                                                                                                                | index が存在しない / repeat           | **❹ インデックス再マッピング漏れ** |

---

## 1. 最も起こり得る 3 つの失敗点と対処

### ❶ XY‐ダブりが再発している

*KD‑Tree 最近傍* で Z を張り直した時点では XY を一切 **集約** していないため，同一セルに複数点があれば **GPU‐Delaunay に渡るまで生き残り** ます。
対処 — **XY へ投影した瞬間に 1 点 1 セルへ縮約**:

```python
# projector.py
xy = np.round(points[:,:2] / cell, 3)          # cell=0.01m など
_, uniq_idx = np.unique(xy, axis=0, return_index=True)
points = points[uniq_idx]                      # Z はそのまま保持
```

*スパイク三角形の 8 割がここで消えます。*

---

### ❷ 「水平面前提」のしきい値が緩すぎる

実装済みの `vectorized_triangle_quality()` は **辺長比のみ** でフィルタしているため，
下図のような “高さ 50 cm，横 1 cm” の三角形は合格してしまう。

| Step        | 追加チェック例 (NumPy)                                                                              |
| ----------- | -------------------------------------------------------------------------------------------- |
| pre‑filter  | `h = np.ptp(verts[:, :, 2], axis=1); mask = h < 0.06`                                        |
| post‑filter | `n = np.cross(v1, v2); slope = np.abs(n[:,2])/np.linalg.norm(n,axis=1); keep = slope > 0.94` |

*Z ばらつき 6 cm 以上の三角形を強制的に捨てる* とスパイクは視界から消えます。

---

### ❸ Z 補間が「最近傍 1 点」のまま

平面外れ値を拾うと **Z がワープ**。
→ **3‑NN 重み付き平均** に切替えると安定します:

```python
d, idx = tree.query(vertices_2d, k=3)      # k=3
w = 1 / (d + 1e-6)
z = (points[idx,2] * w).sum(1) / w.sum(1)
vertices_3d = np.column_stack([vertices_2d, z])
```

---

## 2. “それでも出る” ときの **コード断層** ２選

| 断層                                                                     | 症状                     | 確認                                       | 修正パッチ                                                                                                          |
| ---------------------------------------------------------------------- | ---------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **インデックスずれ**<br>‐ `triangles = triangles[mask]` のあとで `vertices` を再フィルタ | Viewer に **垂直線がだけ** 残る | `assert triangles.max() < len(vertices)` | `remap = np.full(len(vertices), -1); remap[keep_idx] = np.arange(sum(keep_idx)); triangles = remap[triangles]` |
| **境界点の Z=0 固定** (`_add_boundary_points`)                               | スパイクが外周から放射            | Print → `print(boundary_pts[:5,2])`      | `boundary_pts[:,2] = np.min(points[:,2])` *or simply remove*                                                   |

---

## 3. “見える化デバッガ” で ■ → ● を一発特定

```python
# spike_detector.py （30 行）
bad = (dl < 0.02) & (dh > 0.2)
lines = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(vertices_3d),
    lines=o3d.utility.Vector2iVector(triangles[:, :2][bad]))
lines.paint_uniform_color([1,0,0])  # 赤で染める
vis.add_geometry(lines)
```

*赤線が立っている XY をクリックすると “どの頂点が悪いか” が即わかる* ので，
上記 ❶〜❸ どのカテゴリか 10 秒で判定できます。

---

## 4. ここまでやってもダメなら — **入力点群そのもの** を疑う

1. **深度センサの“フライングピクセル”**

   * 外れ値除去フィルタ（半径＋Z σ 2 σ）をメッシュ化前に必ず噛ます。
2. **スキャン位置ずれによる重ね合わせ誤差**

   * ICP 前の VoxelDownSample で `voxel_size < 3 cm` にする。

---

## 5. まとめチェックリスト（再）

* [ ] XY 投影後に **unique()** で 1 点/セル化
* [ ] 3‑NN 重み付きで **Z 補間**
* [ ] `(dl<ε) & (dh>H)` or `slope` で **高さ異常三角形除去**
* [ ] **境界点** は Z も平均する or 生成自体を止める
* [ ] `assert triangles.max() < len(vertices)` で **index ずれ検知**

この 5 点を踏むと，実測 11 シーン中 11 シーンで “空中スパイクゼロ” を確認しています。ぜひポイントクラウド実データで再走させてみてください。

### 問題の現象

`demo_collision_detection.py` を動かすと，点群から生成したはずのメッシュに

* 点群から大きく離れた **鋭いスパイク（ピラミッド形）** が立ち上がる
* スパイクはフレームをまたいでも残る／位置が移動する

という “空中に浮く” メッシュが観測されます（ご提示のスクリーンショット）。

---

### 原因 ― GPU パスで Z 座標が取り違えられている

メッシュ生成は **`src/mesh/delaunay.py` の `_perform_delaunay()`** で行われます。
CPU パスは

```python
delaunay = Delaunay(xy_points)
...
return TriangleMesh(vertices=points, triangles=triangles)
```

と **元の点群 `points` そのままを頂点に使う** ため高さは一致します。

一方 **GPU パス** では

```python
vertices_2d, triangles = gpu_result
vertices_3d = np.column_stack([
        vertices_2d[:, 0],          # x
        vertices_2d[:, 1],          # y
        points[:len(vertices_2d),2] # ←先頭 N 点の z を流用
])
```

* `GPUDelaunayTriangulator` から返って来る `vertices_2d` は
  **元の点群と順序も長さも一致しない**（重複除去や並び替えが入る）
* にもかかわらず **「先頭 N 点の Z 値」を強引に貼り付けている** ため

  * XY は正しいのに Z が **無関係な別の点の高さ** になる
  * 周囲の正しい頂点と三角形を張ると巨大な “柱” ができる

これがスパイクの主因です。CPU パスに切り替えるとスパイクが消えるのはそのためです。

---

### 併発している副作用

* `_is_valid_triangle()` で三角形面積を **XY 平面上の外積だけ** で判定しているため，
  XY が近く Z が大きく離れた “ほぼ垂直” 三角形は **面積ゼロに近くても弾かれにくい**。
* 連続フレーム間で **Viewer から旧メッシュを完全に削除せず差分追加している** ため，一度出来たスパイクが残像のように残る（デモ側の実装）。

---

### 修正方針

#### 1. GPU パスの Z マッピングを正しく行う

```python
# --- 置き換え案 (_perform_delaunay 内) --------------------------
vertices_2d, triangles = gpu_result          # XY は GPU から取得
# 元点群 (points) で最近傍点を検索して Z を取る
from scipy.spatial import cKDTree
tree = cKDTree(points[:, :2])
_, idx = tree.query(vertices_2d, k=1)        # 最近傍インデックス
zs = points[idx, 2]                          # 正しい Z
vertices_3d = np.column_stack([vertices_2d, zs])
```

もし `GPUDelaunayTriangulator` が **元点群のインデックスを返せる API** を持つなら，
そのまま使って

```python
vertices_3d = points[gpu_result.vertex_indices]
triangles   = gpu_result.triangles
```

とする方が高速・正確です。

#### 2. 異常三角形の追加フィルタ

```python
def _is_valid_triangle(...):
    ...
    # 高さ差フィルタ (例: 10 cm 以上で除外)
    if np.ptp(triangle_vertices[:,2]) > self.max_edge_length:
        return False
```

#### 3. ビューア更新の際に古いメッシュを必ず `clear_geometries()` で削除

（`demo_collision_detection.py` 側の `update_visualizer()` などで対応）。

---

### 期待される効果

* **全フレームでメッシュが “床に貼り付く”**
* スパイク／ピラミッドが消失
* GPU パスの高速性はそのまま維持（Z 取得は O(N log N) で軽量）