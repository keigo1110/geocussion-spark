# Geocussion-SP パフォーマンス改善メモ  

以下は **衝突判定 (Collision)** および **音響生成 (Audio)** 周辺コードを調査して判明した主要なボトルネック・非効率実装の一覧です。後続の実装フェーズでは、ここに列挙した `ISSUE-ID` を参照してタスクを管理してください。  

---

## 1. 衝突判定パイプライン  

| ID | 症状 / 問題点 | 影響範囲 | 参考コード | 想定影響 | 改善アイデア |
|----|---------------|----------|------------|-----------|--------------|
| CD-01 | `CollisionSearcher._calculate_distances` が毎呼び出し毎に `get_distance_calculator()` を再取得しており関数呼び出し＋ `import` コストが発生する | 全フレーム | ```260:270:src/collision/search.py
from .distance import get_distance_calculator
calculator = get_distance_calculator()
``` | ~0.1-0.3 ms / call × hands | • `self._dist_calc` に一度だけ保持し再利用  |
| CD-02 | 上記メソッド内で `mesh_vertices[mesh_triangles[triangle_indices]]` を毎回生成→大きな一時配列コピー | 衝突距離計算 | 同上 | >1 MB alloc / hand (高負荷) | • `np.take` + `out=` 再利用バッファ<br/>• 事前に三角形頂点ビューを `self._tri_vertices` として保持 |
| CD-03 | INFO レベルでフレーム毎に大量ログ (`[COLLISION] Frame ...`) を出力 | I/O, CPU | ```5200:5210:demo_collision_detection.py``` (*複数箇所*) | 2-4 ms / frame (stdio flush) | • デフォルトを `logger.debug` に格下げ<br/>• `if frame_cnt % N == 0:` で間引き |
| CD-04 | GPU 距離計算の閾値が高く、小さな三角形集合でも CPU ループにフォールバックしがち | GPU 対応環境 | `SphereTriangleCollision._perform_collision_test` | 3-5 ms / hand | • 閾値(>=2) で GPU パス強制<br/>• `cupy` バッチ距離関数の小集合最適化 |
| CD-05 | `CollisionSearcher.search_cache` は dict による手製 LRU。削除タイミングが粗く、メモリ無制限で増加 | 長時間実行 | ```300:330:src/collision/search.py``` | RAM 増加 / GC stop-the-world | • `collections.OrderedDict` で O(1) LRU<br/>• `functools.lru_cache` 併用 |
| CD-06 | `ContinuousCollisionDetector.generate_interpolation_samples` で Python ループ生成 → NumPy でベクトル化可能 | 高速手移動時 | ```90:130:src/collision/continuous_detection.py``` | 0.3-0.8 ms / hand | • `np.linspace` + broadcasting で一括生成 |

---

## 2. 音響生成パイプライン  

| ID | 症状 / 問題点 | 影響範囲 | 参考コード | 想定影響 | 改善アイデア |
|----|---------------|----------|------------|-----------|--------------|
| AU-01 | `SimpleSoundBank._generate_all_instruments` が起動時に **22 周波数 × 4 楽器 × 2 ch × 数秒** の波形を全生成 → 初期化 1-2 秒＆>20 MB 常駐 | 起動時メモリ & 起動遅延 | ```200:240:src/sound/simple_synth.py``` | 起動体験低下 | • Lazy 生成: 初回要求時にキャッシュ<br/>• 8-bit ADPCM 圧縮サンプルへ切替 |
| AU-02 | `get_sound()` がヒットしない場合に文字列分割＋線形探索で最近周波数を検索 → O(N) | 各ノート発音 | 同上 | 0.05-0.2 ms / note | • 周波数→index dict を事前構築<br/>• `bisect` で対数探索 |
| AU-03 | `SimpleAudioSynthesizer.play_audio_parameters` 全体を `_lock` でシリアライズ → 多重発音時にスレッド競合 | 高速連打時 | ```330:380:src/sound/simple_synth.py``` | ドロップ/レイテンシ 5-15 ms | • lock-free ring buffer と AudioThread で非同期再生<br/>• `queue.SimpleQueue` でイベント投入 |
| AU-04 | INFO レベルでノート毎に `Audio played successfully ...` を出力 | I/O | 同上 | 1-2 ms / note | • debug に変更 or サンプリング出力 |
| AU-05 | `pygame.mixer.find_channel()` 線形走査; チャンネル数≦32でも頻度高 | 発音時 | pygame internals | 0.1 ms / note | • 自前で空きチャネルスタック管理 |
| AU-06 | オーディオバッファ 512 サンプル (≈11.6 ms) でレイテンシ可視化 | 全体 | mixer.pre_init | 可聴レイテンシ | • 256 サンプルに短縮 (CPU 使用率許容なら) |

---

## 3. 共通・その他  

| ID | 症状 | 改善アイデア |
|----|------|--------------|
| GC-01 | NumPy / CuPy 大量一時配列で GC 圧迫 → 明示的 `gc.collect()` 10 s 毎に呼び出し済だがまだスパイク | • オブジェクトプール導入 & `np.empty` 再利用 |
| LOG-01 | デフォルトログレベルが `INFO` で冗長。開発時のみ有効にし、本番は `WARN` 以上推奨 | • 起動引数 `--log-level` を追加 |

---

### 優先度案

1. **CD-01/CD-02/CD-03** ‑ 毎フレーム発生しフレームタイム直結 → 先行して修正
2. **AU-01/AU-02/AU-03** ‑ 体験に直結 (起動遅延 & 音切れ)
3. **CD-05/CD-06** ‑ 長時間・高速動作時の安定性
4. **その他** ‑ 余力次第

---

> **次ステップ** : 上記テーブルをもとに個別 PR / コミットを切り、`speedup.md` に進捗チェックボックスを追加していく予定。
