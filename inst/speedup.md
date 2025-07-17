## 4. 山サイズ別楽器マッピング (MOUNT-INS-01)

### 概要
地形メッシュをクラスタリングし、各「山（連結面パッチ）」に対して固定の楽器タイプを事前に割り当てる。衝突イベント発生時は三角形→山 ID→楽器テーブルの 2 回ルックアップのみで決定するため、ランタイム負荷を増やさずプレイフィールの一貫性を高める。

### 要件
- メッシュ更新タイミング以外では追加計算コスト ≒ 0。
- 衝突イベントから音生成までのレイテンシを増やさない。
- **底面(深度 ≥ `drum_depth`) は強制的に `DRUM`**。
- 山単位で楽器が固定され、プレイヤーが視覚的に理解できる。
- 既存 CLI／既存ロジックを壊さない。初期値を与えれば従来と同じ挙動になる。

### 設計方針
1. **クラスタリング (Mesh 時間)**
   - メッシュ生成完了時に高低差・法線角度・隣接関係で flood-fill 探索。
   - `mountain_id : int` を各三角形へ付与、面積・標高など統計を収集。
2. **楽器割り当て**
   - 面積順位やユーザ定義ルールで `mountain_id → InstrumentType` テーブル (`mountain_instrument_table`) を作成。
3. **イベント拡張**
   - `CollisionEvent` に `mountain_id: int` を追加。
   - `CollisionEventQueue.create_event()` で衝突三角形→山 ID を 1 ルックアップ。
4. **AudioMapper 変更**
   - `mountain_instrument_table` を受け取り、`_select_instrument()` 冒頭でテーブル参照。
   - フォールバックとして既存アルゴリズムを維持。
5. **可視化**
   - DualViewer へ山 ID→色マッピングを渡し、地形ワイヤーフレームを色分け表示 (ON/OFF)。

### 実装タスク
- [ ] **MOUNT-INS-01-1**  クラスタリング関数 `cluster_mountains(tri_mesh)` の実装
- [ ] **MOUNT-INS-01-2**  `triangle_to_mountain: np.ndarray[int32]` をメッシュ属性に保持
- [ ] **MOUNT-INS-01-3**  楽器割当ヘルパ `assign_instruments_by_area(stats)`
- [ ] **MOUNT-INS-01-4**  `CollisionEvent.mountain_id` 追加とキュー内での設定
- [ ] **MOUNT-INS-01-5**  `AudioMapper` へテーブル注入＆ルックアップ処理追加
- [ ] **MOUNT-INS-01-6**  CLI オプション `--mountain-colors` (optional) / 可視化トグルキー
- [ ] **MOUNT-INS-01-7**  ユニットテスト & パフォーマンス計測
- [ ] **MOUNT-INS-01-8**  ドキュメント更新 (README / ヘルプメッセージ)

### 影響範囲
- `mesh.pipeline`→`PipelineManager` (メッシュ後処理フック追加)
- `collision.events` (構造体拡張)
- `sound.mapping.AudioMapper` (楽器選択ロジック)
- `viewer.DualViewer` (色分け表示)

### ロールバック戦略
新機能用フラグ `--enable-mountain-instrument` (デフォルト OFF)。問題発生時はフラグを無効化して従来動作に戻せる。

### メモ
- クラスタリングはメッシュ更新頻度 (≤1 Hz) なので多少重くても許容範囲。
- `triangle_to_mountain` のリサイズはメッシュ更新ごとに行うが NumPy int32 配列なので数万規模でも軽量 (<200 KB)。
- 今後の拡張: 山属性を MIDI CC にマッピングしてモジュレーション演奏など。
