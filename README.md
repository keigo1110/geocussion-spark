# Geocussion-SP 入力フェーズ実装

Orbbecカメラからの深度・RGB画像取得と点群変換のリアルタイム処理パイプラインの入力フェーズ実装です。

## 機能

- **デュアルウィンドウ表示**: RGB画像（深度カラーマップ）と3D点群を同時表示
- **高性能フィルタリング**: median/bilateral/temporal フィルタによるノイズ除去
- **リアルタイム処理**: 5ms以内の目標処理時間でフレーム処理
- **モジュラー設計**: 再利用可能なコンポーネント設計

## システム要件

- Python 3.10以上
- Orbbec 深度カメラ（Femto Bolt 等）
- GPU推奨（Open3D 3D表示用）

## インストール

### 1. 依存関係のインストール

```bash
# 基本パッケージ
pip install numpy opencv-python open3d

# Orbbec SDK（カメラ接続用）
pip install pyorbbecsdk
```

### 2. プロジェクトのクローン

```bash
git clone <repository-url>
cd geocussion-sp
```

## 使用方法

### デュアルウィンドウビューワーの起動

```bash
# 基本実行
python demo_dual_viewer.py

# フィルタ無効で実行
python demo_dual_viewer.py --no-filter

# 特定のフィルタタイプで実行
python demo_dual_viewer.py --filter-type median

# 高頻度点群更新
python demo_dual_viewer.py --update-interval 1

# カスタムウィンドウサイズ
python demo_dual_viewer.py --window-width 800 --window-height 600
```

### 操作方法

#### RGBウィンドウ
- `Q` または `ESC`: アプリケーション終了
- `F`: 深度フィルタのON/OFF切り替え
- `R`: フィルタ履歴リセット

#### 3D点群ビューワー
- **マウス**: 視点の回転/パン/ズーム
- `R`: 視点リセット
- `+`/`-`: 点サイズ変更
- `S`: 現在の点群を保存

### テスト実行

```bash
# 単体テスト（カメラ不要）
python demo_dual_viewer.py --test

# または直接テスト実行
python tests/input_test.py
```

## アーキテクチャ

### ディレクトリ構成

```
geocussion-sp/
├── src/
│   ├── input/              # 入力フェーズ
│   │   ├── stream.py       # カメラ抽象化
│   │   ├── pointcloud.py   # 点群変換
│   │   └── depth_filter.py # ノイズフィルタ
│   └── debug/              # デバッグ・可視化
│       └── dual_viewer.py  # デュアルウィンドウ表示
├── tests/                  # 単体テスト
├── demo_dual_viewer.py     # メインデモ
└── README.md
```

### 主要コンポーネント

#### 1. OrbbecCamera (stream.py)
- Orbbecカメラの抽象化クラス
- 深度・RGB フレームの取得
- カメラ内部パラメータ管理

#### 2. PointCloudConverter (pointcloud.py)
- 深度フレーム → 3D点群変換
- numpy最適化、ゼロコピー転送対応
- メッシュグリッド事前計算によるパフォーマンス向上

#### 3. DepthFilter (depth_filter.py)
- Median フィルタ: 塩胡椒ノイズ除去
- Bilateral フィルタ: エッジ保持平滑化
- Temporal フィルタ: 時間軸での残像低減
- Combined フィルタ: 全フィルタの最適組み合わせ

#### 4. DualViewer (dual_viewer.py)
- RGB + 3D点群の同時表示
- リアルタイムパフォーマンス計測
- インタラクティブ操作

## パフォーマンス

目標処理時間（40msフレーム内）：
- **入力取得**: ~5ms
- **フィルタ処理**: ~2ms
- **点群変換**: ~3ms
- **表示更新**: 残り時間

実際のパフォーマンスは画面下部に表示されます。

## トラブルシューティング

### カメラが認識されない
```bash
# USB権限の設定（Linux）
sudo usermod -a -G plugdev $USER
sudo cp pyorbbecsdk/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# 再ログインまたは再起動
```

### Open3D表示エラー
```bash
# 仮想環境での実行を推奨
# GUI表示にはX11転送が必要（SSH接続時）
ssh -X username@hostname
```

### インポートエラー
```bash
# パスが正しく設定されているか確認
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python demo_dual_viewer.py
```

## 開発・拡張

### 新しいフィルタの追加

```python
# src/input/depth_filter.py にて
class MyCustomFilter(DepthFilter):
    def _apply_custom_filter(self, depth_image, valid_mask):
        # カスタムフィルタ実装
        return filtered_image
```

### テストの追加

```python
# tests/ ディレクトリに新しいテストファイルを作成
class TestMyFeature(unittest.TestCase):
    def test_my_function(self):
        # テスト実装
        pass
```

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。

---

**実装完了項目 ✅**

- [x] Orbbecカメラ抽象化クラス
- [x] 点群変換処理の最適化
- [x] 深度ノイズフィルタ（median/bilateral/temporal）
- [x] デュアルウィンドウ表示システム
- [x] パフォーマンス計測
- [x] 単体テスト
- [x] デモアプリケーション

**次の実装フェーズ**
- [ ] 手検出フェーズ（MediaPipe）
- [ ] 地形メッシュ生成フェーズ
- [ ] 衝突検出フェーズ
- [ ] 音響生成フェーズ 