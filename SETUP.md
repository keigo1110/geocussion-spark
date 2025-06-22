# Geocussion-Spark 開発環境セットアップガイド

## 📋 概要

このガイドでは、Geocussion-Spark Hand-Terrain Collision Detection Systemの開発環境を構築する手順を説明します。

## 🔧 システム要件

### 必須要件
- **Python**: 3.8 以上（推奨: 3.11）
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **RAM**: 8GB以上（推奨: 16GB）
- **GPU**: CUDA対応GPU（オプション、MediaPipe高速化用）

### ハードウェア
- **Orbbec Astra+ カメラ**（深度センサー付き）
- **オーディオ出力デバイス**（スピーカー/ヘッドフォン）

## 🚀 クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-org/geocussion-spark.git
cd geocussion-spark
```

### 2. 仮想環境の作成
```bash
# Python仮想環境作成
python -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール
```bash
# 基本パッケージのインストール
pip install -r requirements.txt

# OrbbecSDKのインストール
pip install ./pyorbbecsdk/install/lib/
```

### 4. システム依存関係のインストール（Linux）
```bash
# オーディオライブラリ
sudo apt-get install libasound2-dev portaudio19-dev
sudo apt-get install libportaudio2 libportaudiocpp0

# OpenCVサポート
sudo apt-get install libopencv-dev python3-opencv

# Open3Dサポート
sudo apt-get install libegl1-mesa-dev
```

### 5. 動作確認
```bash
# テスト実行
python demo_collision_detection.py --test

# デモ実行（カメラ必要）
PYTHONPATH=$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib/ python demo_collision_detection.py
```

## 📦 詳細セットアップ手順

### Python環境の設定

#### 推奨: pyenvでのPythonバージョン管理
```bash
# pyenvのインストール（Linux/Mac）
curl https://pyenv.run | bash

# Python 3.11のインストール
pyenv install 3.11.8
pyenv local 3.11.8
```

#### 仮想環境の詳細設定
```bash
# 仮想環境作成（詳細オプション付き）
python -m venv venv --system-site-packages

# pipのアップグレード
pip install --upgrade pip setuptools wheel
```

### 依存関係の詳細インストール

#### 1. 科学計算ライブラリ
```bash
pip install numpy>=1.26.0 scipy>=1.15.0
```

#### 2. コンピュータビジョン
```bash
pip install opencv-python>=4.10.0 opencv-contrib-python>=4.10.0
```

#### 3. 3D処理とビジュアライゼーション
```bash
pip install open3d>=0.19.0
```

#### 4. 手検出（MediaPipe）
```bash
# CPU版（標準）
pip install mediapipe>=0.10.0

# GPU版（CUDA環境）
pip install mediapipe-gpu>=0.10.0
```

#### 5. オーディオ合成
```bash
pip install pyo>=1.0.0
```

### OrbbecSDKの詳細設定

#### 1. 環境変数の設定
```bash
# ~/.bashrcまたは~/.zshrcに追加
export PYTHONPATH=$PYTHONPATH:/path/to/geocussion-spark/pyorbbecsdk/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/geocussion-spark/pyorbbecsdk/install/lib
```

#### 2. デバイス権限設定（Linux）
```bash
# udevルールの作成
sudo nano /etc/udev/rules.d/99-orbbec.rules

# 以下の内容を追加
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"

# udevの再読み込み
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 開発環境の設定

#### VSCode設定（推奨）
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.analysis.autoImportCompletions": true,
    "files.associations": {
        "*.py": "python"
    }
}
```

#### Git設定
```bash
# .gitignoreの確認
cat .gitignore

# 開発ブランチの作成
git checkout -b feature/your-feature-name
```

## 🧪 テストとデバッグ

### 全体テストの実行
```bash
# 全フェーズのテスト
python -m pytest tests/ -v

# 特定フェーズのテスト
python demo_collision_detection.py --test
python demo_hand_detection.py --test
python demo_dual_viewer.py --test
```

### パフォーマンステスト
```bash
# 衝突検出性能テスト
python tests/collision_performance_test.py

# 音響システム安定性テスト
python -c "
from tests.sound_test import TestSoundIntegration
import unittest
suite = unittest.TestLoader().loadTestsFromTestCase(TestSoundIntegration)
unittest.TextTestRunner(verbosity=2).run(suite)
"
```

### デバッグモード実行
```bash
# 詳細ログ付きで実行
PYTHONPATH=$PYTHONPATH:$(pwd)/pyorbbecsdk/install/lib/ python demo_collision_detection.py --debug

# 低解像度モードでテスト
python demo_collision_detection.py --window-width 640 --window-height 480
```

## 🔧 トラブルシューティング

### よくある問題と解決策

#### 1. カメラが検出されない
```bash
# デバイス確認
lsusb | grep -i orbbec

# 権限確認
ls -la /dev/video*

# ドライバー再インストール
sudo rmmod uvcvideo
sudo modprobe uvcvideo
```

#### 2. オーディオが出力されない
```bash
# ALSA設定確認
aplay -l

# PulseAudio再起動
pulseaudio -k
pulseaudio --start

# オーディオテスト
python -c "
from src.sound.synth import create_audio_synthesizer
synth = create_audio_synthesizer()
print('Audio engine started:', synth.start_engine())
"
```

#### 3. MediaPipeエラー
```bash
# GPU版の問題
pip uninstall mediapipe-gpu
pip install mediapipe

# バージョン確認
python -c "import mediapipe as mp; print(mp.__version__)"
```

#### 4. Open3Dビジュアライゼーションエラー
```bash
# ディスプレイ設定（リモート環境）
export DISPLAY=:0.0

# OpenGLサポート確認
glxinfo | grep OpenGL
```

## 📚 開発リソース

### プロジェクト構造
```
geocussion-spark/
├── src/
│   ├── input/          # カメラ入力・点群変換
│   ├── detection/      # 手検出・トラッキング
│   ├── mesh/          # 地形メッシュ生成
│   ├── collision/     # 衝突検出
│   ├── sound/         # 音響合成
│   └── debug/         # デバッグツール
├── tests/             # 単体テスト
├── demo_*.py          # デモスクリプト
├── requirements.txt   # Python依存関係
└── SETUP.md          # このファイル
```

### 重要な設定ファイル
- `requirements.txt`: Python依存関係
- `inst/next.yaml`: 開発進捗記録
- `technologystack.md`: 技術スタック詳細
- `README.md`: プロジェクト概要

### パフォーマンス指標
- **フレームレート**: 3.0-3.7 FPS（目標）
- **音響レイテンシー**: 5.8ms以下
- **手検出レイテンシー**: 20-30ms以下
- **衝突検出レイテンシー**: 1ms以下

## 🤝 コントリビューション

### 開発ワークフロー
1. Issueの確認・作成
2. フィーチャーブランチの作成
3. 開発とテスト
4. プルリクエストの作成
5. コードレビュー
6. マージ

### コーディング規約
- **Python**: PEP 8準拠
- **関数名**: snake_case
- **クラス名**: PascalCase
- **定数**: UPPER_CASE
- **ドキュメント**: Google docstring形式

## 📞 サポート

問題や質問がある場合：
1. まず`SETUP.md`のトラブルシューティングセクションを確認
2. `tests/`でテストを実行して問題を特定
3. GitHubのIssueを作成して詳細を報告

---

**Happy Coding! 🎯** 