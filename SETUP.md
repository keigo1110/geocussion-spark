# ubuntuでのインストール手順（2025/06/22）
  ## Geocussionリポジトリをインストール
  git clone https://github.com/keigo1110/geocussion-spark.git
  cd geocussion-spark
  
 ## 仮想環境を作る
 sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
 python3 -m venv ./venv
 source venv/bin/activate
 pip install -r requirements.txt
 
 ## pyorbbecsdkリポジトリをインストール
 git clone https://github.com/keigo1110/pyorbbecsdk.git
 cd pyorbbecsdk
 pip3 install -r requirements.txt
 mkdir build
 cd build
 cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
 make -j4
 make install
 cd ..
 export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
 sudo bash ./scripts/install_udev_rules.sh
 sudo udevadm control --reload-rules && sudo udevadm trigger

## Geocussionテスト
 cd ..
 python3 demo_collision_detection.py
 これで動く

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