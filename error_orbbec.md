# ubuntuでのインストール手順（2025/06/20）
 ## Geocussionリポジトリをインストール
 git clone https://github.com/keigo1110/geocussion-spark.git
 cd geocussion-spark
 
## 仮想環境を作る
sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
python3 -m venv ./venv
source venv/bin/activate

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
pip install numpy opencv-python open3d

## pyorbbecsdkテスト
python3 examples/point_cloud_realtime_viewer.py

## Geocussionテスト
cd ..
python3 demo_collision_detection.py