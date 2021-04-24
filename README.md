TensorFlow op for TFPWA
========================

Build
-----

link to cuda
```
mkdir -p third_party/gpus
ln -s /usr/local/cuda third_party/gpus/
```

build so
```
mkdir build
cd build
cmake ..
make
```

install
```
pip install -e . --no-deps
```
