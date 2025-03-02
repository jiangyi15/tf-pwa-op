TensorFlow op for TFPWA
========================

Build
-----

0. set up enviromant

```
conda create -n tf_pwa_op
conda activate tf_pwa_op
```

1. install build dependences

```
conda install tensorflow cuda-minimal-build
```

2. build so file

```
mkdir build
cd build
cmake ..
make
```

3. install

```
pip install -e . --no-deps
```

Acknowledge
-----------

This work starts at OpenACC GPU Hackathon China 2021@CCNU, thanks the guidance of the group.
