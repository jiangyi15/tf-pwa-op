TensorFlow op for TFPWA
========================


Build when install
------------------


Build in local
--------------

0. set up enviromant

```
conda create -n tf_pwa_op
conda activate tf_pwa_op
```

1. install build dependences

```
conda install tensorflow cuda-minimal-build
conda install python-build py-build-cmake
```

3. install

```
git clone https://github.com/jiangyi15/tf-pwa-op && cd tf-pwa-op
pip install -e . --no-deps --no-build-isolation
```

you can also use `git+https://github.com/jiangyi15/tf-pwa-op` to replase the path `.`.

Acknowledge
-----------

This work starts at OpenACC GPU Hackathon China 2021@CCNU, thanks the help of the group.
