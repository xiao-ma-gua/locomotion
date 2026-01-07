## 基于[引擎](https://github.com/OpenHUTB/engine)的人

环境安装
```shell
pip install hutb
pip install stable_baselines3==2.0.0
```

* 如果运行 Carla_Pedestrian_PPO.py 报错
```text
INTEL oneMKL ERROR: 找不到指定的模块。 mkl_intel_thread.2.dll.
```
> 解决办法：`pip install mkl`


* 如果遇到 numpy 的问题
```text
AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?
ImportError: numpy._core.multiarray failed to import
```
而安装对应版本无效：`pip install numpy==1.26.4`

> 解决：需要手动删除虚拟环境下的 numpy 文件夹，然后再重新安装。
