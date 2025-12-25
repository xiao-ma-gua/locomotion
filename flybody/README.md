## 参考示例：果蝇飞行

### 配置
```shell
# 创建虚拟Python环境
conda create -n locomotion -c conda-forge python=3.10 pip ipython cudatoolkit=11.8.0
# 激活虚拟环境
conda activate locomotion
# 安装依赖
pip install -r requirements.txt
# deactivate virtual environment
# conda deactivate locomotion
# remove virtual environment
# conda remove -n locomotion --all
```

果蝇身体模型位于 [资产目录](https://github.com/OpenHUTB/locomotion/tree/master/flybody/fruitfly/assets) 中。为了使其可视化，您可以拖放 `fruitfly.xml`或 `floes.xml` 到 [MuJoCo](https://github.com/google-deepmind/mujoco/releases) 的`simulate` 查看器。

通过 [fly.demo.py](fly_demo.py) 与果蝇进行交互。

开始使用 `flybody` 的最快方法是看 [教程](docs) 。

另外，[果蝇环境示例的脚本](docs/fly_env_example.py) 显示了飞行、步行和视觉引导的飞行强化学习任务环境的示例。

要训练果蝇，请尝试 [分布式RL训练脚本](https://github.com/OpenHUTB/locomotion/blob/master/flybody/train_dmpo_ray.py) ，它使用 [Ray](https://github.com/ray-project/ray) 并行化 [DMPO](https://github.com/google-deepmind/acme/tree/master/acme/agents/tf/dmpo) 代理训练。



### 安装

请按照以下步骤安装`flybody`：

1. 从远程仓库安装：
   ```bash
   pip install git+https://github.com/TuragaLab/flybody.git
   pip install "flybody[tf] @ git+https://github.com/TuragaLab/flybody.git"
   pip install "flybody[ray] @ git+https://github.com/TuragaLab/flybody.git"
   ```

注意：dm-reverb只支持Linux。

2. 执行测试
```shell
ruff check flybody/
pytest tests/
pytest tests/test-tf.py
```