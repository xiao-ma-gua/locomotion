### 一、`myo_train_walk_manager.py` 脚本核心运行模式与命令

这个脚本设计了两种主要的工作流：**从头训练和加载微调**。

#### 1. 从头开始训练

这是最基础的模式。

你需要指定算法 (`--algo`)，算法在脚本里有

指定环境(`--env`)，环境在该[网址](https://myosuite.readthedocs.io/en/latest/suite.html)的最下方自行选择

指定cpu核数(`--num_cpu`)，核数根据自己硬件做决定

指定运行步数(`--total_timesteps`)

指定随机种子(`--seed`)

例如：以下是在终端运行`myo_train_walk_manager.py`

```sh
python myo_train_walk_manager.py \
    --algo TQC \
    --env myoLegWalk-v0 \
    --num_cpu 16 \
    --total_timesteps 20000000 \
    --seed 1024
```

#### 2. 模型微调

当你的基础模型收敛后，你想调整学习率继续压榨模型性能时，使用此模式。

**注意：** 你**必须**同时提供 `.zip` 模型文件和对应的 `.pkl` 归一化文件。

其中：

`--finetune`表示启用微调模式，仅作标志使用	
`--pretrained_model_path`表示预训练模型 .zip 的路径
`--pretrained_vec_path`预训练环境统计量 .pkl 的路径
`--ft_learning_rate`微调时的覆盖学习率
`--ft_ent_coef`微调时的覆盖熵系数

例如：

```sh
python myo_train_walk_manager.py \
    --algo TQC \
    --env myoLegWalk-v0 \
    --num_cpu 16 \
    --seed 512
    --total_timesteps 20000000
    --finetune \
    --pretrained_model_path ./policy/myowalk_tqc.zip \
    --pretrained_vec_path ./policy/myowalk_tqc.pkl \
    --ft_learning_rate 5e-5 \
    --ft_ent_coef "auto_0.01" \
```

### 二、 训练过程监控

你的脚本已经配置了 TensorBoard 日志，只要脚本开始运行并产生了数据，你就可以新建一个终端，在脚本同级目录下运行：

```sh
tensorboard --logdir ./logs/logs_comparison/tensorboard/ 
```

随后就会终端就会出现一个网址，点进去就能看到日志所对应的图像



