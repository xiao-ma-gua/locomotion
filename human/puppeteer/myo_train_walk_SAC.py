import gymnasium as gym
import os
import myosuite
from stable_baselines3 import SAC  # <--- 改为导入 SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# 1. 配置
LOG_DIR = "./logs/myo_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

# 2. 创建环境
# SAC 是 Off-policy 算法，通常不需要多个并行环境，单线程调试更容易
env = Monitor(gym.make('myoLegWalk-v0'))
env = DummyVecEnv([lambda: env])

# 3. 定义 SAC 模型
# policy_kwargs: 增加神经网络的宽度，从默认的[64,64]改为[256,256]，肌骨骼模型需要更大的脑容量
policy_kwargs = dict(net_arch=[256, 256])

# # 模型保存路径
# MODEL_PATH = "./policy/my_walking_policy_sac.zip"
#
# if os.path.exists(MODEL_PATH):
#     print(f"发现已保存的模型 {MODEL_PATH}，正在加载并继续训练...")
#     # 1. 加载旧模型，并绑定当前环境
#     model = SAC.load(MODEL_PATH, env=env)
#
#     # 2. 如果你想调整学习率等参数，可以在这里修改，例如：
#     # model.learning_rate = 1e-4
# else:
# print("未发现模型，开始从头训练...")

# 定义新模型 (你的原代码)
model = SAC("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard/sac_walk_tensorboard/",
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            ent_coef='auto',
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs)


# 4. 回调函数 (保持不变)
eval_env = Monitor(gym.make('myoLegWalk-v0'))
eval_callback = EvalCallback(eval_env,
                             best_model_save_path='best_model/best_model_sac/',
                             log_path='./results_sac/',
                             eval_freq=5000,
                             deterministic=True,
                             render=False)

print("开始 SAC 训练...")
model.learn(total_timesteps=10000000, callback=eval_callback) # SAC 通常只需 PPO 1/5 的步数
# model.learn(total_timesteps=10000000, callback=eval_callback, reset_num_timesteps=False) # SAC 通常只需 PPO 1/5 的步数

model.save("my_walking_policy_sac")
print("SAC 训练完成")