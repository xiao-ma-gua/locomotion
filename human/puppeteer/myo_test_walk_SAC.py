import gymnasium as gym
import myosuite
import os
import time
import numpy as np
import imageio
from stable_baselines3 import SAC

# 1. 路径配置
# MODEL_PATH = "./policy/my_walking_policy_sac.zip"
MODEL_PATH = "./best_model/best_model_sac/best_model.zip"
VIDEO_FOLDER = "./videos/myowalk_sac_videos/"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
current_time = time.strftime("%Y%m%d-%H%M%S")
video_path = os.path.join(VIDEO_FOLDER, f"walk_test_{current_time}.mp4")

# 2. 创建环境
# 使用 rgb_array 模式初始化，尽管 RecordVideo 可能失效，但它能确保环境准备好图像渲染
env = gym.make('myoLegWalk-v0', render_mode='rgb_array')

# 3. 加载模型
if os.path.exists(MODEL_PATH):
    print(f"正在加载模型: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)
else:
    print(f"错误: 找不到模型文件 {MODEL_PATH}")
    exit()

# 4. 运行测试并手动捕获帧
print(f"开始测试并记录视频...")
obs, info = env.reset()
frames = []
total_reward = 0
done = False
truncated = False

# 限制录制步数
max_steps = 4000
step_count = 0

while not (done or truncated) and step_count < max_steps:
    # SAC 推理
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # --- 修复后的渲染逻辑 ---
    try:
        # 尝试使用 Gymnasium 标准的 render() 方法
        frame = env.render()
    except Exception:
        # 如果标准 render() 失败，则使用底层离屏渲染
        # 注意：MyoSuite 较新版本使用 renderer.render_offscreen
        sim = env.unwrapped.sim
        frame = sim.renderer.render_offscreen(
            width=640,
            height=480,
            camera_id='side_view'
        )

    if frame is not None:
        # 1. 首先确保是正确的数组格式
        frame = np.asarray(frame)
        # 2. 旋转 180 度修复倒立问题
        # 如果旋转后方向仍有偏差，可尝试 np.flipud(frame) 或 np.fliplr(frame)
        # fixed_frame = np.rot90(frame, k=2)
        frames.append(frame)

    total_reward += reward
    step_count += 1

# 5. 保存视频
if len(frames) > 0:
    print(f"正在保存视频到: {video_path}")
    # 确保 fps 与仿真频率匹配（通常 MyoSuite 步长较大，30-50fps 比较自然）
    imageio.mimsave(video_path, frames, fps=30)
    print(f"视频录制成功！共 {len(frames)} 帧。")
else:
    print("警告：未捕获到任何画面帧，请检查环境配置。")

print(f"测试结束！总步数: {step_count}, 总奖励: {total_reward:.2f}")
env.close()