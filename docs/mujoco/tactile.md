# 力传感器测量接触力

## 1. 力传感器的类型与测量原理

MuJoCo 提供两类共 6 种力传感器，覆盖从简单碰撞检测到复杂多体动力学分析的全场景需求。传感器类型在[include/mujoco/mjmodel.h](https://github.com/OpenHUTB/mujoco/blob/main/include/mujoco/mjmodel.h)  中定义，核心参数包括测量维度、数据类型和噪声特性。

### 1.1 接触力传感器

接触力传感器(Contact Sensor)通过监测碰撞约束计算接触点的力和力矩，支持7种数据输出格式（力、力矩、位置等）。其核心原理是通过求解碰撞约束方程：

$$
F = J^T \lambda
$$

其中 \( J \) 为接触点雅可比矩阵，\( \lambda \)为约束力向量。在 [test/engine/testdata/sensor/contact.xml](https://github.com/OpenHUTB/mujoco/blob/main/test/engine/testdata/sensor/contact.xml) 中定义了典型配置：
```xml
<sensor>
  <contact name="all"/>                  <!-- 测量所有接触 -->
  <contact name="b1" body1="b1"/>        <!-- 仅测量b1相关接触 -->
  <contact name="g1" geom1="g1"/>        <!-- 仅测量g1相关接触 -->
  <contact name="net" reduce="netforce"/> <!-- 输出合力 -->
</sensor>
```


### 1.2 触觉传感器

触觉传感器（Tactile Sensor）基于**有符号距离场**（Signed Distance Field，SDF）实现面接触分布测量，特别适合柔性体交互场景。在 [model/tactile/tactile.xml](https://github.com/OpenHUTB/mujoco/blob/main/model/tactile/tactile.xml) 中，通过网格采样实现指尖触觉阵列：
```xml
<sensor>
  <tactile geom="finger" mesh="box"/>    <!-- 手指触觉阵列 -->
  <tactile geom="ball" mesh="sphere"/>   <!-- 球体触觉阵列 -->
</sensor>
```

!!! 笔记
    有符号距离场为每个点存储其到目标形状的最短距离，正负号表示点在形状内部（正）或外部（负）。

### 1.3 关键技术参数与配置

力传感器的测量精度取决于 3 个核心参数，这些参数在XML中通过sensor标签配置，在代码中通过mjModel结构体访问。

### 1.4 数据类型与维度

| 传感器类型      | 数据类型（mjtDataType）         | 输出维度 | 典型应用场景                      |
| ------------------------------------------------------- | ------------------------------------------------------- |------|-------------------------|
| touch    | mjDATATYPE_POSITIVE    | 1    | 碰撞检测       |
| force    | mjDATATYPE_REAL    | 3    | 力控制       |
| torque    | mjDATATYPE_REAL    | 3    | 力矩平衡       |
| tactile    | mjDATATYPE_REAL    | N×M    | 抓取姿态识别       |

### 1.5 噪声与滤波

传感器数据默认添加高斯噪声，通过`noise`参数配置标准差：
```xml
<sensor>
  <force name="gripper_force" site="gripper" noise="0.01"/>
</sensor>
```

在代码中可通过 [mj_sensorPos](https://github.com/OpenHUTB/mujoco/blob/727347813f08111a113587b3a2d92ee2ddeba043/include/mujoco/mujoco.h#L371) 等函数获取原始数据后进行滤波处理：
```cpp
mjtNum* force_data = d->sensordata + m->sensor_adr[force_sensor_id];
```

## 2. 实战案例：机器人抓取力控制

以两指抓取任务为例，展示从传感器配置到力反馈控制的完整流程。

### 2.1 模型配置
在XML模型中定义指尖力传感器和接触检测传感器：
```xml

<sensor>
  <!-- 指尖力传感器 -->
  <force name="lf_force" site="left_finger"/>
  <force name="rf_force" site="right_finger"/>
  
  <!-- 接触检测 -->
  <contact name="object_contact" geom1="object"/>
</sensor>
```

### 2.2 数据读取（Python）
通过 [mujoco Python API](https://mujoco.readthedocs.io/en/stable/python.html) 读取传感器数据：

```python
import mujoco
 
model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)
 
# 读取力传感器数据
lf_force = data.sensordata[model.sensor("lf_force").adr]
rf_force = data.sensordata[model.sensor("rf_force").adr]
 
# 读取接触状态
contact_state = data.sensordata[model.sensor("object_contact").adr]
```


### 2.3 力控制实现
基于力传感器反馈实现阻抗控制：
```python
desired_force = np.array([0, 0, -5])  # 期望抓取力
Kp = 100.0  # 比例增益
Kd = 5.0    # 阻尼增益
 
# 计算力误差
force_error = desired_force - lf_force
 
# 阻抗控制律
ctrl = Kp * force_error - Kd * data.qvel[:2]
data.ctrl[:] = ctrl
```

### 2.3 常见问题与解决方案

* 噪声抑制

    **问题**：接触力数据包含高频噪声，影响控制稳定性。

    **方案**：使用滑动平均滤波：

```python
# 滑动平均滤波
window_size = 5
force_buffer = np.zeros((window_size, 3))
force_buffer = np.roll(force_buffer, -1, axis=0)
force_buffer[-1] = raw_force
filtered_force = force_buffer.mean(axis=0)
```


* 接触点定位
    **问题**：需要精确获取接触点位置用于路径规划。

    **方案**：使用带位置输出的接触传感器：

```xml
<contact name="with_pos" data="force pos"/>
```

## 3. 高级应用与未来趋势
* **多传感器融合**：通过融合力传感器与视觉数据，实现更鲁棒的抓取控制。在 [model/humanoid/humanoid.xml](https://github.com/OpenHUTB/mujoco/blob/main/model/humanoid/humanoid.xml) 中，全身力传感器网络与IMU数据融合，实现动态平衡控制。

* **GPU加速计算**：利用 [MJX](https://github.com/OpenHUTB/mujoco/tree/main/mjx) 实现GPU并行力传感器计算，在 [training_apg.ipynb](https://github.com/OpenHUTB/mujoco/blob/main/mjx/training_apg.ipynb) 中，通过CUDA加速实现 thousand-DoF 机器人的实时力控仿真。



## 参考

* [从触觉感知到精准控制：MuJoCo力传感器测量接触力的核心技术](https://blog.csdn.net/gitblog_00967/article/details/151520776)