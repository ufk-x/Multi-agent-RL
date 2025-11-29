# 2D机械臂避障任务 - PPO强化学习

## 任务描述

这是一个使用PPO（Proximal Policy Optimization）算法训练的2D机械臂避障任务。

### 任务目标
- **机械臂配置**: 2连杆机械臂，基座固定在原点(0,0)，每个连杆长度为1
- **运动任务**: 从起点A (1.5, -0.5) 运动到目标点B (1.0, 1.5)
- **避障要求**: 避开位于(1.2, 0.8)处的半径为0.3的球形障碍物

### 环境特性

**状态空间** (9维):
- 关节角度 (θ1, θ2) - 2维
- 关节角速度 (ω1, ω2) - 2维
- 末端到目标的相对位置 (Δx, Δy) - 2维
- 末端到障碍物的相对位置和距离 (Δx_obs, Δy_obs, d_obs) - 3维

**动作空间** (2维):
- 两个关节的扭矩 (τ1, τ2)，范围: [-2.0, 2.0]

**奖励函数**:
- ✅ 到达目标 (距离 < 0.1): +100
- ❌ 碰撞障碍物: -50
- 📉 距离惩罚: -0.1 × distance_to_target
- 📉 动作惩罚: -0.01 × (τ1² + τ2²)
- 🎯 接近目标 (距离 < 0.5): +0.5

## 安装依赖

```bash
# 在conda环境vpsto中
conda activate vpsto
pip install torch numpy matplotlib
```

## 使用方法

### 1. 训练模型

```bash
cd /home/kai/0xPrjs/Multi-agent-RL/动手学强化学习/Hands-on-RL
python 2D_arm.py
```

训练完成后会生成：
- `training_results.png` - 训练曲线图
- `arm_animation.gif` - 机械臂运动动画
- `arm2d_ppo_model.pth` - 训练好的模型

### 2. 测试已训练模型

在Python中运行：
```python
from 2D_arm import test_arm
test_arm('arm2d_ppo_model.pth')
```

这会生成 `arm_test_animation.gif` 动画文件。

## 超参数配置

```python
actor_lr = 1e-4          # Actor网络学习率
critic_lr = 5e-4         # Critic网络学习率
num_episodes = 3000      # 训练episodes数
hidden_dim = 256         # 神经网络隐藏层维度
gamma = 0.99             # 折扣因子
lmbda = 0.95             # GAE参数
epochs = 10              # 每次更新的训练轮数
eps = 0.2                # PPO截断参数
```

## 关键技术实现

### 1. 正运动学
根据关节角度计算末端执行器位置：
```
x_end = l1*cos(θ1) + l2*cos(θ1+θ2)
y_end = l1*sin(θ1) + l2*sin(θ1+θ2)
```

### 2. 逆运动学
使用几何方法求解关节角度以到达目标位置。

### 3. 碰撞检测
检查以下部分与障碍物的距离：
- 末端执行器
- 中间关节
- 两个连杆线段

使用点到线段的距离计算。

### 4. PPO算法
- 使用高斯策略输出连续动作
- 通过截断机制限制策略更新幅度
- Actor-Critic架构同时学习策略和价值函数

## 文件结构

```
2D_arm.py              # 主程序文件
├── Arm2DEnv           # 2D机械臂环境类
├── PolicyNetContinuous # 连续动作策略网络
├── ValueNet           # 价值网络
├── PPOContinuous      # PPO算法实现
├── train_arm()        # 训练函数
├── test_arm()         # 测试函数
└── visualize_episode() # 可视化函数
```

## 预期结果

经过约2000-3000个episodes的训练后，智能体应该能够：
1. 成功避开障碍物
2. 平滑地从起点运动到目标点
3. 达到较高的成功率（>80%）

## 可视化说明

动画中的元素：
- 🟢 绿色圆圈：起点A
- ⭐ 蓝色星星：目标点B
- 🔴 红色圆圈：球形障碍物
- ⚫ 黑色线段：第一个连杆
- 🟣 紫色线段：第二个连杆
- 🔴 红色圆点：末端执行器
- 🔵 青色轨迹：末端执行器的运动轨迹

## 调试和优化

### 如果训练不收敛：
1. 减小学习率（actor_lr, critic_lr）
2. 增加隐藏层维度（hidden_dim）
3. 调整奖励函数权重
4. 增加训练episodes

### 如果频繁碰撞障碍物：
1. 增加碰撞惩罚 `reward_collision`
2. 减小碰撞边界 `collision_margin`
3. 在奖励函数中添加更多避障引导

### 如果运动不平滑：
1. 增加动作惩罚权重 `penalty_action`
2. 减小最大扭矩 `max_torque`
3. 增加时间步长 `dt`

## 参考

本实现参考了：
- OpenAI Spinning Up PPO文档
- 《动手学强化学习》第12章 PPO算法
- Gymnasium环境接口规范

## 作者

基于《动手学强化学习》教材中的PPO算法实现
