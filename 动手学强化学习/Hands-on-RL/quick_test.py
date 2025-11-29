"""
快速训练测试 - 运行50个episodes验证所有功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from importlib import reload
import importlib

# 导入模块
arm_module = importlib.import_module('2D_arm')
import rl_utils

print("=" * 60)
print("快速训练测试 (50 episodes)")
print("=" * 60)

device = torch.device("cpu")

# 创建环境和智能体
env = arm_module.Arm2DEnv()
agent = arm_module.PPOContinuous(
    state_dim=env.state_dim,
    hidden_dim=128,
    action_dim=env.action_dim,
    actor_lr=1e-4,
    critic_lr=5e-4,
    lmbda=0.95,
    epochs=5,
    eps=0.2,
    gamma=0.99,
    device=device
)

print(f"\n开始训练...")
num_episodes = 50
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

print(f"\n训练完成!")
print(f"平均回报: {np.mean(return_list):.2f}")
print(f"最高回报: {np.max(return_list):.2f}")

# 测试可视化（只运行一个episode，不保存动画）
print(f"\n测试可视化功能...")
state, _ = env.reset()
done = False
steps = 0
while not done and steps < 100:
    state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(device)
    mu, _ = agent.actor(state_tensor)
    action = mu.detach().cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    steps += 1

print(f"可视化测试完成! 运行了 {steps} 步")
print(f"\n✅ 所有功能正常！")
