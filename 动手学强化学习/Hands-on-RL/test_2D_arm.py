"""
2D机械臂环境快速测试脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

# 导入2D_arm模块
from importlib import reload
import importlib

def test_environment():
    """测试环境基本功能"""
    print("=" * 60)
    print("测试2D机械臂环境")
    print("=" * 60)
    
    # 动态导入模块
    arm_module = importlib.import_module('2D_arm')
    
    # 创建环境
    env = arm_module.Arm2DEnv()
    
    print(f"\n1. 环境创建成功")
    print(f"   状态维度: {env.state_dim}")
    print(f"   动作维度: {env.action_dim}")
    
    # 测试reset
    state, info = env.reset()
    print(f"\n2. 重置环境")
    print(f"   初始状态形状: {state.shape}")
    print(f"   初始关节角度: {env.joint_angles}")
    
    # 测试正运动学
    base, joint1, end = env.forward_kinematics(env.joint_angles)
    print(f"\n3. 正运动学")
    print(f"   基座位置: {base}")
    print(f"   关节1位置: {joint1}")
    print(f"   末端位置: {end}")
    
    # 测试step
    print(f"\n4. 执行随机动作")
    for i in range(5):
        action = np.random.uniform(-1, 1, 2)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"   步骤{i+1}: reward={reward:.3f}, done={done}")
        if done:
            break
    
    print(f"\n5. 环境测试通过! ✓")
    return True

def test_networks():
    """测试神经网络"""
    print("\n" + "=" * 60)
    print("测试PPO网络")
    print("=" * 60)
    
    arm_module = importlib.import_module('2D_arm')
    
    device = torch.device("cpu")
    state_dim = 9
    action_dim = 2
    hidden_dim = 128
    
    # 测试策略网络
    policy_net = arm_module.PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
    test_state = torch.randn(1, state_dim)
    mu, std = policy_net(test_state)
    
    print(f"\n1. 策略网络")
    print(f"   输入维度: {state_dim}")
    print(f"   输出均值形状: {mu.shape}")
    print(f"   输出标准差形状: {std.shape}")
    
    # 测试价值网络
    value_net = arm_module.ValueNet(state_dim, hidden_dim).to(device)
    value = value_net(test_state)
    
    print(f"\n2. 价值网络")
    print(f"   输出形状: {value.shape}")
    
    print(f"\n3. 网络测试通过! ✓")
    return True

def quick_train_test():
    """快速训练测试（只训练几个episodes）"""
    print("\n" + "=" * 60)
    print("快速训练测试（10个episodes）")
    print("=" * 60)
    
    arm_module = importlib.import_module('2D_arm')
    import rl_utils
    
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
    
    print(f"\n开始快速训练...")
    num_episodes = 10
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    
    print(f"\n训练完成!")
    print(f"平均回报: {np.mean(return_list):.2f}")
    print(f"最高回报: {np.max(return_list):.2f}")
    print(f"最低回报: {np.min(return_list):.2f}")
    
    print(f"\n快速训练测试通过! ✓")
    return True

if __name__ == "__main__":
    try:
        # 运行所有测试
        print("\n" + "🚀 " * 30)
        print("开始2D机械臂环境测试")
        print("🚀 " * 30 + "\n")
        
        test_environment()
        test_networks()
        quick_train_test()
        
        print("\n" + "✅ " * 30)
        print("所有测试通过!")
        print("✅ " * 30 + "\n")
        
        print("现在可以运行完整训练:")
        print("  python 2D_arm.py")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
