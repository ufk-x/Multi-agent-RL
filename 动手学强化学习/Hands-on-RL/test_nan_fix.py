"""
快速测试NaN修复 - 只运行30个episodes
"""
import torch
import numpy as np
from tqdm import tqdm
import sys
import importlib.util

# 动态导入2D_arm模块
spec = importlib.util.spec_from_file_location("arm_module", "2D_arm.py")
arm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm_module)

Arm2DEnv = arm_module.Arm2DEnv
PPOContinuous = arm_module.PPOContinuous
train_on_policy_agent = arm_module.train_on_policy_agent

def quick_test():
    """运行30个episodes的快速测试"""
    print("=" * 60)
    print("NaN修复验证测试 - 30 episodes")
    print("=" * 60)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = Arm2DEnv()
    
    # 创建PPO智能体
    agent = PPOContinuous(
        state_dim=env.state_dim,
        hidden_dim=256,
        action_dim=env.action_dim,
        actor_lr=1e-4,
        critic_lr=5e-4,
        lmbda=0.95,
        epochs=10,
        eps=0.2,
        gamma=0.99,
        device=device
    )
    
    print("\n开始训练30个episodes...")
    
    # 运行30个episodes
    return_list = []
    for episode in tqdm(range(30), desc="训练进度"):
        episode_return = 0
        transition_dict = {
            'states': [], 'actions': [], 
            'next_states': [], 'rewards': [], 'dones': []
        }
        
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward
            step_count += 1
            
            # 防止无限循环
            if step_count > 1000:
                break
        
        return_list.append(episode_return)
        agent.update(transition_dict)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"平均回报: {np.mean(return_list):.2f}")
    print(f"最高回报: {np.max(return_list):.2f}")
    print(f"最低回报: {np.min(return_list):.2f}")
    
    # 检查是否有NaN
    if not np.isnan(return_list).any():
        print("\n✅ 没有检测到NaN值！")
        return True
    else:
        print("\n❌ 仍然存在NaN值！")
        return False

if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
