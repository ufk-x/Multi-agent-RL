
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# 设置随机种子以确保可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 环境参数
L1 = 1.0  # 第一臂长
L2 = 1.0  # 第二臂长
DT = 0.1  # 时间步长
GOAL_POS = np.array([0.0, 1.5])  # 目标位置 (x, y)
START_ANGLES = np.array([0.0, 0.0]) # 起始角度
OBSTACLE_POS = np.array([1.0, 1.0]) # 障碍物位置
OBSTACLE_RADIUS = 0.15 # 障碍物半径
MAX_STEPS = 200 # 最大步数
ACTION_SCALE = 1.0 # 动作缩放（最大角速度）

class ArmEnv:
    def __init__(self):
        self.state_dim = 8 # theta1, theta2, x_obs, y_obs, x_ee, y_ee, x_target, y_target
        self.action_dim = 2 # d_theta1, d_theta2
        self.reset()

    def reset(self):
        self.angles = START_ANGLES.copy()
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        x, y = self._fk(self.angles)
        # 归一化状态通常有助于训练，这里简单处理
        # return np.concatenate([self.angles, OBSTACLE_POS, [x, y], GOAL_POS]) 
        return np.concatenate([self.angles, [x, y],  OBSTACLE_POS, GOAL_POS]) 

    def _fk(self, angles):
        theta1, theta2 = angles
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        return np.array([x, y])

    def _check_collision(self, p1, p2, circle_center, radius):
        # 检测线段 p1-p2 是否与圆相交
        p1 = np.array(p1)
        p2 = np.array(p2)
        c = np.array(circle_center)
        
        d = p2 - p1
        f = p1 - c
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c_val = np.dot(f, f) - radius**2
        
        discriminant = b**2 - 4*a*c_val
        
        if discriminant < 0:
            return False
        else:
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)
            
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
            # 检查端点是否在圆内
            if np.linalg.norm(p1 - c) < radius or np.linalg.norm(p2 - c) < radius:
                return True
            return False

    def step(self, action):
        self.steps += 1
        # Action 是角速度/增量
        action = np.clip(action, -1, 1) * ACTION_SCALE
        self.angles += action * DT
        
        # 限制角度范围 (-pi, pi)
        self.angles = np.arctan2(np.sin(self.angles), np.cos(self.angles))
        
        # 正运动学
        p0 = np.array([0, 0])
        p1 = np.array([L1 * np.cos(self.angles[0]), L1 * np.sin(self.angles[0])])
        ee_pos = self._fk(self.angles)
        
        # 距离目标的距离
        dist_to_goal = np.linalg.norm(ee_pos - GOAL_POS)
        
        # 碰撞检测
        collision = False
        # 检测第一臂
        if self._check_collision(p0, p1, OBSTACLE_POS, OBSTACLE_RADIUS):
            collision = True
        # 检测第二臂
        if self._check_collision(p1, ee_pos, OBSTACLE_POS, OBSTACLE_RADIUS):
            collision = True
            
        # 奖励函数 (稀疏奖励为主)
        reward = 0
        done = False
        info = {}
        
        # 1. 到达目标
        if dist_to_goal < 0.07:
            reward += 20000
            done = True
            info['result'] = 'success'
        # 2. 碰撞
        elif collision:
            reward -= 50
            done = True
            info['result'] = 'collision'
        # 3. 超时
        elif self.steps >= MAX_STEPS:
            done = True
            reward -= 10
            info['result'] = 'timeout'
        else:
            # 4. 步数惩罚 (鼓励快速)
            reward -= 0.1
            # 5. 距离引导 (非常稀疏的引导，或者弱引导)
            # 为了让它能训练出来，加一个弱的距离势能奖励
            reward -= dist_to_goal * 0.5
            pass

        return self._get_state(), reward, done, info

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        mu = 2.0 * torch.tanh(self.fc_mu(x)) # 输出范围 -2 到 2 (再由环境缩放)
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mu, std

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.actor = PolicyNet(state_dim, action_dim).to(DEVICE)
        self.critic = ValueNet(state_dim).to(DEVICE)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, std = self.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=1)
        return action.cpu().numpy().flatten(), action_log_prob.item()

    def update(self, memory):
        # 转换数据
        states = torch.FloatTensor(np.array(memory['states'])).to(DEVICE)
        actions = torch.FloatTensor(np.array(memory['actions'])).to(DEVICE)
        log_probs = torch.FloatTensor(np.array(memory['log_probs'])).to(DEVICE)
        rewards = torch.FloatTensor(np.array(memory['rewards'])).to(DEVICE)
        dones = torch.FloatTensor(np.array(memory['dones'])).to(DEVICE)
        
        # 计算 Monte Carlo 估计的 returns
        returns = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(DEVICE)
        # 归一化 returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # 优化 K 次
        for _ in range(self.K_epochs):
            # 评估旧动作
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            
            state_values = self.critic(states).squeeze()
            
            # 比例
            ratios = torch.exp(new_log_probs - log_probs)
            
            # 优势函数
            advantages = returns - state_values.detach()
            
            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def train():
    # 确保训练开始时种子已设置
    set_seed(42)
    
    env = ArmEnv()
    ppo = PPO(env.state_dim, env.action_dim)
    
    MAX_EPISODES = 1000
    UPDATE_TIMESTEP = 2000
    
    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
    timestep = 0
    
    rewards_history = []
    
    print("Start Training...")
    try:
        for i_episode in range(1, MAX_EPISODES + 1):
            state = env.reset()
            ep_reward = 0
            while True:
                timestep += 1
                action, log_prob = ppo.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                memory['states'].append(state)
                memory['actions'].append(action)
                memory['log_probs'].append(log_prob)
                memory['rewards'].append(reward)
                memory['dones'].append(done)
                
                state = next_state
                ep_reward += reward
                
                if timestep % UPDATE_TIMESTEP == 0:
                    ppo.update(memory)
                    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
                
                if done:
                    break
            
            rewards_history.append(ep_reward)
            
            if i_episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode {i_episode}, Avg Reward: {avg_reward:.2f}, Last Result: {info.get('result', 'unknown')}")
                
                # 如果效果不错，保存模型
                if avg_reward > 50: # 简单阈值
                    torch.save(ppo.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model', 'ppo_actor.pth'))
    except KeyboardInterrupt:
        print("Training interrupted.")
                
    print("Training finished.")
    torch.save(ppo.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model', 'ppo_actor_final.pth'))
    return ppo, rewards_history

def visualize(ppo, save_name='arm_trajectory.mp4'):
    env = ArmEnv()
    state = env.reset()
    done = False
    
    # 准备绘图 - 使用能被16整除的尺寸避免ffmpeg警告
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    
    frames = []
    trajectory = []  # 记录末端轨迹
    
    # 计算起点末端位置
    start_ee = env._fk(START_ANGLES)
    
    total_reward = 0
    step_count = 0
    info = {}
    reward = 0  # 初始化当前奖励
    
    print("Generating visualization...")
    while not done:
        # 记录帧
        ax.clear()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 画障碍物
        circle = patches.Circle((OBSTACLE_POS[0], OBSTACLE_POS[1]), OBSTACLE_RADIUS, color='r', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
        
        # 画起点 (圆圈)
        start_circle = patches.Circle((start_ee[0], start_ee[1]), 0.08, color='blue', alpha=0.7, label='Start')
        ax.add_patch(start_circle)
        
        # 画目标 (星号)
        ax.plot(GOAL_POS[0], GOAL_POS[1], 'g*', markersize=20, label='Goal')
        
        # 画机械臂
        theta1, theta2 = env.angles
        x0, y0 = 0, 0
        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)
        x2 = x1 + L2 * np.cos(theta1 + theta2)
        y2 = y1 + L2 * np.sin(theta1 + theta2)
        
        # 记录轨迹
        trajectory.append([x2, y2])
        
        # 画轨迹
        if len(trajectory) > 1:
            traj_array = np.array(trajectory)
            ax.plot(traj_array[:, 0], traj_array[:, 1], 'c-', linewidth=1, alpha=0.6, label='Trajectory')
        
        ax.plot([x0, x1], [y0, y1], 'b-', linewidth=3)
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=3)
        ax.plot(x0, y0, 'ko', markersize=8)
        ax.plot(x1, y1, 'ko', markersize=6)
        ax.plot(x2, y2, 'ro', markersize=8)  # 末端用红色标记
        
        # 添加信息文本
        info_text = f'Step: {step_count}\nInstant Reward: {reward:.3f}\nTotal Return: {total_reward:.2f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title("2D Arm PPO - Obstacle Avoidance")
        ax.legend(loc='upper right', fontsize=8)
        
        # 保存图像到 buffer
        fig.canvas.draw()
        try:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        except AttributeError:
            # 兼容新版 matplotlib
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            # RGBA to RGB
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            image = image.flatten()
            
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        # 动作
        action, _ = ppo.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    plt.close(fig)
    
    # 打印最终结果
    print(f"Episode finished - Steps: {step_count}, Total Reward: {total_reward:.2f}, Result: {info.get('result', 'unknown')}")
    
    # 保存视频
    try:
        import imageio
        video_path = os.path.join(os.path.dirname(__file__), 'info', save_name)
        # 使用更高的fps和更好的编码质量
        imageio.mimsave(video_path, frames, fps=30, quality=9, codec='libx264')
        print(f"Video saved to {video_path}")
    except ImportError:
        print("imageio not found, skipping video generation.")

def test_single(model_path):
    """使用保存的模型生成一个测试episode并输出视频"""
    env = ArmEnv()
    ppo = PPO(env.state_dim, env.action_dim)
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    ppo.actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ppo.actor.eval()
    print(f"Loaded model from {model_path}")
    
    # 生成可视化
    visualize(ppo, save_name='test_trajectory.mp4')

if __name__ == "__main__":
    import sys
    
    # 确保目录存在
    script_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(script_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "info"), exist_ok=True)
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 测试模式
        model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, 'model', 'ppo_actor_final.pth')
        test_single(model_path)
    else:
        # 训练模式
        ppo_agent, rewards = train()
        
        # 绘制训练曲线
        plt.figure()
        plt.plot(rewards)
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        reward_plot_path = os.path.join(script_dir, "figures", "training_rewards.png")
        plt.savefig(reward_plot_path)
        print(f"Reward plot saved to {reward_plot_path}")
        
        visualize(ppo_agent)
