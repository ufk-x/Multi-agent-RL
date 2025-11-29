"""
2D机械臂避障任务 - 使用PPO算法
任务描述：
- 固定基座的2连杆机械臂（每个连杆长度为1）
- 从起点A运动到目标点B
- 避开半径为0.3的球形障碍物
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from tqdm import tqdm
import os

# ==================== 训练函数 ====================

# ==================== 环境定义 ====================
class Arm2DEnv:
    """
    2D机械臂环境
    - 2个关节，每个连杆长度为1
    - 基座固定在原点(0, 0)
    - 目标：末端执行器从起点运动到目标点，同时避开障碍物
    """
    def __init__(self):
        # 机械臂参数
        self.link_lengths = [1.0, 1.0]  # 两个连杆长度
        self.base_pos = np.array([0.0, 0.0])  # 基座位置
        
        # 障碍物参数
        self.obstacle_pos = np.array([1.2, 0.8])  # 障碍物位置
        self.obstacle_radius = 0.3  # 障碍物半径
        
        # 任务参数
        self.start_pos = np.array([1.5, -0.5])  # 起点A
        self.target_pos = np.array([1.0, 1.5])  # 目标点B
        
        # 状态变量
        self.joint_angles = np.zeros(2)  # [theta1, theta2]
        self.joint_velocities = np.zeros(2)  # [omega1, omega2]
        
        # 环境参数
        self.max_torque = 2.0  # 最大扭矩
        self.dt = 0.05  # 时间步长
        self.max_steps = 500  # 最大步数
        self.current_step = 0
        
        # 奖励权重
        self.reward_target = 1000.0  # 到达目标的奖励
        self.reward_collision = -50.0  # 碰撞惩罚
        self.penalty_distance = -0.1  # 距离惩罚系数
        # self.penalty_action = -0.01  # 动作惩罚系数
        self.penalty_action = 0.0  # 动作惩罚系数
        self.closer_reward_rate = 0.5  # 接近目标的奖励系数
        
        # 距离参数
        self.target_threshold = 0.1  # 到达目标的距离阈值
        self.collision_margin = 0.05  # 碰撞检测边界
        self.dist_to_target_prev = None  # 上一步到目标的距离
        
    def forward_kinematics(self, angles):
        """
        正运动学：根据关节角度计算末端执行器位置和中间关节位置
        返回：joint1_pos, joint2_pos, end_effector_pos
        """
        theta1, theta2 = angles
        
        # 第一个关节位置（连杆1的末端）
        joint1_pos = self.base_pos + np.array([
            self.link_lengths[0] * np.cos(theta1),
            self.link_lengths[0] * np.sin(theta1)
        ])
        
        # 末端执行器位置（连杆2的末端）
        end_effector_pos = joint1_pos + np.array([
            self.link_lengths[1] * np.cos(theta1 + theta2),
            self.link_lengths[1] * np.sin(theta1 + theta2)
        ])
        
        return self.base_pos.copy(), joint1_pos, end_effector_pos
    
    def check_collision(self):
        """
        检查机械臂是否与障碍物碰撞
        需要检查：末端执行器、中间关节、连杆线段
        """
        _, joint1_pos, end_pos = self.forward_kinematics(self.joint_angles)
        
        # 检查末端执行器碰撞
        dist_end = np.linalg.norm(end_pos - self.obstacle_pos)
        if dist_end < self.obstacle_radius + self.collision_margin:
            return True
        
        # 检查中间关节碰撞
        dist_joint1 = np.linalg.norm(joint1_pos - self.obstacle_pos)
        if dist_joint1 < self.obstacle_radius + self.collision_margin:
            return True
        
        # 检查连杆1与障碍物的距离
        dist_link1 = self._point_to_segment_distance(
            self.obstacle_pos, self.base_pos, joint1_pos
        )
        if dist_link1 < self.obstacle_radius + self.collision_margin:
            return True
        
        # 检查连杆2与障碍物的距离
        dist_link2 = self._point_to_segment_distance(
            self.obstacle_pos, joint1_pos, end_pos
        )
        if dist_link2 < self.obstacle_radius + self.collision_margin:
            return True
        
        return False
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """计算点到线段的最短距离"""
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        seg_len = np.linalg.norm(seg_vec)
        
        if seg_len < 1e-6:
            return np.linalg.norm(point_vec)
        
        # 投影参数t
        t = np.dot(point_vec, seg_vec) / (seg_len ** 2)
        t = np.clip(t, 0, 1)
        
        # 最近点
        closest_point = seg_start + t * seg_vec
        return np.linalg.norm(point - closest_point)
    
    def get_state(self):
        """
        获取状态向量
        状态包括：
        - 关节角度 (2维)
        - 关节角速度 (2维)
        - 末端到目标的相对位置 (2维)
        - 末端到障碍物的相对位置和距离 (3维)
        总共9维
        """
        _, _, end_pos = self.forward_kinematics(self.joint_angles)
        
        # 到目标的相对位置
        target_relative = self.target_pos - end_pos
        
        # 到障碍物的相对位置和距离
        obstacle_relative = self.obstacle_pos - end_pos
        obstacle_distance = np.linalg.norm(obstacle_relative)
        
        state = np.concatenate([
            self.joint_angles,
            self.joint_velocities,
            target_relative,
            obstacle_relative,
            [obstacle_distance]
        ])
        
        return state.astype(np.float32)
    
    def reset(self):
        """重置环境到初始状态"""
        # 使用逆运动学找到接近起点的初始关节角度
        self.joint_angles = self._inverse_kinematics(self.start_pos)
        self.joint_velocities = np.zeros(2)
        self.current_step = 0
        
        # Gymnasium API: 返回 (state, info)
        return self.get_state(), {}
    
    def _inverse_kinematics(self, target_pos):
        """
        简单的逆运动学求解（解析解）
        对于2R机械臂，可以使用几何方法
        """
        x, y = target_pos - self.base_pos
        l1, l2 = self.link_lengths
        
        # 计算到目标的距离
        d = np.sqrt(x**2 + y**2)
        
        # 如果目标不可达，使用可达范围内的近似值
        d = np.clip(d, abs(l1 - l2) + 0.1, l1 + l2 - 0.1)
        
        # 使用余弦定理计算theta2
        cos_theta2 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = np.arccos(cos_theta2)
        
        # 计算theta1
        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        return np.array([theta1, theta2])
    
    def step(self, action):
        """
        执行动作
        action: [torque1, torque2] 范围[-max_torque, max_torque]
        """
        # 限制动作范围
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # 简单的动力学模型：tau = I * alpha（忽略重力和摩擦）
        # 这里假设单位惯量
        angular_acceleration = action
        
        # 更新角速度和角度
        self.joint_velocities += angular_acceleration * self.dt
        self.joint_angles += self.joint_velocities * self.dt
        
        # 标准化角度到[-pi, pi]
        self.joint_angles = np.arctan2(np.sin(self.joint_angles), 
                                       np.cos(self.joint_angles))
        
        self.current_step += 1
        
        # 计算奖励
        reward, terminated = self._calculate_reward(action)
        
        # 获取新状态
        next_state = self.get_state()
        
        # Gymnasium API: 返回 (state, reward, terminated, truncated, info)
        truncated = False  # 我们不使用truncated
        return next_state, reward, terminated, truncated, {}
    
    def _calculate_reward(self, action):
        """计算奖励函数"""
        _, _, end_pos = self.forward_kinematics(self.joint_angles)
        
        # 到目标的距离
        dist_to_target = np.linalg.norm(end_pos - self.target_pos)
        if self.dist_to_target_prev is None:
            self.dist_to_target_prev = dist_to_target # 初始化上一距离
        
        # 检查碰撞
        collision = self.check_collision()
        
        # 检查是否到达目标
        reached_target = dist_to_target < self.target_threshold
        
        # 计算奖励
        reward = 0.0
        done = False
        
        if collision:
            reward = self.reward_collision
            done = True
        elif reached_target:
            reward = self.reward_target
            done = True
        else:
            # 距离奖励（越近奖励越大）
            reward += self.penalty_distance * dist_to_target
            
            # 动作惩罚（鼓励平滑运动）
            reward += self.penalty_action * np.sum(action ** 2)
            
            # 鼓励接近目标
            reward += self.closer_reward_rate * (self.dist_to_target_prev - dist_to_target) # 奖励接近目标
            self.dist_to_target_prev = dist_to_target # 更新上一距离
        
        # 超时检查
        if self.current_step >= self.max_steps:
            done = True
        
        return reward, done
    
    @property
    def state_dim(self):
        return 9  # 状态维度
    
    @property
    def action_dim(self):
        return 2  # 动作维度（两个关节的扭矩）

# ==================== PPO网络定义 ====================
class PolicyNetContinuous(torch.nn.Module):
    """
    连续动作空间的策略网络
    输出动作的均值(μ)和标准差(σ)，用于构建正态分布
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        # 共享的隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 输出动作均值的分支
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 输出动作标准差的分支
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 通过共享隐藏层
        x = F.relu(self.fc1(x))
        
        # 计算动作均值，使用tanh确保有界，然后缩放到[-2, 2]
        # 这对Pendulum环境合适，因为其动作范围是[-2, 2]
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        
        # 计算动作标准差，使用softplus确保为正值
        # softplus(x) = log(1 + exp(x))，输出总是正数
        # 添加最小值限制，防止标准差过小导致数值不稳定
        std = F.softplus(self.fc_std(x)) + 1e-5
        
        return mu, std

class ValueNet(torch.nn.Module):
    """
    价值网络(Critic网络)
    用于评估状态的价值函数V(s)
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # 第一层：状态维度 -> 隐藏层维度
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：隐藏层维度 -> 1(状态价值)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 通过ReLU激活函数的第一层
        x = F.relu(self.fc1(x))
        # 输出状态价值(标量)
        return self.fc2(x)

class PPOContinuous:
    """
    处理连续动作空间的PPO算法
    
    与离散动作PPO的主要区别：
    1. 策略网络输出高斯分布的参数(μ, σ)
    2. 动作采样来自正态分布
    3. 对数概率计算使用正态分布的概率密度函数
    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 创建连续动作策略网络
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        # 价值网络与离散情况相同
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 优化器设置
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        
        # PPO参数
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        """
        在连续动作空间中选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            action: 从正态分布采样的连续动作（numpy数组）
        """
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 获取动作分布的参数
        mu, sigma = self.actor(state)
        # 创建正态分布
        action_dist = torch.distributions.Normal(mu, sigma)
        # 从分布中采样动作
        action = action_dist.sample()
        # 返回动作，转换为numpy数组
        return action.detach().cpu().numpy()[0]

    def update(self, transition_dict):
        """
        连续动作PPO的更新函数
        """
        # 数据预处理
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        # 检查输入数据是否包含NaN
        if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(rewards).any():
            print("警告: 检测到NaN值，跳过本次更新")
            return
        
        # 计算TD目标和优势函数（与离散情况相同）
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        
        # 计算旧策略下的对数概率
        mu, std = self.actor(states)
        # 再次检查网络输出是否包含NaN
        if torch.isnan(mu).any() or torch.isnan(std).any():
            print("警告: 网络输出包含NaN值，跳过本次更新")
            return
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 对于连续动作，使用正态分布的对数概率密度，然后对action_dim求和
        old_log_probs = action_dists.log_prob(actions).sum(dim=-1, keepdim=True)

        # PPO多轮更新
        for _ in range(self.epochs):
            # 获取新策略的分布参数
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            
            # 计算新策略的对数概率
            log_probs = action_dists.log_prob(actions).sum(dim=-1, keepdim=True)
            
            # 重要性采样比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO的截断目标函数（与离散情况相同）
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            
            # Critic损失
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            
            # 网络参数更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# ==================== 可视化函数 ====================
def visualize_episode(env, agent, device, save_path='arm_animation.gif'):
    """
    可视化一个episode的执行过程
    """
    state, _ = env.reset()
    done = False
    trajectory = []
    
    while not done:
        # 记录当前状态
        _, joint1_pos, end_pos = env.forward_kinematics(env.joint_angles)
        trajectory.append({
            'base': env.base_pos.copy(),
            'joint1': joint1_pos.copy(),
            'end': end_pos.copy(),
            'angles': env.joint_angles.copy()
        })
        
        # 选择动作
        state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(device)
        mu, _ = agent.actor(state_tensor)
        action = mu.detach().cpu().numpy()[0]
        
        # 执行动作
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def animate(frame):
        ax.clear()
        
        # 设置坐标范围
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制障碍物
        obstacle = Circle(env.obstacle_pos, env.obstacle_radius, 
                         color='red', alpha=0.5, label='Obstacle')
        ax.add_patch(obstacle)
        
        # 绘制起点和目标点
        ax.plot(env.start_pos[0], env.start_pos[1], 'go', 
               markersize=15, label='Start')
        ax.plot(env.target_pos[0], env.target_pos[1], 'b*', 
               markersize=20, label='Target')
        
        # 绘制机械臂
        data = trajectory[frame]
        
        # 绘制连杆
        ax.plot([data['base'][0], data['joint1'][0]], 
               [data['base'][1], data['joint1'][1]], 
               'ko-', linewidth=4, markersize=8, label='Link 1')
        ax.plot([data['joint1'][0], data['end'][0]], 
               [data['joint1'][1], data['end'][1]], 
               'mo-', linewidth=4, markersize=8, label='Link 2')
        
        # 绘制末端执行器
        ax.plot(data['end'][0], data['end'][1], 'ro', 
               markersize=12, label='End Effector')
        
        # 绘制轨迹
        if frame > 0:
            past_trajectory = np.array([trajectory[i]['end'] 
                                       for i in range(frame)])
            ax.plot(past_trajectory[:, 0], past_trajectory[:, 1], 
                   'c-', alpha=0.5, linewidth=1)
        
        ax.set_title(f'2D Arm Obstacle Avoidance - Step {frame}/{len(trajectory)-1}\n'
                    f'Joint Angles: θ1={data["angles"][0]:.2f}, θ2={data["angles"][1]:.2f}',
                    fontsize=14)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(loc='upper right')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory),
                                  interval=50, repeat=True)
    
    # 保存动画
    anim.save(save_path, writer='pillow', fps=20)
    print(f"Animation saved to {save_path}")
    plt.close()

def plot_training_results(return_list, save_path='training_results.png'):
    """绘制训练结果"""
    episodes_list = list(range(len(return_list)))
    
    plt.figure(figsize=(15, 5))
    
    # 原始回报曲线
    plt.subplot(1, 3, 1)
    plt.plot(episodes_list, return_list, alpha=0.6, color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Training Returns')
    plt.grid(True, alpha=0.3)
    
    # 移动平均回报曲线
    plt.subplot(1, 3, 2)
    mv_return = moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return, color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Smoothed Returns (MA-21)')
    plt.grid(True, alpha=0.3)
    
    # 学习进度分析
    plt.subplot(1, 3, 3)
    window_size = 100
    if len(return_list) >= window_size:
        smoothed_returns = []
        for i in range(0, len(return_list) - window_size + 1, window_size):
            smoothed_returns.append(np.mean(return_list[i:i+window_size]))
        
        x_smoothed = np.arange(len(smoothed_returns)) * window_size + window_size // 2
        plt.plot(x_smoothed, smoothed_returns, 'o-', color='green', 
                linewidth=2, markersize=6)
    plt.xlabel('Episodes')
    plt.ylabel('Average Returns')
    plt.title('Learning Progress')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training results saved to {save_path}")
    plt.show()


# ==================== 主训练函数 ====================
def moving_average(a, window_size):
    """计算移动平均，边界处使用部分窗口平均
    """
    a = np.array(a)  # 先转换为 NumPy 数组
    a = np.clip(a, -200, 200)  # 将奖励裁剪到[-200, 200]范围
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env:Arm2DEnv, agent, num_episodes):
    """训练on-policy智能体（如PPO、A2C等）"""
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()  # Gymnasium API
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)  # Gymnasium API
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    """计算GAE优势函数"""
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def train_arm():
    """训练2D机械臂"""
    print("=" * 60)
    print("2D机械臂避障任务 - PPO训练")
    print("=" * 60)
    
    # 超参数设置
    actor_lr = 1e-4 # Actor网络学习率
    critic_lr = 5e-4  # Critic网络学习率，通常设置得比Actor大
    num_episodes = 6000 # 训练episode数量
    hidden_dim = 256  # 神经网络隐藏层维度
    gamma = 0.99  # 折扣因子，接近1表示更关注长期奖励
    lmbda = 0.95  # GAE中的λ参数，平衡偏差和方差
    epochs = 10  # 每次收集经验后的训练轮数
    eps = 0.2  # PPO截断参数，控制策略更新幅度
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = Arm2DEnv()
    print(f"\n环境信息:")
    print(f"  状态维度: {env.state_dim}")
    print(f"  动作维度: {env.action_dim}")
    print(f"  起点位置: {env.start_pos}")
    print(f"  目标位置: {env.target_pos}")
    print(f"  障碍物位置: {env.obstacle_pos}, 半径: {env.obstacle_radius}")
    
    # 创建PPO智能体
    agent = PPOContinuous(
        state_dim=env.state_dim,
        hidden_dim=hidden_dim,
        action_dim=env.action_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        lmbda=lmbda,
        epochs=epochs,
        eps=eps,
        gamma=gamma,
        device=device
    )
    
    print(f"\n开始训练...")
    print(f"  总episodes: {num_episodes}")
    print(f"  Actor学习率: {actor_lr}")
    print(f"  Critic学习率: {critic_lr}")
    print(f"  隐藏层维度: {hidden_dim}")
    
    # 训练
    return_list = train_on_policy_agent(env, agent, num_episodes)
    
    # 训练结果分析
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最后100个episode平均回报: {np.mean(return_list[-100:]):.2f}")
    print(f"最高回报: {np.max(return_list):.2f}")
    print(f"性能改善: {np.mean(return_list[-100:]) - np.mean(return_list[:100]):.2f}")
    
    # 绘制训练结果
    plot_path = '动手学强化学习/Hands-on-RL/2D_arm/figures/arm_training_results.png'
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plot_training_results(return_list, plot_path)
    
    # 可视化训练后的表现
    print("\n生成演示动画...")
    gif_path = '动手学强化学习/Hands-on-RL/2D_arm/figures/arm_training_animation.gif'
    if not os.path.exists(os.path.dirname(gif_path)):
        os.makedirs(os.path.dirname(gif_path))
    visualize_episode(env, agent, device, gif_path)
    
    # 保存模型
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, '动手学强化学习/Hands-on-RL/2D_arm/model/arm2d_ppo_model.pth')
    print("模型已保存到 动手学强化学习/Hands-on-RL/2D_arm/model/arm2d_ppo_model.pth")
    
    return agent, env, return_list


def test_arm(model_path='动手学强化学习/Hands-on-RL/2D_arm/model/arm2d_ppo_model.pth'):
    """测试已训练的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境和智能体
    env = Arm2DEnv()
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
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"模型已从 {model_path} 加载")
    
    # 可视化
    gif_path = '动手学强化学习/Hands-on-RL/2D_arm/figures/arm_testing_animation.gif'
    if not os.path.exists(os.path.dirname(gif_path)):
        os.makedirs(os.path.dirname(gif_path))
    visualize_episode(env, agent, device, gif_path)


if __name__ == "__main__":
    # 训练模型
    agent, env, return_list = train_arm()
    
    # 如果想要测试已保存的模型，可以取消下面的注释
    # test_arm('arm2d_ppo_model.pth')