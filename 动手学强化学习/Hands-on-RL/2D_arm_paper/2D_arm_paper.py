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
    def __init__(self, device=None):
        # 设备配置
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 单位转换参数
        self.m_to_env_scale = 100.0  # 米到环境单位（厘米）的转换比例
        
        # 机械臂参数（使用torch张量，原始单位为米）
        self.link_lengths = torch.tensor([1.0, 1.0], dtype=torch.float32, device=self.device) * self.m_to_env_scale
        self.base_pos = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device) * self.m_to_env_scale
        
        # 障碍物参数（原始单位为米）
        self.obstacle_pos = torch.tensor([1.2, 0.8], dtype=torch.float32, device=self.device) * self.m_to_env_scale
        self.obstacle_radius = 0.3 * self.m_to_env_scale
        
        # 任务参数（原始单位为米）
        self.start_pos = torch.tensor([1.5, -0.5], dtype=torch.float32, device=self.device) * self.m_to_env_scale
        self.target_pos = torch.tensor([1.0, 1.5], dtype=torch.float32, device=self.device) * self.m_to_env_scale
        
        # 状态变量
        self.joint_angles = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.joint_velocities = torch.zeros(2, dtype=torch.float32, device=self.device)
        
        # 环境参数
        self.max_torque = 2.0 * self.m_to_env_scale  # 最大扭矩(从Nm转换为环境单位)
        self.dt = 0.05  # 时间步长
        self.max_steps = 1000  # 最大步数
        self.current_step = 0
        self.if_done_when_collision = False  # 碰撞时是否终止episode
        
        # 奖励权重
        self.reward_target = 1000.0  # 到达目标的奖励
        self.reward_collision = 1000.0  # 碰撞惩罚
        self.penalty_distance = -1.0  # 距离惩罚系数
        self.penalty_action = 0.01  # 动作惩罚系数
        self.closer_reward_rate = 12.0  # 接近目标的奖励系数
        self.penalty_static = -0.5  # 静止不动的惩罚系数
        self.static_threshold = 0.001 * self.m_to_env_scale  # 判定为静止的速度阈值（原始单位为m/s，转换为cm/s）
        
        # 距离参数（单位为cm）
        self.target_threshold = 1e-2 * self.m_to_env_scale  # 到达目标的距离阈值，从米转换到环境单位
        self.tau_e = 1e-4  # 目标距离稳定阈值
        self.collision_margin = 1e-3 * self.m_to_env_scale  # 碰撞检测边界，从米转换为环境单位
        self.collision_punish_distance = 2 * self.collision_margin  # 碰撞惩罚距离，是碰撞边界的两倍
        self.dist_to_target_prev = None  # 上一步到目标的距离
        
    def forward_kinematics(self, angles):
        """
        正运动学：根据关节角度计算末端执行器位置和中间关节位置
        返回：joint1_pos, joint2_pos, end_effector_pos
        """
        # 确保angles是torch张量
        if not isinstance(angles, torch.Tensor):
            angles = torch.tensor(angles, dtype=torch.float32, device=self.device)
        
        theta1, theta2 = angles[0], angles[1]
        
        # 第一个关节位置（连杆1的末端）- 使用torch操作
        joint1_pos = self.base_pos + torch.stack([
            self.link_lengths[0] * torch.cos(theta1),
            self.link_lengths[0] * torch.sin(theta1)
        ])
        
        # 末端执行器位置（连杆2的末端）
        end_effector_pos = joint1_pos + torch.stack([
            self.link_lengths[1] * torch.cos(theta1 + theta2),
            self.link_lengths[1] * torch.sin(theta1 + theta2)
        ])
        
        return self.base_pos.clone(), joint1_pos, end_effector_pos
    
    def check_collision(self):
        """
        检查机械臂是否与障碍物碰撞
        需要检查：末端执行器、中间关节、连杆线段
        """
        _, joint1_pos, end_pos = self.forward_kinematics(self.joint_angles)
        return self.check_collision_fast(end_pos, joint1_pos)
    
    def check_collision_fast(self, end_pos, joint1_pos=None):
        """
        快速碰撞检测 - 接受预计算的位置，避免重复计算forward_kinematics
        """
        # 如果没有提供joint1_pos，需要计算
        if joint1_pos is None:
            _, joint1_pos, _ = self.forward_kinematics(self.joint_angles)
        
        # 使用torch计算距离
        dist_end = torch.norm(end_pos - self.obstacle_pos)
        if dist_end < self.obstacle_radius + self.collision_margin:
            return True
        
        # 检查中间关节碰撞
        dist_joint1 = torch.norm(joint1_pos - self.obstacle_pos)
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
    
    def get_min_distance_to_obstacle(self):
        """
        计算机械臂各部分到障碍物的最小距离
        """
        _, joint1_pos, end_pos = self.forward_kinematics(self.joint_angles)
        
        # 使用torch计算距离
        dist_end = torch.norm(end_pos - self.obstacle_pos)
        dist_joint1 = torch.norm(joint1_pos - self.obstacle_pos)
        dist_link1 = self._point_to_segment_distance(
            self.obstacle_pos, self.base_pos, joint1_pos
        )
        dist_link2 = self._point_to_segment_distance(
            self.obstacle_pos, joint1_pos, end_pos
        )
        
        return torch.min(torch.stack([dist_end, dist_joint1, dist_link1, dist_link2]))
    
    def get_dis_link_to_obstacle_from_positions(self, joint1_pos, end_pos):
        """
        计算每一条连杆和关节到障碍物的距离 - 使用预计算的位置
        """
        base_pos = self.base_pos
        # 使用torch计算距离
        seg_starts = torch.stack([base_pos, joint1_pos])
        seg_ends = torch.stack([joint1_pos, end_pos])
        e12 = seg_ends - seg_starts
        e1x = self.obstacle_pos - seg_starts
        seg_lens = torch.norm(e12, dim=1)
        t = torch.clamp(torch.sum(e1x * e12, dim=1) / (seg_lens ** 2), 0, 1)
        closest_points = seg_starts + (t.unsqueeze(1) * e12)
        dists = torch.norm(self.obstacle_pos - closest_points, dim=1)
        return dists  # 返回每条连杆的距离张量
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """计算点到线段的最短距离 - 使用torch加速"""
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        seg_len = torch.norm(seg_vec)
        
        if seg_len < 1e-6:
            return torch.norm(point_vec)
        
        # 投影参数t
        t = torch.dot(point_vec, seg_vec) / (seg_len ** 2)
        t = torch.clamp(t, 0, 1)
        
        # 最近点
        closest_point = seg_start + t * seg_vec
        return torch.norm(point - closest_point)
    
    def get_state(self, return_numpy=True):
        """
        获取状态向量
        状态包括：
        - 关节角度 (2维)
        - 末端位置 (2维)
        - error向量 [末端到目标距离, 末端到目标的x偏差, 末端到目标的y偏差](3维)
        - 每个连杆到障碍物的距离 (2维)
        总共9维
        """
        _, joint1_pos, end_pos = self.forward_kinematics(self.joint_angles)
        
        # 到目标的距离 - 使用torch计算
        ee_error_vec = self.target_pos - end_pos
        target_dis = torch.norm(ee_error_vec)
        
        # 到障碍物的距离
        dis_to_obs = self.get_dis_link_to_obstacle_from_positions(joint1_pos, end_pos)

        # 使用torch拼接
        state = torch.cat([
            self.joint_angles,
            end_pos,
            torch.tensor([target_dis, ee_error_vec[0], ee_error_vec[1]], device=self.device),
            dis_to_obs  # dis_to_obs已经是1维张量，不需要unsqueeze
        ])
        
        # 只在需要时转换为numpy
        if return_numpy:
            return state.cpu().numpy().astype(np.float32)
        return state
    
    def reset(self):
        """重置环境到初始状态"""
        # 使用逆运动学找到接近起点的初始关节角度
        self.joint_angles = self._inverse_kinematics(self.start_pos)
        self.joint_velocities = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.current_step = 0
        self.dist_to_target_prev = None
        self._cached_end_pos = None  # 清除缓存
        
        # Gymnasium API: 返回 (state, info)
        return self.get_state(), {}
    
    def _inverse_kinematics(self, target_pos):
        """
        简单的逆运动学求解（解析解）- 使用torch加速
        对于2R机械臂，可以使用几何方法
        """
        rel_pos = target_pos - self.base_pos
        x, y = rel_pos[0], rel_pos[1]
        l1, l2 = self.link_lengths[0], self.link_lengths[1]
        
        # 计算到目标的距离
        d = torch.sqrt(x**2 + y**2)
        
        # 如果目标不可达，使用可达范围内的近似值
        d = torch.clamp(d, abs(l1 - l2) + 0.1 * self.m_to_env_scale, l1 + l2 - 0.1 * self.m_to_env_scale)
        
        # 使用余弦定理计算theta2
        cos_theta2 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = torch.clamp(cos_theta2, -1, 1)
        theta2 = torch.acos(cos_theta2)
        
        # 计算theta1
        k1 = l1 + l2 * torch.cos(theta2)
        k2 = l2 * torch.sin(theta2)
        theta1 = torch.atan2(y, x) - torch.atan2(k2, k1)
        
        return torch.stack([theta1, theta2])
    
    def step(self, action, return_reward_details=True):
        """
        执行动作
        action: [torque1, torque2] 范围[-max_torque, max_torque]
        return_reward_details: 是否在info中返回奖励详情
        """
        # 转换action为torch张量并限制范围
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        action = torch.clamp(action, -self.max_torque, self.max_torque)
        
        # 简单的动力学模型：tau = I * alpha（忽略重力和摩擦）
        # 这里假设单位惯量
        angular_acceleration = action # shape: (2,)
        
        # 更新角速度和角度
        self.joint_velocities += angular_acceleration * self.dt
        self.joint_angles += self.joint_velocities * self.dt
        
        # 标准化角度到[-pi, pi] - 使用torch操作
        self.joint_angles = torch.atan2(torch.sin(self.joint_angles), 
                                        torch.cos(self.joint_angles))
        
        # 清除缓存的end_pos
        self._cached_end_pos = None
        
        self.current_step += 1
        
        # 计算奖励（内部会计算end_pos并缓存）
        if return_reward_details:
            reward, reached_target, collision, reward_details = self._calculate_reward(action, return_details=True)
        else:
            reward, reached_target, collision = self._calculate_reward(action, return_details=False)
            reward_details = None
        
        # 获取新状态（使用缓存的end_pos）
        next_state = self.get_state()
        
        # Gymnasium API: 返回 (state, reward, terminated, truncated, info)
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True  # 如果时间到，则truncated为真
        
        info = {}
        if return_reward_details and reward_details is not None:
            info['reward_details'] = reward_details
            
        return next_state, reward, reached_target, collision, truncated, info
    
    def _calculate_reward(self, action, return_details=True):
        """计算奖励函数 - 使用torch加速和缓存
        
        Args:
            action: 动作向量，形状为(2,)
            return_details: 是否返回奖励详细信息字典
        
        Returns:
            如果return_details=False: (reward, reached_target, collision)
            如果return_details=True: (reward, reached_target, collision, reward_details)
        """
        # 计算并缓存end_pos
        _, joint1_pos, end_pos = self.forward_kinematics(self.joint_angles) # shape: (2,)
        # self._cached_end_pos = end_pos
        
        # 到目标的距离（环境单位）- 所有位置已经是环境单位，无需额外转换
        dist_to_target = torch.norm(end_pos - self.target_pos)
        dist_to_target_val = dist_to_target.item()  # 已经是环境单位
        
        if self.dist_to_target_prev is None:
            self.dist_to_target_prev = dist_to_target_val # 初始化上一距离
        
        # 检查碰撞（使用已缓存的end_pos）
        collision = self.check_collision_fast(end_pos,joint1_pos)
        
        # 检查是否到达目标
        reached_target = dist_to_target_val <= self.target_threshold
        
        # 初始化奖励详情字典
        reward_details = {
            'distance_penalty': 0.0,
            'closer_reward': 0.0,
            'collision_penalty': 0.0,
            'action_penalty': 0.0,
            'total': 0.0
        }
        
        # 计算奖励
        reward = 0.0
        # -w1 * e^2
        # reward_details['distance_penalty'] = -self.penalty_distance * dist_to_target_val**2
        # reward += reward_details['distance_penalty']
        reward_details['distance_penalty'] =  -np.log(dist_to_target_val**2 + self.tau_e)
        reward += reward_details['distance_penalty']

        # -ln(e^2 + self.tau_e)
        reward_details['closer_reward'] =  (self.dist_to_target_prev - dist_to_target_val) * self.closer_reward_rate
        self.dist_to_target_prev = dist_to_target_val
        reward += reward_details['closer_reward']

        # -w2 * \sum \phi_i, \phi_i = max(0, 1- d_i /self.collision_punish_distance)
        # d_1, d_2 = self.get_dis_link_to_obstacle_from_positions(joint1_pos, end_pos)
        # phi_1 = max(0.0, 1.0 - d_1.item() / (self.obstacle_radius + self.collision_punish_distance))
        # phi_2 = max(0.0, 1.0 - d_2.item() / (self.obstacle_radius + self.collision_punish_distance))
        # sum_phi = phi_1 + phi_2
        # reward_details['collision_penalty'] = -self.reward_collision * sum_phi
        if collision:
            reward_details['collision_penalty'] = -self.reward_collision
        else:
            reward_details['collision_penalty'] = 0.0
        reward += reward_details['collision_penalty']

        # -w3 * ||action||^2 (对动作的平方进行惩罚)
        action_squared = torch.sum(action ** 2).item()
        reward_details['action_penalty'] = -self.penalty_action * action_squared
        reward += reward_details['action_penalty']

        # 统计总惩罚
        reward_details['total'] = reward
        
        if return_details:
            return reward, reached_target, collision, reward_details
        return reward, reached_target, collision
    
    @property
    def state_dim(self):
        return 9  # 状态维度：关节角度(2) + 末端位置(2) + error向量(3) + 连杆到障碍物距离(2)
    
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
        with torch.no_grad():  # 推理时不需要梯度
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            # 获取动作分布的参数
            mu, sigma = self.actor(state)
            # 创建正态分布
            action_dist = torch.distributions.Normal(mu, sigma)
            # 从分布中采样动作
            action = action_dist.sample()
            # 返回动作，转换为numpy数组
            return action.cpu().numpy()[0]

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
def visualize_episode(env:Arm2DEnv, agent, device, save_path='arm_animation.mp4'):
    """
    可视化一个episode的执行过程
    """
    state, _ = env.reset()
    done = False
    trajectory = []
    rewards = []  # 记录每步的奖励
    reward_details_list = []  # 记录每步的奖励详情
    
    # 记录结束原因
    end_reason = "Unknown"
    
    while not done:
        # 记录当前状态 - 转换torch张量为numpy
        _, joint1_pos, end_pos = env.forward_kinematics(env.joint_angles)
        trajectory.append({
            'base': env.base_pos.cpu().numpy(),
            'joint1': joint1_pos.cpu().numpy(),
            'end': end_pos.cpu().numpy(),
            'angles': env.joint_angles.cpu().numpy()
        })
        
        # 选择动作
        state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(device)
        mu, _ = agent.actor(state_tensor)
        action = mu.detach().cpu().numpy()[0]
        
        # 执行动作，并获取奖励详情
        state, reward, reached_target, collision, truncated, info = env.step(action, return_reward_details=True)
        rewards.append(reward)  # 记录每步奖励
        
        # 从info中提取奖励详情
        if 'reward_details' in info:
            reward_details_list.append(info['reward_details'])
        
        # 判断结束原因
        if reached_target:
            end_reason = "Reached Target ✓"
            done = True
        elif collision and env.if_done_when_collision:
            end_reason = "Collision ✗"
            done = True
        elif truncated:
            end_reason = "Truncated (Max Steps)"
            done = True
    
    # 创建动画 - 使用更大的画布来容纳奖励详情
    fig, (ax, ax_text) = plt.subplots(1, 2, figsize=(16, 12), 
                                       gridspec_kw={'width_ratios': [2, 1]})
    
    def animate(frame):
        ax.clear()
        ax_text.clear()
        
        # 计算当前帧的累积奖励
        cumulative_reward = sum(rewards[:frame+1]) if frame < len(rewards) else sum(rewards)
        
        # 左侧：绘制机械臂场景（坐标范围需要根据环境单位调整）
        scale = env.m_to_env_scale  # 获取单位缩放比例
        ax.set_xlim(-2.5 * scale, 2.5 * scale)
        ax.set_ylim(-2.5 * scale, 2.5 * scale)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制障碍物
        obstacle_pos_np = env.obstacle_pos.cpu().numpy() if isinstance(env.obstacle_pos, torch.Tensor) else env.obstacle_pos
        obstacle = Circle(obstacle_pos_np, env.obstacle_radius, 
                         color='red', alpha=0.5, label='Obstacle')
        ax.add_patch(obstacle)
        
        # 绘制起点和目标点
        start_pos_np = env.start_pos.cpu().numpy() if isinstance(env.start_pos, torch.Tensor) else env.start_pos
        target_pos_np = env.target_pos.cpu().numpy() if isinstance(env.target_pos, torch.Tensor) else env.target_pos
        ax.plot(start_pos_np[0], start_pos_np[1], 'go', 
               markersize=15, label='Start')
        ax.plot(target_pos_np[0], target_pos_np[1], 'b*', 
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
        
        # 计算当前末端到目标的距离
        dist_to_target = np.linalg.norm(data['end'] - target_pos_np)
        
        ax.set_title(f'2D Arm Obstacle Avoidance - Step {frame}/{len(trajectory)-1}\n'
                    f'Joint Angles: θ1={data["angles"][0]:.2f}, θ2={data["angles"][1]:.2f}\n'
                    f'Distance to Target: {dist_to_target:.2f} cm\n'
                    f'Cumulative Reward: {cumulative_reward:.2f}\n'
                    f'End Reason: {end_reason}',
                    fontsize=12)
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        
        # 右侧：显示当前帧的奖励详情
        ax_text.axis('off')
        if frame < len(reward_details_list):
            details = reward_details_list[frame]
            
            # 构建奖励详情文本
            reward_text = f"[Step {frame} Reward Details]\n\n"
            reward_text += f"Distance Penalty: {details['distance_penalty']:+.8f}\n"
            reward_text += f"Closer Reward: {details['closer_reward']:+.8f}\n"
            reward_text += f"Collision Penalty: {details['collision_penalty']:+.8f}\n"
            reward_text += f"Action Penalty: {details['action_penalty']:+.8f}\n"
            reward_text += f"{'─' * 30}\n"
            reward_text += f"Total Reward: {details['total']:+.8f}\n"
            
            ax_text.text(0.1, 0.5, reward_text, 
                        fontsize=11, family='monospace',
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory),
                                  interval=50, repeat=True)
    
    # 保存动画为MP4格式
    total_reward = sum(rewards)
    anim.save(save_path, writer='ffmpeg', fps=20, dpi=100, bitrate=1800)
    print(f"Animation saved to {save_path}")
    print(f"Episode Total Reward: {total_reward:.2f}")
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
    # plt.show()


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

def train_on_policy_agent(env:Arm2DEnv, agent:PPOContinuous, num_episodes):
    """训练on-policy智能体（如PPO、A2C等）"""
    return_list = []
    best_avg_return = -float('inf')
    
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()  # Gymnasium API
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, reached_target, collision, truncated, _ = env.step(action)  # Gymnasium API
                    done = reached_target or (collision and env.if_done_when_collision) or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                
                # 更新显示和检查是否达到好的性能
                if (i_episode+1) % 10 == 0:
                    avg_return = np.mean(return_list[-10:])
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % avg_return})
                    
                    # 如果平均奖励为正且是历史最佳，记录
                    if avg_return > best_avg_return:
                        best_avg_return = avg_return
                        if avg_return > 50:  # 达到较好性能
                            print(f"\n达到目前最佳性能! 平均奖励: {avg_return:.2f}")
                
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

# 超参数设置
m_to_env_scale = 100.0  # 米到环境单位（厘米）的转换比例
actor_lr = 3e-4  # Actor网络学习率（降低从1e-3到3e-4）
critic_lr = 1e-3  # Critic网络学习率（降低从5e-3到1e-3）
num_episodes = 1000  # 训练episode数量（减少以加快迭代）
hidden_dim = 512  # 神经网络隐藏层维度（从512降到256，减少过拟合）
gamma = 0.98  # 折扣因子（从0.99降到0.98，更关注近期奖励）
lmbda = 0.95  # GAE中的λ参数，平衡偏差和方差
epochs = 5  # 每次收集经验后的训练轮数（降低到5加快训练）
eps = 0.2  # PPO截断参数，控制策略更新幅度

# 环境参数
max_torque = 2.0 * m_to_env_scale  # 最大扭矩（恢复到2.0给予更多控制能力）
dt = 0.05  # 时间步长
max_steps = 500  # 最大步数（从1000大幅降低到300，避免累积过多负奖励）
target_threshold = 0.5e-2 * m_to_env_scale  # 到达目标的距离阈值，单位为环境单位,相当于0.5cm
tau_e = 1e-4  # 目标距离稳定阈值
collision_margin = 1e-3 * m_to_env_scale  # 碰撞检测边界
if_done_when_collision = False  # 碰撞时终止episode

# 奖励权重 - 重新设计使奖励更平衡
w1 = 1e-4  # 距离惩罚系数（从-1.0降到-0.1）
w2 = 1000.0  # 碰撞惩罚
penalty_action = 0.1  # 动作惩罚系数

reward_target = 100.0  # 到达目标的奖励（从1000降到100）
closer_reward_rate = 30.0  # 接近目标的奖励系数
penalty_static = -0.5  # 静止不动的惩罚系数
static_threshold = 0.01 * m_to_env_scale  # 判定为静止的速度阈值（原始单位为m/s，转换为cm/s）

def train_arm():
    """训练2D机械臂"""
    print("=" * 60)
    print("2D机械臂避障任务 - PPO训练")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境（传递device以使用torch加速）
    env = Arm2DEnv(device=device)
    env.m_to_env_scale = m_to_env_scale
    env.max_torque = max_torque
    env.dt = dt
    env.max_steps = max_steps
    env.target_threshold = target_threshold
    env.tau_e = tau_e
    env.collision_margin = collision_margin
    env.collision_punish_distance = 2 * collision_margin
    env.if_done_when_collision = if_done_when_collision
    env.reward_target = reward_target
    env.reward_collision = w2
    env.penalty_distance = w1
    env.penalty_action = penalty_action
    env.closer_reward_rate = closer_reward_rate
    env.penalty_static = penalty_static
    env.static_threshold = static_threshold

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
    start_time = os.times()
    
    # 训练
    return_list = train_on_policy_agent(env, agent, num_episodes)
    
    # 训练结果分析
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    end_time = os.times()
    print(f"总训练时间: {end_time.elapsed - start_time.elapsed:.2f} 秒")
    print(f"平均回报: {np.mean(return_list):.2f}")
    print(f"最后100个episode平均回报: {np.mean(return_list[-100:]):.2f}")
    print(f"最高回报: {np.max(return_list):.2f}")
    print(f"性能改善: {np.mean(return_list[-100:]) - np.mean(return_list[:100]):.2f}")
    
    # 绘制训练结果
    plot_path = '动手学强化学习/Hands-on-RL/2D_arm_paper/figures/arm_training_results.png'
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plot_training_results(return_list, plot_path)
    
    # 保存模型
    model_dir = '动手学强化学习/Hands-on-RL/2D_arm_paper/model/arm2d_ppo_model.pth'
    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, model_dir)
    print(f"模型已保存到 {model_dir}")

    # 可视化训练后的表现
    print("\n生成演示动画...")
    mp4_path = '动手学强化学习/Hands-on-RL/2D_arm_paper/figures/arm_training_animation.mp4'
    if not os.path.exists(os.path.dirname(mp4_path)):
        os.makedirs(os.path.dirname(mp4_path))
    visualize_episode(env, agent, device, mp4_path)
    
    return agent, env, return_list


def test_arm(model_path='动手学强化学习/Hands-on-RL/2D_arm_paper/model/arm2d_ppo_model.pth'):
    """测试已训练的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境和智能体（传递device以使用torch加速）
    env = Arm2DEnv(device=device)
    env.m_to_env_scale = m_to_env_scale
    env.max_torque = max_torque
    env.dt = dt
    env.max_steps = max_steps
    env.target_threshold = target_threshold
    env.collision_margin = collision_margin
    env.collision_punish_distance = 2 * collision_margin
    env.if_done_when_collision = if_done_when_collision
    env.reward_target = reward_target
    env.reward_collision = w2
    env.penalty_distance = w1
    env.penalty_action = penalty_action
    env.closer_reward_rate = closer_reward_rate
    env.penalty_static = penalty_static
    env.static_threshold = static_threshold
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
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"模型已从 {model_path} 加载")
    
    # 可视化
    mp4_path = '动手学强化学习/Hands-on-RL/2D_arm_paper/figures/arm_testing_animation.mp4'
    if not os.path.exists(os.path.dirname(mp4_path)):
        os.makedirs(os.path.dirname(mp4_path))
    visualize_episode(env, agent, device, mp4_path)

def main():
    IS_TRAIN = True  # 设置为True进行训练
    # IS_TRAIN = False  # False进行测试已保存的模型
    if IS_TRAIN:
        start_time =  os.times()
        # 训练模型
        agent, env, return_list = train_arm()
        end_time =  os.times()
        print(f"\n总训练时间: {end_time.elapsed - start_time.elapsed:.2f} 秒")
    else:
        test_arm('动手学强化学习/Hands-on-RL/2D_arm_paper/model/arm2d_ppo_model.pth')

if __name__ == "__main__":
    main()