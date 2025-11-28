import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard日志记录工具
from torch.utils import data
import torch
import torch.nn as nn

# 引用上级目录
import sys
# sys.path.append("..")
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import grid_env

def arr_in_list(array, _list):
    """检查数组是否在列表中"""
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False

"""
REINFORCE算法（蒙特卡洛策略梯度算法）

算法简介：
    REINFORCE是一种基于策略梯度的强化学习算法，也是最早的策略梯度方法之一。
    与值函数方法（如Q-learning）不同，REINFORCE直接优化策略参数。

核心思想：
    1. 使用神经网络参数化策略π(a|s,θ)
    2. 通过与环境交互收集完整episode
    3. 计算每个状态-动作对的累积回报（Return）
    4. 使用策略梯度定理更新参数：∇θ J(θ) = E[∇θ log π(a|s,θ) * G_t]

算法特点：
    - 无偏但高方差的梯度估计
    - 适用于连续动作空间
    - 可以学习随机策略
    - 收敛到局部最优解

策略梯度定理：
    策略梯度 = E[∇θ log π(a|s,θ) * Q^π(s,a)]
    在REINFORCE中，用实际回报G_t替代Q值：G_t = Σ_{k=0}^∞ γ^k * r_{t+k+1}

网络结构：
    - 输入：状态坐标(x,y)归一化后的值
    - 隐藏层：100个神经元，ReLU激活
    - 输出：5个动作的概率分布，Softmax激活
"""

# 定义策略网络
class PolicyNet(nn.Module):
    """
    策略神经网络类
    
    用于参数化策略函数π(a|s,θ)，将状态映射为动作概率分布
    
    网络架构：
        状态输入(x,y) → 线性层(2→100) → ReLU → 线性层(100→5) → Softmax → 动作概率
    
    参数：
        input_dim: 输入维度，默认为2（x,y坐标）
        output_dim: 输出维度，默认为5（上下左右停留五个动作）
    """
    def __init__(self, input_dim=2, output_dim=5):
        """
        初始化策略网络
        
        :param input_dim: 输入维度（状态空间维度）
        :param output_dim: 输出维度（动作空间大小）
        """
        super(PolicyNet, self).__init__()
        # 定义神经网络结构
        self.fc = nn.Sequential(
            # 第一层：线性变换，从状态空间映射到隐藏层
            nn.Linear(in_features=input_dim, out_features=100),
            # 激活函数：ReLU，增加非线性能力
            nn.ReLU(),
            # 输出层：从隐藏层映射到动作空间
            nn.Linear(in_features=100, out_features=output_dim),
            # Softmax层：将输出转换为概率分布，确保所有动作概率和为1
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        前向传播函数
        
        :param x: 输入状态，形状为(batch_size, input_dim)
        :return: 动作概率分布，形状为(batch_size, output_dim)
        """
        # 确保输入为float32类型，满足PyTorch计算要求
        x = x.type(torch.float32)
        return self.fc(x)


class REINFORCE():
    """
    REINFORCE算法实现类
    
    实现经典的蒙特卡洛策略梯度算法，用于解决网格世界导航问题
    """
    def __init__(self, alpha, env):
        """
        初始化REINFORCE算法求解器
        
        :param alpha: 学习率，控制策略参数更新的步长
        :param env: 网格世界环境实例
        """
        # ============ 算法参数设置 ============
        self.gama = 0.9  # 折扣因子γ，决定未来奖励的权重，越接近1越重视长期回报
        self.alpha = alpha  # 学习率α，控制神经网络参数更新的步长
        self.env = env  # 环境对象实例
        
        # ============ 状态和动作空间设置 ============
        self.action_space_size = env.action_space_size  # 动作空间大小：5（上下左右停留）
        self.state_space_size = env.size ** 2  # 状态空间大小：25（5x5网格）
        
        # ============ 奖励系统设置 ============
        # reward_list通常为[other, target, forbidden, overflow]对应的奖励值
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # 例如：[0, 1, -10, -10]
        
        # ============ 值函数和策略初始化 ============
        # 状态值函数V(s)，虽然REINFORCE不直接学习值函数，但用于记录和可视化
        self.state_value = np.zeros(shape=self.state_space_size)  # 形状：(25,)
        print("self.state_value:", self.state_value)
        
        # Q值表Q(s,a)，用于存储episode中计算得到的累积回报
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 形状：(25, 5)
        
        # 平均策略：每个状态下所有动作概率均等，作为参考基准
        self.mean_policy = np.ones(  # self.mean_policy shape: (25, 5)
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        
        # 当前策略π(a|s)，初始化为平均策略，训练后会被神经网络策略替代
        self.policy = self.mean_policy.copy()
        
        # ============ 日志记录系统 ============
        self.writer = SummaryWriter("logs")  # TensorBoard日志写入器，用于记录训练过程

        # ============ 打印环境信息 ============
        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))
        print('----------------------------------------------------------------')

    def show_policy(self):
        """
        在网格世界中可视化策略
        
        对每个状态，根据策略概率绘制动作箭头：
        - 箭头长度与该动作的选择概率成正比
        - 箭头方向表示动作方向（上下左右或停留）
        - 通过箭头的大小和方向可以直观看出策略的偏好
        
        可视化原理：
        - policy * 0.4：控制箭头长度，概率越大箭头越长
        - action_to_direction：将动作索引映射为方向向量
        - policy * 0.1：控制箭头粗细，概率越大箭头越粗
        """
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]  # 获取在状态state下选择动作action的概率
                # 绘制动作箭头：位置、方向、大小都由策略概率决定
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        """
        在网格世界中显示状态值
        
        :param state_value: 要显示的状态值数组，形状为(state_space_size,)
        :param y_offset: 文本在Y轴上的偏移量，用于在同一格子显示多个值时避免重叠
        
        在每个状态位置显示其对应的值函数数值，便于观察值函数的分布：
        - 数值保留1位小数以提高可读性
        - size_discount=0.7缩小字体以适应格子大小
        """
        for state in range(self.state_space_size):
            # 在每个状态位置显示该状态的值（保留1位小数）
            self.env.render_.write_word(pos=self.env.state2pos(state), 
                                       word=str(round(state_value[state], 1)),
                                       y_offset=y_offset,
                                       size_discount=0.7)
    
    def compute_state_values(self, policy_net, max_episodes=100):
        """
        通过蒙特卡洛方法计算状态值函数V^π(s)
        
        对于每个状态，运行多个episodes来估计该状态的期望回报
        
        :param policy_net: 训练好的策略网络
        :param max_episodes: 用于估计每个状态值的episode数量
        :return: 状态值数组
        """
        state_values = np.zeros(self.state_space_size)
        state_visit_count = np.zeros(self.state_space_size)
        
        # 对每个状态运行多个episodes来估计其值
        for start_state in range(self.state_space_size):
            # 跳过禁区状态和边界外状态
            start_pos = self.env.state2pos(start_state)
            if (arr_in_list(start_pos, self.env.forbidden_location) or 
                start_pos[0] < 0 or start_pos[0] >= self.env.size or
                start_pos[1] < 0 or start_pos[1] >= self.env.size):
                continue
            
            total_return = 0.0
            valid_episodes = 0
            
            for _ in range(max_episodes):
                # 从当前状态开始生成episode
                x, y = start_pos / self.env.size
                prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
                start_action = np.random.choice(np.arange(self.action_space_size), p=prb.detach().numpy())
                
                try:
                    episode = self.obtain_episode_net(policy_net, start_state, start_action)
                    # 计算episode的总回报
                    episode_return = 0.0
                    for step in range(len(episode)):
                        episode_return += (self.gama ** step) * episode[step]['reward']
                    
                    total_return += episode_return
                    valid_episodes += 1
                except:
                    # 如果episode生成失败，跳过
                    continue
            
            # 计算平均回报作为状态值
            if valid_episodes > 0:
                state_values[start_state] = total_return / valid_episodes
        
        return state_values

    # 其中一个episode就是一个从start到target的轨迹。
    def obtain_episode_net(self, policy_net, start_state, start_action):
        """
        使用策略网络生成一个完整的episode（轨迹）
        
        这是REINFORCE算法的核心组件之一，用于收集训练数据。
        
        :param policy_net: 策略神经网络，用于根据当前状态选择动作
        :param start_state: 起始状态索引（0-24）
        :param start_action: 起始动作索引（0-4）
        :return: episode列表，每个元素包含：
                {"state": s_t,           # 当前状态
                 "action": a_t,          # 当前动作  
                 "reward": r_{t+1},      # 执行动作后获得的即时奖励
                 "next_state": s_{t+1},  # 下一状态
                 "next_action": a_{t+1}} # 下一动作
        
        Episode生成流程：
        1. 设置智能体初始位置
        2. 重复以下步骤直到到达终止状态：
           a. 在环境中执行当前动作
           b. 观察奖励和下一状态
           c. 使用策略网络选择下一动作
           d. 记录这一步的经验
        3. 返回完整的轨迹
        
        注意：这是Monte Carlo方法的关键特征，必须等待episode完成才能进行学习更新
        """
        # 设置智能体在环境中的初始位置
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []  # 存储episode中所有步骤的经验
        next_action = start_action
        next_state = start_state
        terminated = False  # episode终止标志
        
        # 执行episode直到到达终止条件（通常是到达目标状态）
        while not terminated:
            # 当前时刻的状态和动作
            state = next_state
            action = next_action
            
            # 在环境中执行动作，获取奖励和新状态
            _, reward, terminated, _, _ = self.env.step(action)
            
            # 获取执行动作后智能体的新位置对应的状态索引
            next_state = self.env.pos2state(self.env.agent_location)
            
            # 将新状态的坐标归一化到[0,1]区间，作为神经网络的输入
            x, y = self.env.state2pos(next_state) / self.env.size
            
            # 使用策略网络计算在新状态下各动作的选择概率
            prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
            
            # 根据概率分布随机选择下一动作（策略采样）
            next_action = np.random.choice(np.arange(self.action_space_size), p=prb.detach().numpy())
            
            # 将这一步的完整经验添加到episode中
            episode.append({"state": state, 
                          "action": action, 
                          "reward": reward, 
                          "next_state": next_state,
                          "next_action": next_action})
        
        return episode

    def reiniforce(self, epochs=20000):
        """
        REINFORCE算法的主要训练循环
        
        实现经典的蒙特卡洛策略梯度算法：
        1. 采样完整episode
        2. 计算累积回报（Return） 
        3. 计算策略梯度
        4. 更新网络参数
        
        :param epochs: 训练轮数，每轮生成一个episode并更新一次参数
        
        REINFORCE算法流程：
        For each epoch:
            1. 根据当前策略π(a|s,θ)生成episode: {s₀,a₀,r₁,s₁,a₁,r₂,...,s_T}
            2. 对episode中每个时刻t，计算累积回报: G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k+1}
            3. 计算策略梯度: ∇θ J(θ) ≈ Σ_t ∇θ log π(a_t|s_t,θ) * G_t
            4. 更新参数: θ ← θ + α * ∇θ J(θ)
            
        关键技术细节：
        - 使用反向遍历episode来高效计算累积回报
        - 对过短episode进行惩罚以避免不良行为
        - 记录训练过程中的关键指标用于监控
        """
        # ============ 初始化策略网络和优化器 ============
        policy_net = PolicyNet()  # 创建策略神经网络
        # 使用Adam优化器，学习率为self.alpha
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.alpha)
        
        # ============ 主训练循环 ============
        for epoch in range(epochs):
            # 初始化损失变量
            total_loss = 0.0
            
            # ============ Episode采样阶段 ============
            # 使用当前策略在起始状态(0,0)选择初始动作
            prb = policy_net(torch.tensor((0, 0)).reshape(-1, 2))[0]
            print("epoch:{} , prb:{}".format(epoch, prb))
            
            # 根据起始状态的动作概率分布采样初始动作
            start_action = np.random.choice(np.arange(self.action_space_size), p=prb.detach().numpy())
            
            # 生成一个完整的episode
            episode = self.obtain_episode_net(policy_net, start_state=0, start_action=start_action)
            
            # ============ 回报计算和策略更新阶段 ============
            # 对过短的episode进行惩罚，鼓励智能体寻找有效路径
            if len(episode) < 10:
                g = -100  # 惩罚过短的episode
            else:
                g = 0  # 正常episode从0开始累积
            
            # 清零之前计算的梯度
            optimizer.zero_grad()
            
            # ============ 反向遍历episode计算累积回报和梯度 ============
            # REINFORCE的关键：从episode末尾开始反向计算累积回报
            for step in reversed(range(len(episode))):
                # 获取当前步骤的奖励、状态和动作
                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                
                # 调试信息：对于过长的episode打印回报信息
                if len(episode) > 1000:
                    # print(g, reward)  # 可选的调试输出
                    pass
                
                # 计算累积回报：G_t = γ*G_{t+1} + r_{t+1}
                # 这是动态规划的方式，从后往前计算每个时刻的累积回报
                g = self.gama * g + reward
                
                # 存储计算得到的Q值（实际上是累积回报）
                self.qvalue[state, action] = g
                
                # ============ 策略梯度计算 ============
                # 将状态坐标归一化作为网络输入
                x, y = self.env.state2pos(state) / self.env.size
                
                # 前向传播获取当前策略下各动作的概率
                prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
                
                # 计算所选动作的对数概率：log π(a_t|s_t,θ)
                log_prob = torch.log(prb[action])
                
                # 计算策略梯度的损失：L = -log π(a_t|s_t,θ) * G_t
                # 负号是因为PyTorch默认最小化损失，而我们要最大化期望回报
                step_loss = -log_prob * g
                total_loss += step_loss.item()
                
                # 反向传播计算梯度：∇θ log π(a_t|s_t,θ)
                step_loss.backward()
            
            # ============ 记录训练指标 ============
            # 使用TensorBoard记录训练过程中的关键指标
            self.writer.add_scalar("loss", total_loss, epoch)  # 总损失值
            self.writer.add_scalar('g', g, epoch)  # 累积回报
            self.writer.add_scalar('episode_length', len(episode), epoch)  # episode长度
            
            # 可选的控制台输出
            # print(epoch, len(episode), g)
            
            # ============ 参数更新 ============
            # 应用计算得到的梯度更新网络参数：θ ← θ + α * ∇θ J(θ)
            optimizer.step()
        
        # ============ 训练完成后提取最终策略 ============
        # 将训练好的神经网络策略转换为策略表格形式，便于可视化
        for s in range(self.state_space_size):
            x, y = self.env.state2pos(s) / self.env.size
            prb = policy_net(torch.tensor((x, y)).reshape(-1, 2))[0]
            self.policy[s, :] = prb.detach().numpy()  # 添加.detach().numpy()以正确复制
        
        # ============ 计算状态值函数 ============
        print("正在计算状态值函数...")
        self.state_value = self.compute_state_values(policy_net, max_episodes=50)
        
        # 关闭TensorBoard日志记录器
        self.writer.close()

if __name__ == '__main__':
    """
    主程序：运行REINFORCE算法解决网格世界导航问题
    
    实验设置：
    - 环境：5x5网格世界
    - 目标位置：[2,3]（第3行第4列，0-indexed）
    - 禁区位置：[[1,1],[2,1],[2,2],[1,3],[3,3],[1,4]]
    - 起始位置：[0,0]（左上角）
    - 奖励结构：到达目标+1，撞墙或禁区-10，普通移动0
    
    执行流程：
    1. 创建网格世界环境
    2. 初始化REINFORCE求解器
    3. 执行策略梯度训练
    4. 可视化学习到的策略和状态值
    
    预期结果：
    - 智能体学会从起始位置导航到目标位置
    - 策略箭头指向能够到达目标的方向
    - 避开禁区和边界
    """
    # ============ 环境初始化 ============
    # 创建5x5网格世界环境，设置目标位置和禁区
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    
    # ============ 算法初始化 ============
    # 创建REINFORCE求解器，设置学习率
    solver = REINFORCE(alpha=0.001, env=gird_world)
    
    # ============ 训练过程 ============
    start_time = time.time()  # 记录训练开始时间
    
    # 执行REINFORCE算法训练（默认20000个episodes）
    solver.reiniforce(epochs=250)
    
    # 打印状态值（在REINFORCE中主要用于调试）
    print("solver.state_value:", solver.state_value)
    
    # 计算并显示训练耗时
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:", cost_time)
    
    # ============ 结果可视化 ============
    # 显示学习到的策略（动作箭头）
    solver.show_policy()
    
    # 显示状态值函数
    solver.show_state_value(solver.state_value, y_offset=0.25)
    
    # 渲染最终的网格世界图像
    solver.env.render()
