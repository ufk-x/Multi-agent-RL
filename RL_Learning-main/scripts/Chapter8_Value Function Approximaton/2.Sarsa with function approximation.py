import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter用于记录训练日志

# 引用上级目录
import sys
# sys.path.append("..")
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import grid_env

"""
SARSA with Function Approximation (带函数近似的SARSA算法)

算法简介：
    SARSA是一种on-policy时序差分(TD)强化学习算法，名称来源于其更新序列：
    State -> Action -> Reward -> State -> Action
    
    本文件实现了结合线性函数近似的SARSA算法，主要特点：
    1. 使用线性函数近似Q值：Q(s,a) ≈ φ(s,a)^T * w
    2. 其中φ(s,a)是状态-动作对的特征向量，w是权重向量
    3. 通过TD(0)方法更新权重：w ← w + α * δ * φ(s,a)
    4. 其中TD误差 δ = r + γ * Q(s',a') - Q(s,a)
    
算法流程：
    1. 初始化权重向量w
    2. 对每个episode：
        a. 初始化状态s，使用ε-greedy策略选择动作a
        b. 执行动作a，观察奖励r和下一状态s'
        c. 在s'使用ε-greedy策略选择下一动作a'
        d. 计算TD误差：δ = r + γ * φ(s',a')^T*w - φ(s,a)^T*w
        e. 更新权重：w ← w + α * δ * φ(s,a)
        f. 更新策略为基于当前Q值的ε-greedy策略
        g. s ← s', a ← a'
    3. 返回学习到的策略

特征向量设计：
    - 状态特征：使用多项式基函数[1, x, y, x^2, xy, y^2, ...]
    - 动作特征：使用one-hot编码表示动作
    - 最终特征：将状态特征和动作特征拼接
    
优势：
    - 可扩展到大规模状态-动作空间
    - 参数数量远小于表格方法
    - 具有泛化能力，相似状态-动作对有相似Q值
"""
class Sarsa():
    """
    SARSA with Function Approximation 类
    
    实现带线性函数近似的SARSA算法，用于求解网格世界问题。
    """
    def __init__(self, alpha, env=grid_env.GridEnv):
        """
        初始化SARSA求解器
        
        :param alpha: 学习率α，控制权重更新的步长，通常取值0.01-0.5
        :param env: 网格世界环境对象，包含状态转移、奖励等信息
        """
        # ============ 基本参数设置 ============
        self.gama = 0.9  # 折扣因子γ，决定未来奖励的权重，越接近1越重视长期回报
        self.alpha = alpha  # 学习率α，控制每次更新的步长
        self.env = env  # 环境对象
        
        # ============ 状态和动作空间 ============
        self.action_space_size = env.action_space_size  # 动作空间大小，通常为5（上下左右+停留）
        self.state_space_size = env.size ** 2  # 状态空间大小，对于5x5网格为25
        
        # ============ 奖励设置 ============
        # reward_list通常为[other, target, forbidden, overflow]对应的奖励值
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # 例如：[-10,-10,0,1]
        
        # ============ 值函数和策略初始化 ============
        # 状态值函数V(s)，虽然SARSA直接学习Q值，但可用于可视化
        self.state_value = np.zeros(shape=self.state_space_size)  # 形状：(25,)
        print("self.state_value:", self.state_value)
        
        # Q值表Q(s,a)，用于存储学习到的Q值（虽然使用函数近似，但仍保留用于对比）
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 形状：(25, 5)
        
        # 平均策略：每个状态下所有动作概率均等，用作初始策略
        self.mean_policy = np.ones(
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        
        # 当前策略π(a|s)，初始化为平均策略，训练过程中会逐步优化为ε-greedy策略
        self.policy = self.mean_policy.copy()
        
        # ============ 日志记录 ============
        self.writer = SummaryWriter("logs")  # TensorBoard日志写入器，用于可视化训练过程

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
        """
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                # 获取在当前状态下选择该动作的概率
                policy = self.policy[state, action]
                # 绘制箭头：位置、方向（概率加权）、半径（概率相关）
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        """
        在网格世界中显示状态值
        
        :param state_value: 要显示的状态值数组，形状为(state_space_size,)
        :param y_offset: 文本在Y轴上的偏移量，用于在同一格子显示多个值时避免重叠
        
        在每个状态位置显示其对应的值函数数值，便于观察值函数的分布
        """
        for state in range(self.state_space_size):
            # 在每个状态位置显示该状态的值（保留1位小数）
            self.env.render_.write_word(pos=self.env.state2pos(state), 
                                        word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)  # 字体缩小以适应格子

    def obtain_episode(self, policy, start_state, start_action, length):
        """
        根据给定策略生成一个episode（轨迹）
        
        :param policy: 策略矩阵，形状为(state_space_size, action_space_size)
                      policy[s, a]表示在状态s下选择动作a的概率
        :param start_state: 起始状态索引
        :param start_action: 起始动作索引
        :param length: episode的长度（步数），即采样多少个(s,a,r,s',a')五元组
        :return: episode列表，每个元素是包含以下键的字典：
                {"state": s_t,           # 当前状态
                 "action": a_t,          # 当前动作
                 "reward": r_{t+1},      # 执行动作后获得的即时奖励
                 "next_state": s_{t+1},  # 下一状态
                 "next_action": a_{t+1}} # 下一动作
        
        注：此函数主要用于离线生成训练数据或评估策略，在实际SARSA训练中
            通常是在线采样而不预先生成完整episode
        """
        # 设置智能体初始位置
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []  # 存储episode的列表
        next_action = start_action
        next_state = start_state
        
        # 生成指定长度的轨迹
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            
            # 在环境中执行动作，获取奖励和是否终止的信息
            _, reward, done, _, _ = self.env.step(action)
            
            # 获取执行动作后的下一状态
            next_state = self.env.pos2state(self.env.agent_location)
            
            # 根据策略在下一状态采样下一动作
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            
            # 将这一步的经验(s,a,r,s',a')添加到episode中
            episode.append({"state": state, 
                           "action": action, 
                           "reward": reward, 
                           "next_state": next_state,
                           "next_action": next_action})
        
        return episode  # 返回完整的episode列表
    def get_feature_vector_with_action(self, fourier: bool, state: int, action: int, ord: int) -> np.ndarray:
        """
        生成状态-动作对的特征向量 φ(s, a)
        
        这是函数近似的核心：将(状态,动作)对映射到特征空间，用于线性近似Q值。
        Q(s,a,w) = φ(s,a)^T * w，其中w是权重向量
        
        :param fourier: 是否使用傅里叶基函数
                       - True: 使用傅里叶基 cos(πk·x)，适合周期性或平滑函数
                       - False: 使用多项式基 [1, x, y, x², xy, y², ...]，适合网格世界
        :param state: 状态索引（0到state_space_size-1）
        :param action: 动作索引（0到action_space_size-1）
        :param ord: 特征函数的最高阶数
                   - 对于多项式：ord=2生成[1, x, y, x², xy, y²]
                   - 对于傅里叶：ord=2生成(ord+1)²=9个基函数
        :return: 特征向量，形状为(feature_dim,)
                包含状态特征和动作特征的拼接
        
        设计思想：
        1. 状态特征：捕获智能体在网格中的位置信息
        2. 动作特征：使用one-hot编码区分不同动作
        3. 拼接后的特征既包含"在哪里"也包含"做什么"的信息
        """
        # ============ 输入验证 ============
        if state < 0 or state >= self.state_space_size or action >= self.action_space_size:
            raise ValueError("Invalid state or action value")

        # ============ 获取状态的网格坐标 ============
        # state2pos将状态索引转换为(x, y)坐标，+1是为了避免0坐标（便于归一化）
        x, y = self.env.state2pos(state) + (1, 1)
        feature_vector = []

        # ============ 生成状态特征向量 ============
        if fourier is not True:
            # -------- 傅里叶基函数特征 --------
            # 使用 φ_{i,j}(x,y) = cos(π(i·x_norm + j·y_norm))
            # 优点：对平滑函数近似效果好，频率可调
            
            # 归一化坐标到 [0, 1] 区间
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            
            # 生成所有(i,j)组合的傅里叶基函数，总共(ord+1)²个
             # 傅里叶基函数，包含状态和动作
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(
                            np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized)))
            
            return np.array(feature_vector)

        else:
            # -------- 多项式基函数特征 --------
            # 使用 [1, x, y, x², xy, y², x³, x²y, xy², y³, ...]
            # 优点：简单直观，适合网格世界这种离散环境
            
            # 归一化并中心化坐标到 [-1, 1] 区间
            # 中心化可以避免数值偏向一端，提高数值稳定性
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            
            # 初始化特征向量，第一个元素是常数项（偏置）
            feature_vector = [1]  # 偏置项，对应w_0
            
            # 生成多项式特征：按阶数递增
            # ord=1: [1, x, y]  (3个特征)
            # ord=2: [1, x, y, x², xy, y²]  (6个特征)
            # ord=3: [1, x, y, x², xy, y², x³, x²y, xy², y³]  (10个特征)
            for i in range(1, ord + 1):  # i是当前阶数
                for j in range(i + 1):  # j是y的指数，则x的指数为(i-j)
                    # 生成 x^(i-j) * y^j 项
                    feature_vector.append((x_normalized ** (i - j)) * (y_normalized ** j))
            
            feature_vector = np.array(feature_vector)
            
            # ============ 生成动作特征向量 ============
            # 使用one-hot编码表示动作，将离散动作转换为向量形式
            # 例如：action=2, action_space_size=5 -> [0, 0, 1, 0, 0]
            action_feature_vector = np.zeros(self.action_space_size)
            action_feature_vector[action] = 1
        
            # ============ 拼接状态特征和动作特征 ============
            # 最终特征向量 = [状态特征, 动作特征]
            # 例如：状态特征6维 + 动作特征5维 = 总共11维特征向量
            # 这样的设计使得Q(s,a)既依赖于状态，也依赖于动作
            return np.concatenate((feature_vector, action_feature_vector), axis=0)

    def epsilon_greedy_policy(self, state: int, w: np.ndarray, ord: int, epsilon: float) -> int:
        """
        使用ε-greedy策略选择动作
        
        ε-greedy策略平衡探索(exploration)和利用(exploitation)：
        - 以概率ε随机选择动作（探索未知区域）
        - 以概率1-ε选择当前Q值最大的动作（利用已知最优动作）
        
        :param state: 当前状态索引
        :param w: 当前的权重向量，用于计算Q(s,a) = φ(s,a)^T * w
        :param ord: 多项式特征的阶数
        :param epsilon: 探索率ε，取值范围[0, 1]
                       - ε=0: 完全贪婪，总是选择最优动作
                       - ε=1: 完全随机，总是随机探索
                       - 通常取0.1-0.3，训练后期可逐渐衰减
        :return: 选择的动作索引
        """
        if np.random.rand() < epsilon:
            # -------- 探索(Exploration) --------
            # 以概率ε随机选择动作，确保每个动作都有机会被尝试
            return np.random.choice(self.action_space_size)
        else:
            # -------- 利用(Exploitation) --------
            # 以概率1-ε选择Q值最大的动作（贪婪选择）
            q_values = np.zeros(self.action_space_size)
            
            # 计算当前状态下所有动作的Q值
            for action in range(self.action_space_size):
                # 获取(state, action)对的特征向量
                feature_vector_sa = self.get_feature_vector_with_action(False, state, action, ord=ord)
                # 计算Q值：Q(s,a) = φ(s,a)^T * w
                q_values[action] = np.dot(feature_vector_sa, w)
            
            # 返回Q值最大的动作（如有多个最大值，返回第一个）
            return np.argmax(q_values)

    def update_policy(self, state: int, w: np.ndarray, ord: int, epsilon: float):
        """
        将策略更新为基于当前Q值的ε-greedy策略
        
        更新后的策略满足：
        - π(a*|s) = 1 - ε + ε/|A|，其中a*是Q值最大的动作
        - π(a|s) = ε/|A|，对于其他动作a ≠ a*
        
        这确保了：
        1. 最优动作有最高的选择概率
        2. 所有动作都有非零概率被选择（满足探索要求）
        3. 所有概率之和为1
        
        :param state: 要更新策略的状态索引
        :param w: 当前的权重向量
        :param ord: 多项式特征的阶数
        :param epsilon: 探索率ε
        
        注：此函数更新self.policy[state, :]，使其反映当前的ε-greedy策略
        """
        # ============ 计算所有动作的Q值 ============
        q_values = np.zeros(self.action_space_size)
        for action in range(self.action_space_size):
            feature_vector_sa = self.get_feature_vector_with_action(False, state, action, ord=ord)
            q_values[action] = np.dot(feature_vector_sa, w)

        # ============ 找到Q值最大的动作（贪婪动作）============
        best_action = np.argmax(q_values)

        # ============ 更新策略为ε-greedy ============
        for a in range(self.action_space_size):
            if a == best_action:
                # 最优动作：获得(1-ε)的贪婪概率 + ε/|A|的探索概率
                self.policy[state, a] = 1 - epsilon + (epsilon / self.action_space_size)
            else:
                # 非最优动作：只获得ε/|A|的探索概率
                self.policy[state, a] = epsilon / self.action_space_size

    def Sarsa_alg_with_approximation(self, initial_location, epsilon=0.1, ord=3):
        """
        使用函数近似的SARSA算法学习最优策略
        
        算法核心：
        1. 使用线性函数近似Q值：Q(s,a,w) = φ(s,a)^T * w
        2. 通过TD(0)更新权重：w ← w + α[r + γQ(s',a',w) - Q(s,a,w)]φ(s,a)
        3. 使用ε-greedy策略平衡探索与利用
        
        :param initial_location: 智能体的初始位置，格式为[x, y]
        :param epsilon: ε-greedy策略的探索率，默认0.1（10%概率随机探索）
        :param learning_rate: 学习率α，控制权重更新步长，默认0.001
        :param ord: 多项式特征的阶数，控制特征向量复杂度，默认5
                   - ord越大，特征维度越高，表达能力越强，但也更容易过拟合
        :return: (total_rewards, episode_lengths)
                - total_rewards: 每个episode的累积奖励列表
                - episode_lengths: 每个episode的长度（步数）列表
        
        训练流程：
        - 运行1000个episode
        - 每个episode从initial_location开始
        - 使用ε-greedy策略选择动作
        - 每步更新权重w和策略π
        - 到达目标状态或达到最大步数时episode结束
        """
        # ============ 初始化 ============
        # 将初始位置转换为状态索引
        initial_state = self.env.pos2state(initial_location)
        print("initial_state:", initial_state)
        
        # 用于记录训练过程的统计信息
        total_rewards, episode_lengths = [], []  # 每个episode的奖励和长度
        
        # ============ 初始化函数近似参数 ============
        # 计算特征向量的维度（状态特征 + 动作特征）
        dim = len(self.get_feature_vector_with_action(0, 0, 0, ord))
        print(f"Feature vector dimension: {dim}")
        
        # 初始化权重向量w，从标准正态分布N(0,1)采样
        # 小的随机初始化有助于打破对称性并加快收敛
        w = np.random.default_rng().normal(size=dim)
        
        # q_hat用于存储近似的Q值（可选，用于调试或可视化）
        q_hat = np.zeros((self.state_space_size, self.action_space_size))
        # ============ 主训练循环：运行多个episode ============
        for episode_num in range(300):
            # -------- Episode初始化 --------
            self.env.reset()  # 重置环境
            self.env.agent_location = initial_location.copy()  # 设置智能体初始位置
            total_reward = 0  # 本episode的累积奖励
            episode_length = 0  # 本episode的步数
            done = False  # 是否到达终止状态
            if episode_num % 100 == 0:
                print("episode_num:", episode_num)
            
            # 从初始状态开始
            state = initial_state
            
            # 使用ε-greedy策略选择初始动作a_0
            # 注：SARSA是on-policy算法，动作必须从当前策略采样
            action = self.epsilon_greedy_policy(state, w, ord=ord, epsilon=epsilon)
            
            # -------- Episode内循环：直到到达目标或超时 --------
            while not done:
                episode_length += 1
                
                # ============ (1) 执行动作，观察结果 ============
                # 在环境中执行动作a_t，获得奖励r_{t+1}和下一状态s_{t+1}
                observation, reward, done, _, _ = self.env.step(action)
                
                # ============ (2) 获取下一状态 s_{t+1} ============
                next_state = self.env.pos2state(self.env.agent_location)
                
                # ============ (3) 选择下一动作 a_{t+1} ============
                # 关键：SARSA在更新前就需要知道下一动作（与Q-learning不同）
                next_action = self.epsilon_greedy_policy(next_state, w, ord=ord, epsilon=epsilon)
                
                # 累积奖励
                total_reward += reward

                # ============ (4) 计算特征向量 ============
                # 获取当前(s_t, a_t)和下一(s_{t+1}, a_{t+1})的特征向量
                feature_vector_sa = self.get_feature_vector_with_action(False, state, action, ord)
                feature_vector_sa_next = self.get_feature_vector_with_action(False, next_state, next_action, ord)
                
                # ============ (5) 计算Q值 ============
                # Q(s_t, a_t) = φ(s_t, a_t)^T * w
                q_sa = np.dot(feature_vector_sa, w)
                # Q(s_{t+1}, a_{t+1}) = φ(s_{t+1}, a_{t+1})^T * w
                q_sa_next = np.dot(feature_vector_sa_next, w)

                # ============ (6) 计算TD误差 ============
                # TD误差：δ = r_{t+1} + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
                # 这是SARSA的核心：使用实际选择的下一动作a_{t+1}来更新
                td_error = reward + self.gama * q_sa_next - q_sa
                
                # ============ (7) 更新权重（梯度下降）============
                # 半梯度SARSA更新规则：
                # w ← w + α * δ * φ(s_t, a_t)
                # 其中：
                # - α是学习率
                # - δ是TD误差
                # - φ(s_t, a_t)是特征向量，也是Q对w的梯度：∇_w Q(s,a,w) = φ(s,a)
                w = w + self.alpha * td_error * feature_vector_sa

                # ============ (8) 更新策略 ============
                # 基于新的权重w更新当前状态的策略为ε-greedy
                # 这确保策略始终反映最新的Q值估计
                self.update_policy(state, w, ord=ord, epsilon=epsilon)

                # ============ (9) 状态转移：S ← S', A ← A' ============
                action = next_action
                state = next_state
            
            # -------- Episode结束，记录统计信息 --------
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        # ============ 训练完成，计算状态值 ============
        # SARSA学习的是Q值，但可以从Q值导出V值用于可视化
        # V(s) = Σ_a π(a|s) * Q(s,a) 或 V(s) = max_a Q(s,a) （贪婪策略）
        print("\n计算状态值...")
        for state in range(self.state_space_size):
            # 计算该状态下所有动作的Q值
            q_values = np.zeros(self.action_space_size)
            for action in range(self.action_space_size):
                feature_vector_sa = self.get_feature_vector_with_action(False, state, action, ord=ord)
                q_values[action] = np.dot(feature_vector_sa, w)
            
            # 使用当前策略计算状态值：V(s) = Σ_a π(a|s) * Q(s,a)
            self.state_value[state] = np.dot(self.policy[state, :], q_values)
            
            # 也更新qvalue表用于调试
            self.qvalue[state, :] = q_values
        
        print(f"状态值计算完成！")
        print(f"状态值范围: [{self.state_value.min():.2f}, {self.state_value.max():.2f}]")
        
        # ============ 返回结果，包括权重w用于后续使用 ============
        return total_rewards, episode_lengths, w


if __name__ == "__main__":
    # ============ 创建环境 ============
    # 创建5x5网格世界，目标位置[2,3]，多个禁区
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')  # 不保存视频
    
    # ============ 创建SARSA求解器 ============
    # alpha=0.1是学习率，控制Q值更新的步长
    solver = Sarsa(alpha=0.01, env=gird_world)

    # ============ 运行SARSA算法 ============
    print("\n" + "="*50)
    print("开始训练SARSA with Function Approximation")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    # 设置初始位置（左上角）
    initial_location = [0, 0]
    
    # 训练：使用函数近似的SARSA算法学习最优策略
    # 默认参数：epsilon=0.1, learning_rate=0.001, ord=5
    total_rewards, episode_lengths, learned_w = solver.Sarsa_alg_with_approximation(
        initial_location=initial_location)

    end_time = time.time()
    cost_time = end_time - start_time
    
    # ============ 输出训练结果 ============
    print("\n" + "="*50)
    print(f"训练完成！耗时：{cost_time:.2f}秒")
    print("="*50)
    print(f"轨迹点数量：{len(gird_world.render_.trajectory)}")
    
    # 查看初始状态的策略分布
    initial_state = solver.env.pos2state(initial_location)
    print(f"\n初始状态{initial_state}的策略分布:")
    print("[上, 右, 下, 左, 停留]")
    print(solver.policy[initial_state, :])
    print(f"最优动作：{np.argmax(solver.policy[initial_state, :])}")
    
    # ============ 可视化结果 ============
    print("\n可视化学习结果...")
    print(f"状态值统计信息:")
    print(f"  最小值: {solver.state_value.min():.2f}")
    print(f"  最大值: {solver.state_value.max():.2f}")
    print(f"  平均值: {solver.state_value.mean():.2f}")
    print(f"  前5个状态的值: {solver.state_value[:5]}")
    
    # 在网格世界中显示学习到的策略（箭头）
    print("\n绘制策略箭头...")
    solver.show_policy()
    
    # 显示状态值（从学习到的Q值导出）
    print("显示状态值...")
    solver.show_state_value(solver.state_value, y_offset=0.25)
    
    # 渲染网格世界
    print("渲染网格世界...")
    gird_world.render()
    
    # ============ 绘制训练曲线 ============
    print("\n" + "="*50)
    print("绘制训练曲线")
    print("="*50)
    
    # 图1：每个episode的累积奖励
    # 期望看到：随着训练进行，奖励逐渐增加（因为智能体学会更快到达目标）
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(total_rewards) + 1), total_rewards,
             markeredgecolor='blue',
             markersize=10,
             linestyle='-', color='blue', label="total_rewards")
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('Total Rewards', fontsize=12)
    plt.title('Total Rewards per Episode', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 图2：每个episode的长度（步数）
    # 期望看到：随着训练进行，episode长度逐渐减少（智能体找到更短路径）
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths,
             markeredgecolor='blue',
             markersize=10,
             linestyle='-', color='blue', label="episode_length")
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.title('Episode Length per Episode', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 显示所有图表
    plt.show()
    
    print("\n训练和可视化完成！")
