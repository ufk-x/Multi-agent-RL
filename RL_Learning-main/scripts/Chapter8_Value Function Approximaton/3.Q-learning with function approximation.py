import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter用于记录训练日志

# 引用上级目录
import sys
import os
# 添加scripts目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import grid_env

"""
Q-learning with Function Approximation (带函数近似的Q-learning算法)

算法简介：
    Q-learning是一种off-policy时序差分(TD)强化学习算法，名称来源于Q函数（动作值函数）。
    Q-learning直接学习最优动作值函数Q*(s,a)，而不依赖于特定的策略。

    本文件实现了结合线性函数近似的Q-learning算法，主要特点：
    1. 使用线性函数近似Q值：Q(s,a,w) ≈ φ(s,a)^T * w
    2. 其中φ(s,a)是状态-动作对的特征向量，w是权重向量
    3. 通过TD(0)方法更新权重：w ← w + α[r + γ * max_a' Q(s',a',w) - Q(s,a,w)]φ(s,a)
    4. 其中TD误差 δ = r + γ * max_a' Q(s',a') - Q(s,a)

算法流程：
    1. 初始化权重向量w
    2. 对每个episode：
        a. 初始化状态s，使用ε-greedy策略选择动作a
        b. 执行动作a，观察奖励r和下一状态s'
        c. 计算TD误差：δ = r + γ * max_a' Q(s',a') - Q(s,a)
        d. 更新权重：w ← w + α * δ * φ(s,a)
        e. 更新策略为基于当前Q值的ε-greedy策略
        f. s ← s'
    3. 返回学习到的策略

与SARSA的区别：
    - SARSA是on-policy：使用实际执行的下一动作a'来更新
    - Q-learning是off-policy：使用最优的下一动作max_a' Q(s',a')来更新
    - Q-learning更激进，学习最优策略；SARSA学习当前策略下的值函数

特征向量设计：
    - 状态特征：使用多项式基函数[1, x, y, x^2, xy, y^2, ...]
    - 动作特征：使用one-hot编码表示动作
    - 最终特征：将状态特征和动作特征拼接

优势：
    - 可扩展到大规模状态-动作空间
    - 参数数量远小于表格方法
    - 具有泛化能力，相似状态-动作对有相似Q值
    - off-policy特性使其可以从任何策略生成的经验中学习
"""

class Q_learning_with_function_approximation():
    """
    Q-learning with Function Approximation 类

    注意：虽然类名是Sarsa，但实际实现的是Q-learning算法。
    实现带线性函数近似的Q-learning，用于求解网格世界问题。
    """
    def __init__(self, alpha, env):
        """
        初始化Q-learning求解器

        :param alpha: 学习率α，控制权重更新的步长，通常取值0.01-0.5
        :param env: 网格世界环境对象实例，包含状态转移、奖励等信息
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
        # 状态值函数V(s)，虽然Q-learning学习Q值，但可用于可视化
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
                policy = self.policy[state, action]
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
                {\"state\": s_t,           # 当前状态
                 \"action\": a_t,          # 当前动作
                 \"reward\": r_{t+1},      # 执行动作后获得的即时奖励
                 \"next_state\": s_{t+1},  # 下一状态
                 \"next_action\": a_{t+1}} # 下一动作

        注：此函数主要用于离线生成训练数据或评估策略，在实际Q-learning训练中
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
        生成状态-动作对的特征向量（用于函数逼近）

        :param fourier: 是否使用傅里叶特征函数（True）还是多项式特征函数（False）
        :param state: 状态索引（0-24，对应5x5网格）
        :param action: 动作索引（0-4，对应上下左右停留）
        :param ord: 特征函数的最高阶次数（多项式阶数或傅里叶阶数）
        :return: 特征向量，形状为(feature_dim,)，包含状态特征和动作特征的拼接

        特征向量设计：
        1. 状态特征（位置相关）：
           - 傅里叶特征：使用cos函数，基于位置的正弦余弦变换
             φ_fourier(x,y) = [cos(π*i*x_norm + π*j*y_norm) for i,j in range(ord+1)]
           - 多项式特征：使用多项式基函数
             φ_poly(x,y) = [x^i * y^j for i+j <= ord]

        2. 动作特征：one-hot编码（5维，对应5个动作）

        3. 最终特征向量：状态特征 + 动作特征拼接

        这种设计允许Q函数学习状态和动作的复杂非线性关系
        """
        if state < 0 or state >= self.state_space_size or action>= self.action_space_size:
            raise ValueError("Invalid state value")

        x, y = self.env.state2pos(state) + (1, 1)
        feature_vector = []


        if fourier is True:
            # 傅里叶特征向量：使用余弦函数进行位置编码
            # 将位置归一化到[0,1]区间
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(
                            np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized)))
            
            return np.array(feature_vector)
        else:
            # 多项式特征向量：使用多项式基函数
            # 将数据中心化到[0, size-1]区间的中心，确保归一化后数据平衡分布
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            # 初始化特征向量，常数项为1
            feature_vector = [1]  # 特征向量的第一个元素总是常数1
            for i in range(1, ord + 1):  # 从1到ord阶
                for j in range(i + 1):  # j表示y的指数，i-j表示x的指数
                    feature_vector.append((x_normalized ** (i - j)) * (y_normalized ** j))  #[1, x, y, x^2, xy, y^2]
            feature_vector = np.array(feature_vector)
            
            # 动作特征向量：使用one-hot编码表示（对所有特征类型都适用）
            action_feature_vector = np.zeros(self.action_space_size)
            action_feature_vector[action] = 1
            
            # 将状态特征和动作特征拼接在一起
            return np.concatenate((feature_vector, action_feature_vector), axis=0)

    def epsilon_greedy_policy(self, state: int, w: np.ndarray, ord: int, epsilon: float):
        """
        使用ε-greedy策略选择动作（平衡探索与利用）

        :param state: 当前状态索引
        :param w: 当前的权重向量（用于计算Q值）
        :param ord: 多项式特征的阶数（用于特征向量生成）
        :param epsilon: 探索概率ε（0到1之间）
        :return: 选择的动作索引（0-4）

        ε-greedy策略原理：
        - 以概率ε进行随机探索（uniform随机选择动作）
        - 以概率1-ε进行贪婪利用（选择当前估计Q值最大的动作）

        这种策略确保了算法既能利用已学知识，又能探索未知动作空间
        """
        if np.random.rand() < epsilon:
            # 随机选择动作（探索）：以均匀概率选择任一动作
            return np.random.choice(self.action_space_size)
        else:
            # 根据Q值选择最优动作（利用）：计算所有动作的Q值，选择最大者
            q_values = np.zeros(self.action_space_size)
            for action in range(self.action_space_size):
                feature_vector_sa = self.get_feature_vector_with_action(True, state, action, ord= ord)
                q_values[action] = np.dot(feature_vector_sa, w)
            return np.argmax(q_values)

    def update_policy(self, state: int, w: np.ndarray, ord: int, epsilon: float):
        """
        根据当前Q值估计更新ε-greedy策略

        :param state: 要更新策略的状态索引
        :param w: 当前的权重向量（用于计算Q值）
        :param ord: 多项式特征的阶数
        :param epsilon: 探索概率ε

        更新规则：
        - 计算当前状态下所有动作的Q值
        - 找到Q值最大的动作（贪婪动作）
        - 将策略设置为ε-greedy形式：
          * 贪婪动作的概率 = 1 - ε + ε/|A|
          * 其他动作的概率 = ε/|A|

        这种更新确保策略始终反映最新的Q值估计
        """
        # 计算当前状态的所有动作的 Q 值
        q_values = np.zeros(self.action_space_size)
        for action in range(self.action_space_size):
            feature_vector_sa = self.get_feature_vector_with_action(True, state, action, ord= ord)
            q_values[action] = np.dot(feature_vector_sa, w)

        # 找到 Q 值最大的动作
        best_action = np.argmax(q_values)

        # 更新策略为ε-greedy
        for a in range(self.action_space_size):
            if a == best_action:
                self.policy[state, a] = 1 - epsilon + (epsilon / self.action_space_size)
            else:
                self.policy[state, a] = epsilon / self.action_space_size

    def get_max_q_value(self, state: int, w: np.ndarray, ord: int) -> float:
        """
        计算给定状态下的最大Q值（用于Q-learning的TD目标）

        :param state: 状态索引
        :param w: 当前权重向量
        :param ord: 多项式特征阶数
        :return: 该状态下所有动作的最大Q值

        计算过程：
        1. 对该状态的每个可能动作，计算Q(s,a) = φ(s,a)^T * w
        2. 返回所有Q值中的最大值：max_a Q(s,a)

        这个最大Q值用于Q-learning的TD目标计算：r + γ * max_a' Q(s',a')
        """
        q_values = np.zeros(self.action_space_size)
        for action in range(self.action_space_size):
            feature_vector_sa = self.get_feature_vector_with_action(True, state, action, ord)
            q_values[action] = np.dot(feature_vector_sa, w)
        return np.max(q_values)  # 返回最大的Q值

    '''
        Learn an optimal policy that can lead the agent to the target state from an initial state s0.
        add approximation method into Sarsa
    '''
    def Q_learning_alg_with_approximation(self,initial_location, epsilon, learning_rate, ord, max_episode_num):
        """
        使用函数逼近的Q-learning算法学习最优策略

        Q-learning算法原理：
        - 离线策略TD学习：使用行为策略采样，使用目标策略（贪婪策略）更新
        - TD目标：r + γ * max_a' Q(s',a') （不依赖于实际执行的下一动作）
        - 更新规则：Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]

        函数逼近扩展：
        - Q(s,a) ≈ φ(s,a)^T * w
        - 参数更新：w ← w + α * δ * φ(s,a)，其中δ为TD误差

        :param initial_location: 智能体起始位置 [x,y]
        :param epsilon: ε-greedy探索率（默认0.1）
        :param learning_rate: 学习率α（默认0.001）
        :param ord: 多项式特征阶数（默认5）
        :return: (total_rewards, episode_lengths, final_w) - 每个episode的总奖励、长度和最终权重向量

        算法流程：
        1. 初始化权重向量w（随机高斯分布）
        2. 对每个episode重复：
           a. 重置环境到起始状态
           b. 使用ε-greedy策略选择初始动作
           c. 重复直到终止：
              - 执行动作，观察奖励和下一状态
              - 使用ε-greedy选择下一动作（仅用于采样，不用于更新）
              - 计算TD误差：δ = r + γ*max_a' Q(s',a') - Q(s,a)
              - 更新权重：w ← w + α*δ*φ(s,a)
              - 更新策略为基于新权重的ε-greedy
              - 转移到下一状态-动作对
        """
        initial_state = self.env.pos2state(initial_location)
        print("initial_state:", initial_state)
        total_rewards, episode_lengths = [],[]
        dim = len(self.get_feature_vector_with_action(True, 0, 0, ord))  # 确定特征向量的维度
        w = np.random.default_rng().normal(size=dim, scale=0.1)  # 使用较小的初始权重，提高稳定性
        
        # 添加学习率衰减机制
        initial_learning_rate = learning_rate
        for episode_num in range(max_episode_num): # episode_num
            # 学习率衰减：随着训练进行逐渐减小学习率
            current_lr = initial_learning_rate * (0.99 ** (episode_num // 50))
            
            self.env.reset()
            self.env.agent_location = initial_location.copy()
            total_reward = 0
            episode_length = 0
            done = False
            if episode_num % 50 == 0:  # 每50个episode打印一次
                print(f"Episode {episode_num}, LR: {current_lr:.4f}")
            state = initial_state
            # action = np.random.choice(a=np.arange(self.action_space_size),
            #                           p=self.policy[state, :])  # Generate a0 at s0 following π0(s0)
            #initialize buffers
            # states = [state]
            # aciton = [action]
            # rewards = [0]
            while not done:  #If s_t is not the target state, do
                episode_length += 1
                action = self.epsilon_greedy_policy(state, w, ord = ord, epsilon = epsilon)
                observation, reward, done, _, _ = self.env.step(action) #Collect an experience sample (rt+1, st+1, at+1)
                # print("observation:", observation)
                #S
                next_state = self.env.pos2state(self.env.agent_location)
                # print("next_state:",next_state, "self.env.agent_location:",self.env.agent_location)
                #A
                # next_action = np.random.choice(np.arange(self.action_space_size),
                #                                p=self.policy[next_state,:])
                total_reward += reward

                feature_vector_sa = self.get_feature_vector_with_action(True, state, int(action), ord)

                #Value update(parameter update):
                # q(s, a, w) = φ^T (s, a)*w,
                # ∇w q(s, a, w) = φ(s, a).
                #q_gradient 为feature vector
                # 计算当前和下一个状态-动作对的 Q 值
                q_sa = np.dot(feature_vector_sa, w)
                
                # Q-learning关键修正：正确处理终止状态
                if done:
                    # 终止状态的下一状态Q值为0（没有后续状态）
                    max_q_next_state = 0.0
                else:
                    # 非终止状态：使用下一状态的最大Q值
                    max_q_next_state = self.get_max_q_value(next_state, w, ord)

                # Q-learning with function approximation
                td_error = reward + self.gama * max_q_next_state - q_sa
                w = w + current_lr * td_error * feature_vector_sa

                # print("dimension feature_vector_sa", np.shape(feature_vector_sa))
                # print("dimension w:",np.shape(w))

                #update policy
                self.update_policy(state, w, ord=ord, epsilon=epsilon)

                state = next_state
            
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        print(f"Training completed.")
        # 返回训练结果和最终权重
        return total_rewards, episode_lengths, w

    def compute_state_values(self, w: np.ndarray, ord: int):
        """
        从学习到的Q值权重计算状态值函数V(s)

        :param w: 训练得到的权重向量
        :param ord: 多项式特征阶数

        在Q-learning中，状态值函数V(s) = max_a Q(s,a)
        由于我们使用函数逼近,Q(s,a) = φ(s,a)^T * w
        因此V(s) = max_a [φ(s,a)^T * w]

        这个函数将计算所有状态的状态值，用于可视化
        """
        for state in range(self.state_space_size):
            # 计算当前状态下所有动作的Q值
            q_values = np.zeros(self.action_space_size)
            for action in range(self.action_space_size):
                feature_vector_sa = self.get_feature_vector_with_action(True, state, action, ord)
                q_values[action] = np.dot(feature_vector_sa, w)
            
            # 状态值等于该状态下所有Q值的最大值（贪婪策略下的期望回报）
            self.state_value[state] = np.max(q_values)

    def update_final_policy(self, w: np.ndarray, ord: int):
        """
        基于最终学习到的Q值计算确定性贪婪策略用于可视化

        :param w: 训练完成后的权重向量
        :param ord: 多项式特征阶数

        计算每个状态下的最优动作，并将策略设置为确定性贪婪策略：
        - 最优动作的概率 = 1.0
        - 其他动作的概率 = 0.0

        这样可以清晰地显示学习到的最优策略方向
        """
        # 调试信息：打印一些关键状态的Q值
        for state in range(self.state_space_size):
            # 计算当前状态下所有动作的Q值
            q_values = np.zeros(self.action_space_size)
            for action in range(self.action_space_size):
                feature_vector_sa = self.get_feature_vector_with_action(True, state, action, ord)
                q_values[action] = np.dot(feature_vector_sa, w)
            
            # 调试信息：打印一些关键状态的Q值
            if state == 0 or state == 17 or state == 12:  # 起始、目标、和中心状态
                pos = self.env.state2pos(state)
                print(f"State {state} at pos {pos}: Q-values = {q_values}")
                print(f"State {state}: Best action = {np.argmax(q_values)}")
                
                # 检查是否是目标状态
                target_pos = self.env.target_location
                print(f"  Target location: {target_pos}")
                print(f"  Is target: {np.array_equal(pos, target_pos)}")
                
            # 找到Q值最大的动作（如果有多个最优动作，选择第一个）
            best_action = np.argmax(q_values)
            
            # 设置为确定性贪婪策略
            self.policy[state, :] = 0.0  # 先清零所有动作概率
            self.policy[state, best_action] = 1.0  # 最优动作概率设为1

def main():
    """
    主函数：运行Q-learning函数逼近算法并可视化结果

    实验设置：
    - 环境：5x5网格世界
    - 目标位置：[2,3]（第3行第4列，0-indexed）
    - 禁区位置：[[1,1],[2,1],[2,2],[1,3],[3,3],[1,4]]
    - 奖励结构：到达目标+1，撞墙或禁区-10，普通移动0
    - 起始位置：[0,0]（左上角）

    执行流程：
    1. 初始化网格世界环境和求解器
    2. 运行Q-learning算法训练1000个episodes
    3. 记录训练时间和轨迹长度
    4. 显示训练后的策略（ε-greedy形式）
    5. 可视化策略（动作箭头）和状态值函数
    6. 绘制训练曲线：总奖励和episode长度随训练进行的变化
    """
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    solver = Q_learning_with_function_approximation(alpha=0.01, env=gird_world)
    # solver.sarsa()
    # print("env.policy[0, :]:",solver.policy[0, :])
    # for _ in range(20):
    #     a0 = np.random.choice(5, p=solver.policy[0, :] )
    #
    #     print("a0:",a0)

    start_time = time.time()

    initial_location = [0,0]
    value_estimation_ord = 3
    ## 主要训练过程
    total_rewards, episode_lengths, final_w = solver.Q_learning_alg_with_approximation(
        initial_location=initial_location, 
        epsilon=0.15,  # 傅里叶特征需要适度探索
        learning_rate=0.003,  # 傅里叶特征使用更小的学习率
        ord=value_estimation_ord,  # 傅里叶特征使用3阶
        max_episode_num=300  # 增加训练轮数
    )

    # 训练完成后，从学习到的Q值计算状态值函数用于可视化
    solver.compute_state_values(final_w, ord=value_estimation_ord)
    
    # 计算最终的贪婪策略用于策略箭头显示
    solver.update_final_policy(final_w, ord=value_estimation_ord)
    
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:",cost_time)
    print(len(gird_world.render_.trajectory))

    initial_state = solver.env.pos2state(np.array(initial_location))
    print("训练后的policy结果为:\n",solver.policy[initial_state,:])
    print("计算得到的状态值范围: [{:.2f}, {:.2f}]".format(np.min(solver.state_value), np.max(solver.state_value)))
    
    # 显示一些关键状态的策略以验证学习结果
    print("关键状态的学习策略:")
    target_state = solver.env.pos2state(np.array([2,3]))  # 目标状态
    print(f"目标状态[2,3] (state {target_state}): {solver.policy[target_state,:]}")
    print(f"起始状态[0,0] (state {initial_state}): {solver.policy[initial_state,:]}")
    
    # 解释动作含义
    action_names = ["上", "右", "下", "左", "停留"]
    best_action_initial = np.argmax(solver.policy[initial_state,:])
    print(f"起始状态最优动作: {action_names[best_action_initial]}")
    
    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    # gird_world.plot_title("Episode_length = " + str(i))
    gird_world.render()
    # gird_world.render_clear()
    print("--------------------")
    print("Plot")
    # 绘制第一个图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(total_rewards) + 1), total_rewards,   # 空心，设置填充色为透明
             markeredgecolor='blue',  # 边框颜色为蓝色
             markersize=10,
             linestyle='-', color='blue',label = "total_rewards")
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('total_rewards', fontsize=12)

    # 绘制第二个图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths,  # 空心，设置填充色为透明
             markeredgecolor='blue',  # 边框颜色为蓝色
             markersize=10,
             linestyle='-', color='blue',label = "episode_length")
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('episode_length', fontsize=12)

    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()

if __name__ =="__main__":
    main()