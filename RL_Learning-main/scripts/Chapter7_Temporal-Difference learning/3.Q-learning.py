import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
# sys.path.append("..")
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import grid_env

"""
SARSA: State - action - reward - state - action

TD learning of acton values: Sarsa  ->  directly estimate action values.
"""
class Q_learning():
    def __init__(self,alpha,env = grid_env.GridEnv):
        self.gamma = 0.9  # discount rate
        self.alpha = alpha  #learning rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # [-10,-10,0,1]  reward list
        self.state_value = np.zeros(shape=self.state_space_size)  # 一维列表
        print("self.state_value:", self.state_value)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 二维： state数 x action数
        self.mean_policy = np.ones(     #self.mean_policy shape: (25, 5)
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size  # 平均策略，即取每个动作的概率均等
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))

        print('----------------------------------------------------------------')

    def show_policy(self):
        """可视化当前策略
        
        在网格世界中用箭头显示每个状态下各动作的选择概率
        箭头长度和粗细反映了该动作被选择的概率大小
        """
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]  # 获取在state下选择action的概率
                # 在网格中绘制动作箭头，箭头方向表示动作方向，大小表示概率
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        """可视化状态价值函数
        
        在每个网格中显示该状态的价值估计
        状态价值V(s) = max_a Q(s,a)，表示在该状态下能获得的最大预期回报
        
        Args:
            state_value: 状态价值数组
            y_offset: 文字在网格中的y轴偏移量
        """
        for state in range(self.state_space_size):
            # 在每个网格中心位置显示状态价值，保留1位小数
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def q_learning(self, initial_location, epsilon=0.4, n=3):
        """Q-learning算法主函数：学习从初始状态到目标状态的最优策略
        
        Q-learning是一种off-policy TD算法，它学习最优策略，同时使用另一个策略来探索环境。
        
        核心思想：
        1. 使用ε-greedy策略选择动作（行为策略，用于探索）
        2. 使用贪婪策略更新Q值（目标策略，学习最优策略）
        3. Q值更新：Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        与Sarsa的区别：
        - Sarsa：Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]  (使用实际执行的a')
        - Q-learning：Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]  (使用最优a')
        
        Args:
            initial_location: 初始位置坐标，如[0,0]或[4,0]
            epsilon: ε-greedy策略中的探索概率，范围[0,1]
                    以ε的概率随机选择动作（探索），以1-ε的概率选择最优动作（利用）
            n: 未使用的参数（可能是为了接口兼容性保留）
                    
        Returns:
            total_rewards: 每个episode的累计奖励列表，用于评估学习性能
            episode_lengths: 每个episode的长度（步数）列表，用于观察是否找到更短路径
        """
        # 用于记录训练过程的性能指标
        total_rewards = []  # 存储每个episode的总奖励
        episode_lengths = []  # 存储每个episode的步数
        
        # 将位置坐标转换为状态索引
        initial_state = self.env.pos2state(initial_location)
        print("initial_state:", initial_state)

        # 开始训练循环，共1000个episodes
        for episode_num in range(1000):
            self.env.reset()  # 重置环境到初始状态
            # 重要：将智能体放回初始位置（修复：需要添加这一行）
            self.env.agent_location = initial_location
            
            total_reward = 0  # 当前episode的累计奖励
            episode_length = 0  # 当前episode的步数
            done = False  # 是否到达终止状态（目标）
            max_steps = 2000  # 最大步数限制，防止无限循环
            
            # 每100个episode打印一次进度
            if episode_num % 100 == 0:
                print(f"episode_num: {episode_num}")
            
            state = initial_state  # 当前状态
            
            # Episode主循环：持续执行直到到达目标状态或达到最大步数
            while not done and episode_length < max_steps:
                # ========== 1. 动作选择：使用ε-greedy策略 ==========
                # 这是行为策略（behavior policy），用于探索环境
                if np.random.rand() < epsilon:
                    # 探索：以ε的概率随机选择动作
                    action = np.random.choice(self.action_space_size)
                else:
                    # 利用：以1-ε的概率选择Q值最大的动作（贪婪选择）
                    action = np.argmax(self.qvalue[state])

                # ========== 2. 执行动作并观察结果 ==========
                # 在环境中执行选择的动作，获取即时奖励和新状态
                _, reward, done, _, _ = self.env.step(action)
                next_state = self.env.pos2state(self.env.agent_location)

                # ========== 3. Q值更新：Q-learning核心算法 ==========
                # 这里使用目标策略（target policy），即贪婪策略
                # 找到下一个状态s'的最优动作a* = argmax_a Q(s',a)
                best_next_action = np.argmax(self.qvalue[next_state])
                
                # 计算TD目标：r + γ max_a' Q(s',a')
                # 这里使用max而不是实际执行的动作，这是Q-learning的关键特征
                td_target = reward + self.gamma * self.qvalue[next_state][best_next_action]
                
                # 计算TD误差：δ = Q(s,a) - [r + γ max_a' Q(s',a')]
                td_error = self.qvalue[state][action] - td_target
                
                # 更新Q值：Q(s,a) ← Q(s,a) - α × δ
                # 等价于：Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                self.qvalue[state][action] -= self.alpha * td_error

                # ========== 4. 策略更新（可选，用于可视化） ==========
                # 注意：Q-learning是off-policy的，策略更新不是必需的
                # 这里更新策略主要是为了可视化学到的行为
                qvalue_star = self.qvalue[state].max()  # 当前状态的最大Q值
                action_star = self.qvalue[state].tolist().index(qvalue_star)  # 最优动作
                
                # 使用ε-greedy策略更新策略概率
                for a in range(self.action_space_size):
                    if a == action_star:
                        # 最优动作获得大部分概率
                        self.policy[state, a] = 1 - epsilon + (epsilon / self.action_space_size)
                    else:
                        # 其他动作分享探索概率
                        self.policy[state, a] = epsilon / self.action_space_size

                # 更新状态价值：V(s) = max_a Q(s,a)
                self.state_value[state] = qvalue_star
                
                # ========== 5. 转移到下一个状态 ==========
                state = next_state
                total_reward += reward
                episode_length += 1

            # Episode结束，记录性能指标
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        return total_rewards, episode_lengths
if __name__ =="__main__":
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    solver = Q_learning(alpha =0.05, env = gird_world)
    # solver.sarsa()
    # print("env.policy[0, :]:",solver.policy[0, :])
    # for _ in range(20):
    #     a0 = np.random.choice(5, p=solver.policy[0, :] )
    #
    #     print("a0:",a0)

    start_time = time.time()

    initial_location = [4,0]
    total_rewards, episode_lengths = solver.q_learning(initial_location = initial_location)


    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:",cost_time)
    print(len(gird_world.render_.trajectory))

    initial_state = solver.env.pos2state(initial_location)
    print("训练后的policy结果为:\n",solver.policy[initial_state,:])
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
