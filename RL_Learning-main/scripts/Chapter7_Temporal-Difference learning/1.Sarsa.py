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
class Sarsa():
    """Sarsa算法类
    
    SARSA是一种时序差分(TD)学习算法，用于学习动作值函数Q(s,a)
    算法名称来源于更新序列: State - Action - Reward - State - Action
    这是一种on-policy算法，即用于生成行为的策略和被改进的策略是同一个
    """
    def __init__(self,alpha,env = grid_env.GridEnv):
        """
        初始化Sarsa算法参数
        
        Args:
            alpha: 学习率，控制每次更新Q值的步长，范围[0,1]
            env: 网格世界环境对象
        """
        self.gama = 0.9  # 折扣因子γ，用于计算未来奖励的现值，范围[0,1]
        self.alpha = alpha  # 学习率α，决定新信息覆盖旧信息的速度
        self.env = env  # 环境对象
        self.action_space_size = env.action_space_size  # 动作空间大小（通常是5：上下左右和停留）
        self.state_space_size = env.size ** 2  # 状态空间大小（网格世界的格子总数）
        # 奖励空间大小和奖励列表：[-10,-10,0,1] 分别对应禁区、撞墙、普通区域、目标
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list
        # 状态价值函数V(s)，一维数组，每个元素对应一个状态的价值
        self.state_value = np.zeros(shape=self.state_space_size)
        print("self.state_value:", self.state_value)
        # Q值表Q(s,a)，二维数组，行表示状态，列表示动作
        # 用于存储每个状态-动作对的价值估计
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # 初始策略：均匀随机策略，每个动作被选择的概率相等
        # shape: (state_space_size, action_space_size)，每行和为1
        self.mean_policy = np.ones(
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()  # 当前策略，初始化为均匀策略
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))

        print('----------------------------------------------------------------')

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_episode(self, policy, start_state, start_action, length):
        """
        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: 一个episode 长度
        :return: 一个列表，其中是字典格式: state,action,reward,next_state,next_action
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)  # 一步动作
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})  #向列表中添加一个字典
        return episode  #返回列表，其中的元素为字典

    def Sarsa_alg(self,initial_location, epsilon = 0.1):
        """Sarsa算法主函数：学习从初始状态到目标状态的最优策略
        
        使用时序差分(TD)方法，通过与环境交互来学习Q值函数和改进策略。
        采用ε-greedy策略平衡探索与利用。
        
        算法流程：
        1. 在状态s选择动作a（根据当前策略）
        2. 执行动作a，观察奖励r和新状态s'
        3. 在状态s'选择动作a'（根据当前策略）
        4. 更新Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]  (Sarsa更新公式)
        5. 根据新的Q值改进策略（ε-greedy）
        6. s←s', a←a'，继续下一步
        
        Args:
            initial_location: 初始位置坐标，如[0,0]
            epsilon: ε-greedy策略中的探索概率，范围[0,1]
                    以ε的概率随机选择动作（探索），以1-ε的概率选择最优动作（利用）
                    
        Returns:
            total_rewards: 每个episode的累计奖励列表
            episode_lengths: 每个episode的长度（步数）列表
        """
        # 用于记录训练过程的指标
        total_rewards = []  # 存储每个episode的总奖励
        episode_lengths = []  # 存储每个episode的长度
        
        # 将位置坐标转换为状态索引
        initial_state = self.env.pos2state(initial_location)
        print("initial_state:", initial_state)
        
        # 开始训练循环，共1000个episodes
        for episode_num in range(1000):
            # epsilon衰减：从高探索率逐渐减少到低探索率
            # 前期多探索，后期多利用
            current_epsilon = epsilon * (1 - episode_num / 1000)  # 线性衰减到0
            current_epsilon = max(0.01, current_epsilon)  # 保持最小1%的探索
            
            self.env.reset()  # 重置环境
            # 重要：将智能体放回初始位置
            self.env.agent_location = initial_location
            
            total_reward = 0  # 当前episode的累计奖励
            episode_length = 0  # 当前episode的步数
            done = False  # 是否到达终止状态（目标）
            max_steps = 200  # 减少最大步数，强制智能体学习更短路径
            
            # 每100个episode打印一次进度
            if episode_num % 100 == 0:
                print(f"episode_num: {episode_num}, epsilon: {current_epsilon:.3f}")

            # 初始化：在起始状态s0根据策略π选择初始动作a0
            state = initial_state
            action = np.random.choice(a=np.arange(self.action_space_size),
                                      p=self.policy[state, :])
            
            # 初始化缓冲区（用于记录轨迹，虽然在此代码中未使用）
            states = [state]
            aciton = [action]
            rewards = [0]
            
            # Episode主循环：持续执行直到到达目标状态或达到最大步数
            while not done and episode_length < max_steps:
                episode_length += 1
                
                # 执行动作a，观察即时奖励r和是否终止
                _, reward, done, _, _ = self.env.step(action)
                
                # 获取新状态s'（执行动作后智能体所在的状态）
                next_state = self.env.pos2state(self.env.agent_location)
                
                # 在新状态s'根据当前策略π选择新动作a'
                # 这是Sarsa的关键：使用实际执行的策略来选择a'
                next_action = np.random.choice(np.arange(self.action_space_size),
                                               p=self.policy[next_state,:])
                
                # 累加本步奖励
                total_reward += reward
                
                # Sarsa核心更新公式：Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                # TD误差：δ = r + γQ(s',a') - Q(s,a)
                # 新Q值 = 旧Q值 + α × TD误差
                td_target = reward + self.gama * self.qvalue[next_state, next_action]
                td_error = td_target - self.qvalue[state, action]
                self.qvalue[state][action] = self.qvalue[state, action] + self.alpha * td_error
                
                # 策略改进：根据更新后的Q值，使用ε-greedy策略更新策略
                # 找到当前状态s下Q值最大的动作（贪婪动作）
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                
                # ε-greedy策略更新：使用当前epsilon
                # - 最优动作a*的概率：1-ε+ε/|A| （大部分概率+一小部分随机概率）
                # - 其他动作的概率：ε/|A| （只有随机探索的概率）
                for a in range(self.action_space_size):
                    if a == action_star:
                        # 贪婪动作：获得(1-ε)的基础概率 + ε/|A|的随机探索概率
                        self.policy[state, a] = 1 - current_epsilon + (current_epsilon / self.action_space_size)
                    else:
                        # 非贪婪动作：只有ε/|A|的随机探索概率
                        self.policy[state, a] = current_epsilon / self.action_space_size

                # 更新状态价值：V(s) = max_a Q(s,a)
                self.state_value[state] = qvalue_star
                
                # 转移到下一个状态-动作对 (s,a) ← (s',a')
                action = next_action
                state = next_state
                
            # Episode结束，记录性能指标
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)

        return total_rewards,episode_lengths

if __name__ =="__main__":
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')
    # 降低学习率以提高稳定性
    solver = Sarsa(alpha =0.05, env = gird_world)
    # solver.sarsa()
    # print("env.policy[0, :]:",solver.policy[0, :])
    # for _ in range(20):
    #     a0 = np.random.choice(5, p=solver.policy[0, :] )
    #
    #     print("a0:",a0)

    start_time = time.time()

    initial_location = [0,0]
    # 使用更高的探索率，帮助智能体探索到目标
    total_rewards, episode_lengths = solver.Sarsa_alg(initial_location = initial_location, epsilon=0.3)


    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:",cost_time)
    print(len(gird_world.render_.trajectory))

    # 打印训练统计信息
    print("\n=== 训练统计 ===")
    print(f"平均episode长度: {np.mean(episode_lengths):.2f}")
    print(f"平均总奖励: {np.mean(total_rewards):.2f}")
    print(f"最后100个episodes平均长度: {np.mean(episode_lengths[-100:]):.2f}")
    print(f"最后100个episodes平均奖励: {np.mean(total_rewards[-100:]):.2f}")
    
    initial_state = solver.env.pos2state(initial_location)
    print("\n训练后的policy结果为:\n",solver.policy[initial_state,:])
    print(f"初始状态的Q值: {solver.qvalue[initial_state,:]}")
    print(f"初始状态的状态价值: {solver.state_value[initial_state]}")
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
