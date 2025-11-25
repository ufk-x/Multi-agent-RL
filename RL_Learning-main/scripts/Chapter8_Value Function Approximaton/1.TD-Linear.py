import random
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端以支持图形显示
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
# sys.path.append("..")
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import grid_env


'''
首先通过policy iteration计算ground truth，再通过 TD linear 方法去拟合，观察效果。
有点问题，没有解决2024.8.30

本文件实现了时序差分学习（Temporal Difference Learning）与线性函数近似（Linear Function Approximation）的结合。
主要思想：使用线性函数近似来表示状态值函数，而不是维护一个完整的状态值表。
这对于大型状态空间非常有用，可以显著减少参数数量。

算法原理：
- TD学习是一种无模型的强化学习方法，通过采样来更新值函数。
- 线性函数近似使用特征向量φ(s)来表示状态s，然后用权重向量w来近似值函数：V(s) ≈ φ(s)^T w
- TD(0)更新规则：w ← w + α [r + γ φ(s')^T w - φ(s)^T w] φ(s)
  其中δ = r + γ V(s') - V(s)，但这里用线性近似表示。

特征向量：
- 多项式特征：对于二维网格，使用多项式基函数，如1, x, y, x^2, xy, y^2等。
- 傅里叶特征：使用余弦函数作为基函数，更适合周期性数据。

实验设置：
- 使用5x5网格世界，有目标、禁区等。
- 先用策略评估计算真实状态值（ground truth）。
- 然后用TD线性近似学习近似状态值，并比较RMSE。
'''
class TD_learning_with_FunctionApproximation():
    def __init__(self,alpha,env = grid_env.GridEnv):
         # 初始化折扣因子γ，通常设为0.9，表示未来奖励的权重递减
         self.gamma = 0.9  # discount rate
         # 学习率α，控制每次更新的步长，太大可能不稳定，太小收敛慢
         self.learning_rate = alpha  #learning rate
         # 环境对象，包含网格世界的状态、动作、奖励等信息
         self.env = env
         # 动作空间大小，通常为4（上下左右）或5（包括不动）
         self.action_space_size = env.action_space_size
         # 状态空间大小，对于5x5网格为25
         self.state_space_size = env.size ** 2
         # 奖励空间大小和奖励列表，网格世界中奖励为-1,0,1等
         self.reward_space_size, self.reward_list = len(
             self.env.reward_list), [-1,-1,0,1]  # [-10,-10,0,1]  reward list
         # 真实状态值数组，用策略评估计算得到，作为ground truth
         self.state_value = np.zeros(shape=self.state_space_size)  # 一维列表
         print("self.state_value:", self.state_value)
         # Q值表，状态-动作值函数，但这里主要用于状态值近似
         self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 二维： state数 x action数
         # 平均策略，每个动作概率相等，用于生成episode
         self.mean_policy = np.ones(     #self.mean_policy shape: (25, 5)
             shape=(self.state_space_size, self.action_space_size)) / self.action_space_size  # 平均策略，即取每个动作的概率均等
         # 当前策略，初始化为平均策略
         self.policy = self.mean_policy.copy()
         # TensorBoard写入器，用于记录训练日志
         self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

         # 打印环境信息，便于调试
         print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
         print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                      self.qvalue.shape,
                                                                                      self.mean_policy.shape))

         print('----------------------------------------------------------------')

    def show_policy(self):
         """
         在网格世界中可视化策略。
         对于每个状态，根据策略概率绘制动作箭头。
         箭头长度和半径与动作概率成正比。
         """
         for state in range(self.state_space_size):
             for action in range(self.action_space_size):
                 policy = self.policy[state, action]
                 self.env.render_.draw_action(pos=self.env.state2pos(state),
                                              toward=policy * 0.4 * self.env.action_to_direction[action],
                                              radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2, color='black'):
         """
         在网格世界中显示状态值。
         在每个状态位置显示其值函数的数值。
         :param state_value: 要显示的状态值数组
         :param y_offset: 文本在Y轴上的偏移，用于区分多个值显示
         """
         for state in range(self.state_space_size):
             self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                         y_offset=y_offset,
                                         size_discount=0.7, color=color)
    def obtain_episode(self, policy, start_state, start_action, length):
        """
        根据给定的策略生成一个episode（轨迹）。
        从指定状态和动作开始，按照策略采样动作，直到达到指定长度。
        :param policy: 用于采样的策略，二维数组[state][action] = 概率
        :param start_state: 起始状态
        :param start_action: 起始动作
        :param length: episode的长度（步数）
        :return: episode列表，每个元素为字典{"state", "action", "reward", "next_state", "next_action"}
        """
        # 设置智能体初始位置
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            # 执行动作，获取奖励和下一状态
            _, reward, done, _, _ = self.env.step(action)  # 一步动作
            next_state = self.env.pos2state(self.env.agent_location)
            # 根据策略采样下一动作
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            # 记录这一步的经验
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})  #向列表中添加一个字典
        return episode  #返回列表，其中的元素为字典


    def get_feature_vector(self, fourier: bool, state: int, ord: int) -> np.ndarray:
        """
        生成状态的特征向量φ(s)，用于线性函数近似。
        支持多项式和傅里叶两种基函数。
        :param fourier: True使用傅里叶基函数，False使用多项式基函数
        :param state: 状态索引
        :param ord: 基函数的阶数，控制特征向量的维度
        :return: 特征向量数组
        """
        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state value")

        # 将状态转换为网格坐标，并调整为1-based
        x, y = self.env.state2pos(state) + (1, 1)
        feature_vector = []

        if fourier:
            # 傅里叶基函数：φ_i(s) = cos(π * (i_x * x_norm + i_y * y_norm))
            # 归一化坐标到[0,1]，但代码中除以size，可能为[0,1]
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * y_normalized)))
        else:
            # 多项式基函数：φ(s) = [1, x, y, x^2, xy, y^2, ...]
            # 归一化并中心化坐标，避免偏向区间一端
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            # 初始化特征向量，常数项为1
            feature_vector = [1]  # 特征向量的第一个元素总是常数1
            for i in range(1, ord + 1):  # 从1到ord阶
                for j in range(i + 1):  # j表示y的指数，i-j表示x的指数
                    feature_vector.append((x_normalized ** (i - j)) * (y_normalized ** j))  #[1, x, y, x^2, xy, y^2]
        return np.array(feature_vector)

    def get_feature_vector_with_action(self, fourier: bool, state: int, action: int, ord: int) -> np.ndarray:
        """
        生成状态-动作对的特征向量，用于Q函数近似。
        类似get_feature_vector，但包含动作信息。
        :param fourier: True使用傅里叶基函数
        :param state: 状态
        :param action: 动作
        :param ord: 阶数
        :return: 特征向量
        """
        if state < 0 or state >= self.state_space_size or action < 0 or action >= self.action_space_size:
            raise ValueError("Invalid state/action value")
        feature_vector = []
        y, x = self.env.state2pos(state) + (1, 1)

        if fourier:
            # 傅里叶基函数，包含状态和动作
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(
                            np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized)))
        else:
            # 多项式基函数，包含状态和动作
            state_normalized = (state - (self.state_space_size - 1) * 0.5) / (self.state_space_size - 1)
            action_normalized = (action - (self.action_space_size - 1) * 0.5) / (self.action_space_size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(state_normalized ** (ord - i) * action_normalized ** j)
        return np.array(feature_vector)


    def state_iteration(self, policy, tolerance=0.0001, max_iterations=10000):
        """
        策略评估：计算给定策略下的状态值函数。
        使用迭代方法求解贝尔曼方程：V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V^π(s')]
        迭代直到收敛或达到最大步数。
        :param policy: 要评估的策略
        :param tolerance: 收敛阈值，前后两次V的L1范数差小于此值认为收敛
        :param steps: 最大迭代步数
        :return: 收敛后的状态值数组
        """
        # 初始化状态值
        state_value = np.zeros(self.state_space_size)
        iteration_count = 0
        while True:
            # 保存旧的状态值用于检查收敛
            state_value_old = state_value.copy()
            # 对每个状态进行策略评估更新
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    # 计算Q值并加权求和，使用旧的state_value（同步更新）
                    value += policy[state, action] * self.calculate_qvalue(state=state, action=action, state_value=state_value_old)
                state_value[state] = value
            
            # 检查是否收敛
            iteration_count += 1
            if np.linalg.norm(state_value - state_value_old, ord=1) < tolerance:
                print(f"State iteration converged after {iteration_count} iterations")
                break
            if iteration_count >= max_iterations:
                print(f"State iteration stopped after {max_iterations} iterations")
                break
        return state_value
    
    def calculate_qvalue(self, state, action, state_value):
        """
        计算Q值：Q(s,a) = Σ_r p(r|s,a) * r + γ Σ_{s'} p(s'|s,a) * V(s')
        :param state: 当前状态
        :param action: 执行动作
        :param state_value: 当前的状态值估计
        :return: Q值
        """
        qvalue = 0
        # 奖励期望
        for i in range(self.reward_space_size):
            qvalue += self.env.reward_list[i] * self.env.Rsa[state, action, i]
        # 折扣未来值期望
        for next_state in range(self.state_space_size):
            qvalue += self.gamma * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue
    
    def td_state_value_hat(self, epochs=5000, fourier=False, ord=1):  # False 时为多项式feature vector
        """
        TD(0)学习与线性函数近似的主函数。
        先计算真实状态值作为ground truth，然后用TD学习近似状态值函数。
        :param epochs: 训练轮数，每轮生成一个episode并更新权重
        :param fourier: 是否使用傅里叶特征
        :param ord: 特征阶数
        :return: 近似状态值数组
        """
        # 先用state iteration计算真实状态值
        self.state_value = self.state_iteration(self.policy)
        
        # 输入验证
        if not isinstance(self.learning_rate, float) or not isinstance(epochs, int) or not isinstance(
                fourier, bool) or not isinstance(ord, int):
            raise TypeError("Invalid input type")
        if self.learning_rate <= 0 or epochs <= 0 or ord <= 0:
            raise ValueError("Invalid input value")

        # 计算特征向量维度
        dim = (ord + 1) ** 2 if fourier else np.arange(ord + 2).sum()  #条件表达式;  计算特征向量的维度
        # 初始化权重向量，从标准正态分布采样
        w = np.random.default_rng().normal(size=dim) # 初始化权重参数; 生成了一个长度为 dim 的向量，其中每个元素都服从标准正态分布（均值为 0，方差为 1）
        print("feature vector w:",w) # parameter

        rmse = []  # 记录每轮的均方根误差
        value_hat = np.zeros(self.state_space_size)

        for epoch in range(epochs):
            # 随机选择起始状态和动作
            start_state = np.random.randint(self.state_space_size)
            start_action = np.random.choice(np.arange(self.action_space_size),
                                            p=self.mean_policy[start_state])
            # 生成episode
            episode = self.obtain_episode(self.mean_policy, start_state, start_action, length=epochs)
            # 对episode中的每一步进行TD更新
            for sample in episode:
                reward = sample['reward']
                state = sample['state']
                next_state = sample['next_state']
                # TD更新公式：w += α [r + γ φ(s')^T w - φ(s)^T w] φ(s)
                # 这是半梯度TD(0)方法
                w += self.learning_rate*(reward+ self.gamma*np.dot(self.get_feature_vector(fourier, next_state, ord),w)- np.dot(self.get_feature_vector(fourier, state, ord),w) )*self.get_feature_vector(fourier, state, ord)

            # 计算当前近似值
            for state in range(self.state_space_size):
                value_hat[state] = np.dot(self.get_feature_vector(fourier, state, ord), w)
            # 计算与真实值的RMSE
            rmse.append(np.sqrt(np.mean((value_hat - self.state_value) ** 2)))
            if epoch % 100 == 0:
                print(epoch)

        # 可视化结果
        X, Y = np.meshgrid(np.arange(1, 6), np.arange(1, 6))  # position on grid world.
        Z = self.state_value.reshape(5, 5)
        Z1 = value_hat.reshape(5, 5)
        
        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(14, 6))  # 设置图形的尺寸，宽度为14以容纳两个子图
        
        # 左图：真实状态值
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title('True State Value')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('State Value')
        
        # 右图：估计状态值
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1, cmap='plasma')
        ax1.set_title('Estimated State Value')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('State Value')
        # 不设置固定的z轴范围，让matplotlib自动调整以显示所有数据
        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        # 不立即显示，让main函数统一管理显示顺序
        return value_hat
    
def main():
    """
    主函数：设置网格世界环境，创建求解器，运行TD学习并可视化结果。
    """
    print("Creating grid world")
    # 创建5x5网格世界，设置目标位置[2,3]，禁区等
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')

    print("Creating solver")
    # 创建TD学习求解器，学习率设为0.0005
    solver = TD_learning_with_FunctionApproximation(alpha=0.0005, env=gird_world)  # 实例化5

    print("Calculating state value hat")
    # 运行TD学习，默认参数：5000轮，不使用傅里叶，1阶多项式
    state_value_hat = solver.td_state_value_hat(epochs=500, fourier=True, ord=3)
    
    print("state_value_hat:", state_value_hat)
    print("solver.state_value:", solver.state_value)
    
    # 在网格上显示策略箭头
    print("Showing policy arrows")
    solver.show_policy()
    
    # 显示真实状态值（上方，偏移-0.25）
    print("Showing true state values")
    solver.show_state_value(state_value=solver.state_value, y_offset=-0.25, color='green')
    
    # 显示近似状态值（下方，偏移0.25）
    print("Showing estimated state values")
    solver.show_state_value(state_value=state_value_hat, y_offset=0.25)
    
    # 显示所有matplotlib图形（包括网格世界、3D图和RMSE图）
    print("Displaying all figures...")
    plt.show()

if __name__ == "__main__":
    main()