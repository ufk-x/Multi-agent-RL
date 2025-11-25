import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter


# 引用上级目录
import sys
# sys.path.append("..")
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import grid_env

from model import *

import matplotlib.pyplot as plt

class QNET(nn.Module):
    """
    Q网络: 用于逼近状态-动作值函数Q(s,a)
    输入: (y坐标, x坐标, 动作) - 3维向量
    输出: Q值 - 标量,表示在该状态下采取该动作的期望回报
    
    网络结构: 3 -> 128 -> 64 -> 32 -> 1
    使用ReLU作为激活函数,最后一层无激活函数(回归任务)
    """
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        # 构建全连接神经网络
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),  # 输入层到第一隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(in_features=128, out_features=64),  # 第一隐藏层到第二隐藏层
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),  # 第二隐藏层到第三隐藏层
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),  # 输出层,输出Q值
        )

    def forward(self, x):
        """前向传播,计算Q值"""
        x = x.type(torch.float32)  # 确保输入是float32类型
        return self.fc(x)


class PolicyNet(nn.Module):
    """
    策略网络: 用于随机策略(本DQN实现中未使用,仅作参考)
    输入: (y坐标, x坐标) - 2维向量
    输出: 动作概率分布 - 5维向量,对应5个动作的概率
    """
    def __init__(self, input_dim=2, output_dim=5):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
            nn.Softmax(dim=1)  # Softmax确保输出是概率分布
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class DPolicyNet(nn.Module):
    """
    确定性策略网络: 用于确定性策略梯度算法(本DQN实现中未使用,仅作参考)
    输入: 状态
    输出: 动作值(连续动作空间)
    """
    def __init__(self, input_dim=2, output_dim=1):
        super(DPolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class ValueNet(torch.nn.Module):
    """
    状态价值网络: 用于逼近状态价值函数V(s)(本DQN实现中未使用,仅作参考)
    输入: 状态
    输出: 状态价值V(s)
    """
    def __init__(self, input_dim=2, output_dim=1):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)

class DQN():
    """
    DQN (Deep Q-Network) 算法实现
    
    DQN是第一个成功将深度学习与强化学习结合的算法,主要特点:
    1. 使用神经网络逼近Q函数
    2. 经验回放(Experience Replay): 存储历史经验,随机采样打破数据相关性
    3. 目标网络(Target Network): 使用固定参数的目标网络计算TD目标,提高稳定性
    """
    def __init__(self,alpha,env = grid_env.GridEnv):
        """
        初始化DQN算法
        
        参数:
            alpha: 学习率(本实现中未直接使用,在dqn方法中设置)
            env: 网格世界环境
        """
        # 超参数
        self.gama = 0.9  # 折扣因子,用于计算未来奖励的现值
        self.alpha = alpha  # 学习率
        self.env = env
        
        # 环境信息
        self.action_space_size = env.action_space_size  # 动作空间大小: 5 (上下左右停)
        self.state_space_size = env.size ** 2  # 状态空间大小: 25 (5x5网格)
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # 奖励列表: [普通, 目标, 禁区, 越界]
        
        # 初始化价值函数和策略
        self.state_value = np.zeros(shape=self.state_space_size)  # 状态价值: V(s)
        print("self.state_value:", self.state_value)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # Q值表: Q(s,a)
        
        # 均匀策略: 每个动作被选择的概率相等,用于生成初始经验
        self.mean_policy = np.ones(
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()  # 当前策略(贪婪策略)
        
        # TensorBoard日志记录器
        self.writer = SummaryWriter("logs")

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))

        print('----------------------------------------------------------------')

    def show_policy(self):
        """
        可视化策略: 在网格世界中绘制策略箭头
        
        对于确定性策略(DQN学到的贪婪策略):
        - 在每个状态位置绘制一个箭头,指向最优动作方向
        - 箭头长度和粗细与策略概率成正比(贪婪策略为1或0)
        """
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]  # 获取在该状态下选择该动作的概率
                # 绘制动作箭头: 位置、方向、大小
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        """
        可视化状态价值: 在网格世界的每个格子中显示状态价值
        
        参数:
            state_value: 状态价值数组
            y_offset: 文字在y方向的偏移量,避免与箭头重叠
        """
        for state in range(self.state_space_size):
            # 在每个状态位置写入价值(保留1位小数)
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_episode(self, policy, start_state, start_action, length):
        """
        生成经验序列(Episode) - DQN中用于填充经验回放缓冲区
        
        经验回放是DQN的核心技巧之一:
        - 将agent与环境交互产生的经验(s,a,r,s')存储起来
        - 训练时从中随机采样,打破数据的时间相关性
        - 提高样本利用效率,使训练更稳定
        
        参数:
            policy: 用于生成经验的策略(这里使用均匀随机策略)
            start_state: 起始状态
            start_action: 起始动作
            length: episode长度(采集多少条经验)
        
        返回:
            episode: 经验列表,每条经验是一个字典 {state, action, reward, next_state, next_action}
        """
        # 重置agent位置到起始状态
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []  # 存储经验的列表
        next_action = start_action
        next_state = start_state
        
        # 生成指定长度的经验序列
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            
            # 在环境中执行动作,获取奖励和下一个状态
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            
            # 根据策略选择下一个动作
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            
            # 存储经验元组 (s, a, r, s', a')
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode


    def get_data_iter(self, episode, batch_size=64, is_train=True):
        """
        构造PyTorch数据迭代器 - 用于小批量训练
        
        将经验序列转换为PyTorch的DataLoader,支持:
        - 小批量随机梯度下降(Mini-batch SGD)
        - 数据打乱(shuffle),进一步打破相关性
        - 自动批处理
        
        参数:
            episode: 经验序列
            batch_size: 批大小
            is_train: 是否为训练模式(决定是否打乱数据)
        
        返回:
            DataLoader: PyTorch数据迭代器
        """
        reward = []  # 存储奖励
        state_action = []  # 存储(状态,动作)对,作为Q网络输入
        next_state = []  # 存储下一个状态
        
        # 从经验中提取数据
        for i in range(len(episode)):
            reward.append(episode[i]['reward'])
            action = episode[i]['action']
            # 将状态转换为坐标形式 (y, x, action)
            y, x = self.env.state2pos(episode[i]['state'])
            state_action.append((y, x, action))
            # 下一个状态只需要坐标 (y, x)
            y, x = self.env.state2pos(episode[i]['next_state'])
            next_state.append((y, x))
        
        # 转换为PyTorch张量
        reward = torch.tensor(reward).reshape(-1, 1)  # shape: (N, 1)
        state_action = torch.tensor(state_action)  # shape: (N, 3)
        next_state = torch.tensor(next_state)  # shape: (N, 2)
        
        # 创建数据集和数据加载器
        data_arrays = (state_action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)


    def dqn(self, learning_rate=0.0015, episode_length=5000, epochs=600, batch_size=100, update_step=10):
        """
        DQN算法主函数
        
        算法流程:
        1. 初始化主网络Q和目标网络Q'
        2. 使用随机策略生成经验,填充经验回放缓冲区
        3. 重复以下步骤:
           - 从缓冲区随机采样一批经验
           - 用目标网络计算TD目标: y = r + γ * max_a' Q'(s', a')
           - 用主网络计算当前Q值: Q(s, a)
           - 最小化TD误差: L = (y - Q(s,a))²
           - 每隔C步,更新目标网络: Q' ← Q
        
        参数:
            learning_rate: 学习率
            episode_length: 经验序列长度(经验回放缓冲区大小)
            epochs: 训练轮数
            batch_size: 小批量大小
            update_step: 目标网络更新频率(每C步更新一次)
        """
        # 保存初始策略和价值,用于计算RMSE
        policy = self.policy.copy()
        state_value = self.state_value.copy()
        
        # ============ 初始化神经网络 ============
        # 主网络(Main Network): 用于选择动作和更新
        q_net = QNET()
        # 目标网络(Target Network): 用于计算TD目标,参数固定一段时间后才更新
        q_target_net = QNET()
        q_target_net.load_state_dict(q_net.state_dict())  # 初始化为相同参数
        
        # 优化器: 使用随机梯度下降(SGD)
        optimizer = torch.optim.SGD(q_net.parameters(), lr=learning_rate)
        
        # ============ 生成经验回放缓冲区 ============
        # 使用均匀随机策略生成初始经验
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=episode_length)
        # 转换为PyTorch数据迭代器
        date_iter = self.get_data_iter(episode, batch_size)
        
        # ============ 初始化训练相关变量 ============
        loss = torch.nn.MSELoss()  # 均方误差损失函数
        approximation_q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 存储Q值
        i = 0  # 迭代计数器
        rmse_list = []  # 记录每轮的RMSE
        loss_list = []  # 记录每轮的loss
        # ============ DQN训练主循环 ============
        for epoch in range(epochs):
            # 遍历经验回放缓冲区中的小批量数据
            for state_action, reward, next_state in date_iter:
                i += 1
                
                # -------- 步骤1: 计算当前Q值 Q(s,a) --------
                q_value = q_net(state_action)  # shape: (batch_size, 1)
                
                # -------- 步骤2: 用目标网络计算TD目标 --------
                # 对于下一个状态s',计算所有动作的Q值: Q'(s', a') for all a'
                q_value_target = torch.empty((batch_size, 0))  # shape: (batch_size, 0)
                
                for action in range(self.action_space_size):  # 遍历所有动作(0,1,2,3,4)
                    # 构造输入: (y, x, action)
                    s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                    # 计算Q'(s', a)
                    q_value_target = torch.cat((q_value_target, q_target_net(s_a)), dim=1)
                
                # 取最大Q值: max_a' Q'(s', a')
                q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]  # shape: (batch_size, 1)
                
                # -------- 步骤3: 计算TD目标 y = r + γ * max_a' Q'(s', a') --------
                y_target_value = reward + self.gama * q_star
                
                # -------- 步骤4: 计算损失并更新主网络 --------
                l = loss(q_value, y_target_value)  # MSE损失: (Q(s,a) - y)²
                optimizer.zero_grad()  # 清空梯度
                l.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新主网络参数
                
                # -------- 步骤5: 定期更新目标网络 --------
                # 目标网络的参数固定C步后才更新,提高训练稳定性
                if i % update_step == 0 and i != 0:
                    q_target_net.load_state_dict(q_net.state_dict())  # Q' ← Q
            
            # 记录loss
            loss_list.append(float(l))
            print("loss:{},epoch:{}".format(l, epoch))
            # -------- 步骤6: 从Q网络提取贪婪策略和状态价值 --------
            # 每个epoch结束后,根据学到的Q函数提取策略
            self.policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            self.state_value = np.zeros(shape=self.state_space_size)

            for s in range(self.state_space_size):
                y, x = self.env.state2pos(s)  # 获取状态坐标
                
                # 计算该状态下所有动作的Q值
                for a in range(self.action_space_size):
                    approximation_q_value[s, a] = float(q_net(torch.tensor((y, x, a)).reshape(-1, 3)))
                
                # 贪婪策略: 选择Q值最大的动作
                q_star_index = approximation_q_value[s].argmax()
                self.policy[s, q_star_index] = 1  # 确定性策略
                self.state_value[s] = approximation_q_value[s, q_star_index]  # V(s) = max_a Q(s,a)
            
            # 计算与初始价值函数的RMSE(用于评估收敛情况)
            rmse_list.append(np.sqrt(np.mean((state_value - self.state_value) ** 2)))
        
        # ============ 训练完成后,绘制训练曲线 ============
        fig_rmse = plt.figure(figsize=(8, 12))
        
        # 子图1: RMSE曲线(状态价值函数的均方根误差)
        ax_rmse = fig_rmse.add_subplot(211)
        ax_rmse.plot(rmse_list)
        ax_rmse.set_title('RMSE - State Value Function Convergence')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        # 子图2: Loss曲线(TD误差)
        ax_loss = fig_rmse.add_subplot(212)
        ax_loss.plot(loss_list)
        ax_loss.set_title('Training Loss - TD Error')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        
        # 返回图形对象供主程序显示
        return fig_rmse

def main():
    """
    主程序: DQN算法在网格世界中的应用
    
    问题设定:
    - 5x5网格世界
    - 目标位置: [2, 3]
    - 障碍物(禁区): [[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]]
    - 动作空间: 上、右、下、左、停留
    - 目标: 学习从任意位置到达目标的最优策略
    """
    
    # ============ 1. 创建环境 ============
    gird_world = grid_env.GridEnv(
        size=5,  # 5x5网格
        target=[2, 3],  # 目标位置
        forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],  # 障碍物
        render_mode=''  # 不使用视频模式,直接显示图形窗口
    )
    
    # ============ 2. 创建DQN智能体 ============
    solver = DQN(alpha=0.1, env=gird_world)
    
    # ============ 3. 训练DQN ============
    print("开始训练DQN...")
    start_time = time.time()
    
    # 训练并获取loss图表对象
    fig_loss = solver.dqn(
        learning_rate=0.0015,  # 学习率
        episode_length=50000,  # 经验回放缓冲区大小
        epochs=100,  # 训练轮数
        batch_size=100,  # 小批量大小
        update_step=10  # 目标网络更新频率
    )
    
    end_time = time.time()
    cost_time = end_time - start_time
    
    # ============ 4. 输出结果 ============
    print("\n训练完成!")
    print("训练时间: {:.2f}秒".format(cost_time))
    print("\n学到的状态价值函数:")
    print("solver.state_value:", solver.state_value)
    
    # ============ 5. 可视化结果 ============
    # 5.1 先显示网格世界(带策略箭头和状态价值)
    print("\n正在显示网格世界(策略和状态价值)...")
    solver.show_policy()  # 绘制策略箭头
    solver.show_state_value(solver.state_value, y_offset=0.25)  # 显示状态价值
    solver.env.render()  # 显示图形窗口(第一个窗口)
    
    # 5.2 关闭网格世界窗口后,显示训练曲线
    print("关闭网格世界窗口后将显示训练曲线(Loss和RMSE)...")
    if fig_loss is not None:
        plt.show()  # 显示loss和RMSE图表(第二个窗口)


if __name__ == '__main__':
    main()