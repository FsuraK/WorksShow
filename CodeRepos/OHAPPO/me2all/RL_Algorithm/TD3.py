import gym
import torch
import random
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal

'''
TD3：截断双Q 加入噪声 降低策略网络更新频率
是确定策略梯度，需要使用辅助网络，从而进行反向传播更新梯度
异策略，可以使用经验回放
一共6个网络，截断双Q,策略网络及3个目标网络
'''


def plotimage():
    plt.close()
    plt.figure('loss and reward')
    plt.subplot(211)
    plt.plot(np.arange(0, len(plt_loss_policy), 1), np.array(plt_loss_policy))
    plt.ylabel('policy_loss')

    plt.subplot(212)
    plt.plot(np.arange(0, len(plt_loss_q1), 1), np.array(plt_loss_q1))
    plt.xlabel('iter_times')
    plt.ylabel('q1_Loss')
    plt.show()


class ReplayBeffer():
    """ SAC使用replay buffer经验回放
    这个buffer的结构 相比之前使用的很清晰"""

    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []
        next_action_list = []

        batch = random.sample(self.buffer, batch_size)
        # 抽取n给batch，每一个batch都是一个五元组：state, action, reward, next_state, done
        for experience in batch:
            s, a, r, n_s, d, n_a = experience

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)
            next_action_list.append(n_a)

        return torch.FloatTensor(state_list), \
            torch.FloatTensor(action_list), \
            torch.FloatTensor(reward_list).unsqueeze(-1), \
            torch.FloatTensor(next_state_list), \
            torch.FloatTensor(done_list).unsqueeze(-1), \
            torch.FloatTensor(next_action_list)

    def buffer_len(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)
        self.linear4.weight.data.uniform_(-0.003, 0.003)
        self.linear4.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, action_dim)

        self.linear4.weight.data.uniform_(-3e-3, 3e-3)
        self.linear4.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        a = self.linear4(x)
        a = torch.tanh(a)
        return a


class TD3:
    def __init__(self, buffer_maxlen=50000, policy_lr=3e-3):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.tau = 0.1
        self.count = 0
        self.buffer = ReplayBeffer(buffer_maxlen)

        """六个网络"""
        self.Q1 = QNet(self.state_dim, self.action_dim)
        self.Q2 = QNet(self.state_dim, self.action_dim)
        self.policy = PolicyNet(self.state_dim, self.action_dim)
        self.target_Q1 = QNet(self.state_dim, self.action_dim)
        self.target_Q2 = QNet(self.state_dim, self.action_dim)
        self.target_policy = PolicyNet(self.state_dim, self.action_dim)

        """目标网络参数初始化"""
        for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
            target_param.data.copy_(param)
        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param)
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(param)

        """优化器"""
        self.q1_optimizer = optim.Adam(self.Q1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.Q2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

    def agent_action(self, state):

        state = torch.FloatTensor(state)
        a = self.policy(state) + torch.FloatTensor(Normal(0, 0.1).sample())
        a = a.data.numpy()

        return a

    def learn(self, batch_size):
        state, action, reward, next_state, done, next_action = self.buffer.sample(batch_size)

        # a = self.policy(state)
        next_a_target = self.target_policy(next_state)

        '''损失函数'''
        next_q1_target = self.target_Q1(next_state, next_a_target)
        next_q2_target = self.target_Q2(next_state, next_a_target)
        TD_target = reward + 0.9 * torch.min(next_q2_target, next_q1_target)
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        q1_loss = F.mse_loss(q1, TD_target.detach())
        q2_loss = F.mse_loss(q2, TD_target.detach())

        '''更新参数'''
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.count += 1
        if self.count % 20 == 0:
            action_now = self.policy(state)
            q1_now = self.Q1(state, action_now)
            policy_loss = -torch.mean(q1_now)

            plt_loss_policy.append(policy_loss.tolist())  # ////////
            plt_loss_q1.append(q1_loss.tolist())

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            # print('1212',policy_loss)

            for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    # Params
    tau = 0.01
    gamma = 0.9
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    Episode = 80
    batch_size = 128
    count = 0
    plt_loss_policy = []
    plt_loss_q1 = []

    Return = []
    action_range = [env.action_space.low, env.action_space.high]

    agent = TD3()
    for episode in range(Episode):
        score = 0
        state = env.reset()
        state = torch.FloatTensor(state[0])

        for i in range(300):
            action = agent.agent_action(state)
            action_in = action * (action_range[1] - action_range[0]) / 2.0 + (action_range[1] + action_range[0]) / 2.0

            next_state, reward, done, o, p = env.step(action_in)
            next_action = agent.agent_action(next_state)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, next_state, done_mask, next_action))
            state = next_state

            score += reward
            if done:
                break
            if agent.buffer.buffer_len() > 500:
                agent.learn(batch_size)
                # env.render()

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
        Return.append(score)

    env.close()
    plotimage()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()
