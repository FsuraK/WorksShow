# python3
# Create Dat3: 2022-12-27
# Func: PPO 输出action为连续变量
# =====================================================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import gym
import copy
import random
from collections import deque
from tqdm import tqdm
import typing as typ
import os, math
from env.LinearBaseEnv import USV
from function import rotate_points, cal_pref_velocity, all_arrive_target_pos, cal_target_pos
from plot.plot import plot_traj_static


class policyNet(nn.Module):
    """
    continuity action:
    normal distribution (mean, std)
    """

    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        """hidden_layers_dim: [dim1, dim2, dim3, ...] ---> total hidden layers = hidden_layers_dim.size()"""
        super(policyNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx - 1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True)
            }))

        self.fc_mu = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.fc_std = nn.Linear(hidden_layers_dim[-1], action_dim)

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))

        mean_ = 70.0 * torch.tanh(self.fc_mu(x))
        # np.log(1 + np.exp(2))
        std = F.softplus(self.fc_std(x)) + 1e-4
        return mean_, std


class valueNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim):
        super(valueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx - 1] if idx else state_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))

        self.head = nn.Linear(hidden_layers_dim[-1], 1)

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    adv_list = []
    adv = 0
    for delta in td_delta[::-1]:
        adv = gamma * lmbda * adv + delta
        adv_list.append(adv)
    adv_list.reverse()
    return torch.FloatTensor(adv_list)


class PPO:
    """
    PPO算法, 采用截断方式
    """

    def __init__(self,
                 state_dim: int,
                 hidden_layers_dim: typ.List,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 PPO_kwargs: typ.Dict,
                 device: torch.device
                 ):
        self.actor = policyNet(state_dim, hidden_layers_dim, action_dim).to(device)
        self.critic = valueNet(state_dim, hidden_layers_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = PPO_kwargs['lmbda']
        self.ppo_epochs = PPO_kwargs['ppo_epochs']  # 一条序列的数据用来训练的轮次
        self.eps = PPO_kwargs['eps']  # PPO中截断范围的参数
        self.count = 0
        self.device = device

    def policy(self, state):
        state = torch.FloatTensor(state).to(self.device)
        state = state.to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return action

    def update(self, samples: deque):
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)
        # state = torch.stack([sample[0] for sample in samples]).to(self.device)
        # state = state.squeeze(-1)
        # action = torch.stack([sample[1] for sample in samples]).to(self.device)
        # reward = torch.stack([sample[2] for sample in samples]).to(self.device)
        # reward = (reward + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        # next_state = torch.stack([sample[3] for sample in samples]).to(self.device)
        # next_state = next_state.squeeze(-1)
        # done = torch.stack([sample[4] for sample in samples]).to(self.device)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        reward = (reward + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)

        # td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_target = reward + self.gamma * self.critic(next_state)
        td_delta = td_target - self.critic(state)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        mu, std = self.actor(state)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(action)
        for _ in range(self.ppo_epochs):
            mu, std = self.actor(state)
            action_dists = torch.distributions.Normal(mu, std)
            log_prob = action_dists.log_prob(action)

            # e(log(a/b))
            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2)).float()
            critic_loss = torch.mean(
                F.mse_loss(self.critic(state).float(), td_target.detach().float())
            ).float()
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()


class replayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)


def play(env, env_agent, cfg, episode_count=2):
    for e in range(episode_count):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            env.render()
            a = env_agent.policy(s)
            n_state, reward, done, _, _ = env.step(a)
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if (episode_cnt >= 3 * cfg.max_episode_steps) or (episode_reward >= 3 * cfg.max_episode_rewards):
                break

        print(f'Get reward {episode_reward}. Last {episode_cnt} times')
    env.close()


class Config:
    num_episode = 5000
    state_dim = None
    hidden_layers_dim = [128, 128, 64]
    action_dim = 2
    actor_lr = 5e-5
    critic_lr = 5e-4
    PPO_kwargs = {
        'lmbda': 0.9,
        'eps': 0.2,
        'ppo_epochs': 10
    }
    gamma = 0.9
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 20480
    minimal_size = 1024
    batch_size = 64
    save_path = os.getcwd() + r'/model/ppo_model.ckpt'
    model_path = os.getcwd() + r'/model/ppo_model_250_base.ckpt'
    # 回合停止控制
    max_episode_rewards = 0
    max_episode_steps = 200

    def __init__(self, env):
        self.state_dim = 32
        try:
            self.action_dim = env.action_dim
        except Exception as e:
            self.action_dim = env.action_dim
        print(f'device={self.device} | env={str(env)}')


def train_agent(env, cfg):
    ac_agent = PPO(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device
    )
    if not train or continue_train:
        # ac_agent.actor.load_state_dict(torch.load(cfg.model_path))
        ac_agent.actor.load_state_dict(torch.load(cfg.save_path))

    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    for i in tq_bar:
        buffer_ = replayBuffer(cfg.buffer_size)
        tq_bar.set_description(f'Episode [ {i + 1} / {cfg.num_episode} ]')
        # env reset
        env.reset()
        # evader reset
        evader = np.array([ex, ey], dtype=float)
        target_pos = cal_target_pos(evader, e_angle_4, e_radius)
        angles = [math.atan2(point[1] - evader[1], point[0] - evader[0]) for point in target_pos]
        target_posi_tensor = torch.tensor(target_pos[env_i], dtype=torch.float).view(-1, 1)
        s = torch.cat((env.pxy, env.vxy, target_posi_tensor, obst_tensor), dim=0)
        s = s.view(32, ).numpy()

        """if not train"""
        if not train:
            x_swap, y_swap = [], []
            x_swap.append(s[0].item())
            x_swap.append(evader[0].item())
            y_swap.append(s[1].item())
            y_swap.append(evader[1].item())

            px_lst_usv.append(x_swap)
            py_lst_usv.append(y_swap)

        episode_rewards = 0
        done = False
        for i in range(cfg.max_episode_steps):

            # action - policy
            a = ac_agent.policy(s)

            # env step
            env.step(a)

            # # r: reward
            # r = cal_reward(s)

            # evader step
            evader[0] += 0.12
            evader[1] = 0.02 * (evader[0]) ** 2 - 0.8 * evader[0] + 28
            target_pos, angles = rotate_points(evader, rotate_speed_rad, time_step, e_radius, angles=angles)

            # n_s: next state
            target_posi_tensor = torch.tensor(target_pos[env_i], dtype=torch.float).view(-1, 1)
            n_s = torch.cat((env.pxy, env.vxy, target_posi_tensor, obst_tensor), dim=0)

            # r: reward
            r = cal_reward2(n_s[:6])
            # r = cal_reward(n_s)

            # buffer push
            n_s = n_s.view(32, ).numpy()
            buffer_.add(s, a.numpy(), r, n_s, done)

            # state update
            s = n_s
            episode_rewards += r
            if episode_rewards >= cfg.max_episode_rewards:
                break
            env.grad_free()

            if not train:
                x_swap, y_swap = [], []
                x_swap.append(s[0].item())
                x_swap.append(evader[0].item())
                y_swap.append(s[1].item())
                y_swap.append(evader[1].item())

                px_lst_usv.append(x_swap)
                py_lst_usv.append(y_swap)

        if train:
            ac_agent.update(buffer_.buffer)
            rewards_list.append(episode_rewards)
            now_reward = np.mean(rewards_list[-10:])
            if bf_reward < now_reward:
                torch.save(ac_agent.actor.state_dict(), cfg.save_path)
                bf_reward = now_reward
            tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
        else:
            break


    # env.close()
    return ac_agent


def cal_reward(state):
    """need to create a reward describe the action reward, which means normalization"""
    coef1, coef2 = 0.1, 0.1
    reward_sum = 0

    state_2 = state[:2]
    reward1 = -((state_2 - state[-2:]).T @ (state_2 - state[-2:]))**0.5
    reward_sum = reward_sum + reward1 * coef1
    for obs in obst_pos:
        obs = torch.tensor(obs, dtype=torch.float).view(2, 1)
        distance = ((state_2 - obs).T @ (state_2 - obs)) ** 0.5
        if distance > 6:
            reward2 = 0
        else:
            reward2 = - (5 - distance) / (distance + 0.05)

        reward_sum = reward_sum + reward2 * coef2
    if type(state) == np.ndarray:
        return reward_sum
    else:
        return reward_sum[0].item()


def cal_reward2(state):
    coef1, coef2 = 0.1, 0.1
    reward_sum = 0

    now_pos = state[:2]
    target_pos = state[-2:]
    now_vec = state[2:-2]
    v_pref = cal_pref_velocity(now_pos, target_pos, max_speed)
    v_pref = torch.tensor(v_pref).view(2, 1)
    reward1 = -( (now_vec - v_pref).T @ (now_vec - v_pref) )**0.5

    reward_sum = reward_sum + reward1 * coef1
    for obs in obst_pos:
        obs = torch.tensor(obs, dtype=torch.float).view(2, 1)
        distance = ( (now_pos - obs).T @ (now_pos - obs) )**0.5
        if distance > 6:
            reward2 = 0
        else:
            reward2 = - (5 - distance) / (distance + 0.01)

        reward_sum = reward_sum + reward2 * coef2
    if type(state) == np.ndarray:
        return reward_sum
    else:
        return reward_sum[0][0].item()


if __name__ == '__main__':
    env_i = 3
    max_speed = 6
    env_init_pos = [(5.0, 5.0), (20.0, 5.0), (5.0, 35.0), (5.0, 20.0)]
    time_step = 1 / 10.
    env = USV(env_init_pos[env_i], time_interval=time_step)
    # env = USV((20.0, 5.0), time_interval=time_step)
    # env = USV((5.0, 35.0), time_interval=time_step)
    # env = USV((5.0, 20.0), time_interval=time_step)

    """INIT evader and target pos"""
    ex, ey = 20, 20
    evader = np.array([ex, ey], dtype=float)
    e_angle_4 = [3 * math.pi / 2, 0, math.pi / 2, math.pi]
    e_radius = 12
    target_pos = cal_target_pos(evader, e_angle_4, e_radius)
    rotate_speed_rad = math.pi / 10
    # INIT OBST
    obst_pos = [(ex + 15 * math.cos(math.pi / 4), ey + 15 * math.sin(math.pi / 4)),
                (ex + 15 * math.cos(3 * math.pi / 4), ey + 15 * math.sin(math.pi / 4)),
                (10., 10.),
                (25., 10.),
                (18., 12.),
                (10., 25.),
                (20., 35.),
                (53., 8.),
                (50., 55.),
                (49., 20.),
                (55., 15.),
                (10., 40.),
                (45., 43.),
                ]
    num_obst = len(obst_pos)
    obst_tensor = torch.tensor(obst_pos, dtype= torch.float).view(-1,1)

    print('==' * 35)
    print('Training PPO-MpeRvo')

    # train = False
    # continue_train = False
    train = True
    continue_train = True
    px_lst_usv, py_lst_usv = [], []
    cfg = Config(env)
    ac_agent = train_agent(env, cfg)
    if not train:
        np.savetxt(os.path.dirname(os.getcwd()) + f"/PPO-Clip/data/px_{env_i}.txt", px_lst_usv)
        np.savetxt(os.path.dirname(os.getcwd()) + f"/PPO-Clip/data/py_{env_i}.txt", py_lst_usv)
        plot_traj_static(px_lst_usv, py_lst_usv, obst_pos, num_agents=1, circle_radius=e_radius, xy_lim=[0, 90],
                         background_img=r'/home/liuyangyang/MARL/MpeRvo/plot/2.jpg',
                         save_path=os.path.dirname(os.getcwd()) + "/PPO-Clip/ImgSave/Sota2Rvo.svg")
