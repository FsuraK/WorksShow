from gym.spaces import Box
import numpy as np
from collections import namedtuple
import torch


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class Single_USV_Unlin_Env:
    """数据用numpy 输入输出判断是否需要和tensor转换"""
    def __init__(self, args, seed=None):
        super(Single_USV_Unlin_Env).__init__()
        self.args = args
        self.steps = 0
        self.u_dim = args.u_dim
        self.x_dim = args.x_dim  #
        self.pxy_dim = args.pxy_dim
        self.n_agents = args.n_agents
        self.seed = seed
        self.time_interval = args.time_interval # 0.01
        self.dis_time = 0

        # matrices

        # init {ndarray:{4,1}}
        self.x = torch.zeros((self.x_dim, self.n_agents))
        self.u = torch.zeros((self.u_dim, self.n_agents))

    def seed(self, args):
        pass

    def reset(self):
        self.x = torch.zeros((self.x_dim, self.n_agents))
        # 此处把USV的所有初始化一下，下面目前写了两个
        self.x[0] = 0
        self.x[1] = 0
        self.x[2] = 0
        self.x[3] = 0

        return self.x

    def step(self, x_now, u):
        """make sure that actions of n_rollout_threads envs all be done"""
        # self.steps += 1
        # self.dis_time += 1
        # matrices
        t = self.time_interval
        A = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
        B = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=torch.float32)
        dx = A @ x_now + B @ u.T
        # x_next = A @ x_now + B @ u.T

        x_next = torch.zeros((4, 1), dtype=torch.float32)
        x_next[2] = x_now[2] + dx[2] * t
        x_next[3] = x_now[3] + dx[3] * t
        x_next[0] = x_now[0] + x_next[2] * t
        x_next[1] = x_now[1] + x_next[3] * t

        done = np.zeros((self.args.n_rollout_threads, self.n_agents), dtype=bool)
        infos = {}

        return x_next, done, infos

    def render(self):
        pass

    def close(self):
        print("env close")
