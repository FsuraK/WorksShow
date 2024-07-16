import numpy as np
import torch
import torch.nn as nn
from algorithms.utils.util import init, check
import os


def raw_sys(x_now_raw, u):
    x_now_raw = x_now_raw.numpy()
    u = u.numpy()
    time_interval = 0.1
    x_now_raw[2] = x_now_raw[2] + u[0] * time_interval
    x_now_raw[3] = x_now_raw[3] + u[1] * time_interval
    x_now_raw[0] = x_now_raw[0] + x_now_raw[2] * time_interval
    x_now_raw[1] = x_now_raw[1] + x_now_raw[3] * time_interval

    x_next_raw = x_now_raw
    return x_next_raw


def obs_sys(x_now_obs, u, v_star):
    """ take: F= [[1,0],[0,1],[1,0],[0,1]]
        then: v[0]=v1, v[1]=v2, v[2]=v1, v[3]=v2
    """
    x_now_obs = x_now_obs.numpy()
    u = u.numpy()
    v_star = v_star.numpy()

    v = np.zeros((4, 1))
    v[0], v[1], v[2], v[3] = v_star[0], v_star[1], v_star[0], v_star[1]
    time_interval = 0.1

    x_now_obs[2] = x_now_obs[2] + u[0] * time_interval + v[0]
    x_now_obs[3] = x_now_obs[3] + u[1] * time_interval + v[1]
    x_now_obs[0] = x_now_obs[0] + x_now_obs[2] * time_interval + v[2]
    x_now_obs[1] = x_now_obs[1] + x_now_obs[3] * time_interval + v[3]

    x_next_obs = x_now_obs
    return x_next_obs


class net(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, lr=0.001):
        super(net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2.weight.data.normal_(0, 0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ADP:
    def __init__(self, lr=0.01, model_path=None):
        self.model = net(input_dim=4, output_dim=4)

        self.model_path = model_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        if self.model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def update(self, reward, x_now_obs):
        reward = torch.tensor(reward, dtype=torch.float32)
        x_now_obs = torch.tensor(x_now_obs.T, dtype=torch.float32)
        # hj_pred = reward + self.model(x_now_obs)
        hj_true = torch.tensor(0, dtype=torch.float32)

        loss = self.loss_fn(reward, hj_true)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def gradient_calculate(self, x):
        x = torch.tensor(x.T, requires_grad=True, dtype=torch.float32)
        value = self.model(x)
        value.backward()
        dvalue_dx = x.grad.numpy()
        return value, dvalue_dx

    def direct_calculate(self, x):
        x = torch.tensor(x.T, requires_grad=True, dtype=torch.float32)
        dvalue_dx = self.model(x)
        return dvalue_dx

if __name__ == '__main__':
    now_path = os.getcwd()
    save_path = now_path + '/ADP_model.pt'
    # super param
    episodes = 2000
    lr = 0.005
    # init
    adp = ADP(lr=lr, model_path=None)
    time_interval = 0.1
    # A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    # B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    # Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # F = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    # G = np.array([[1, 0], [0, 1]])
    # R = np.array([[1, 0], [0, 1]])
    A = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=torch.float32)
    Q = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    F = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    G = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    R = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    # x_now = np.zeros((4,1))  # x_now.shape = (4,1)

    # train
    for episode in range(episodes):
        x_now_raw = np.zeros((4, 1))  # x_now.shape = (4,1)
        x_now_obs = np.zeros((4, 1))

        episode_reward_sum = 0
        # v_star = np.zeros((4, 1))
        for step in range(200):
            dvalue_dx = adp.direct_calculate(x_now_obs)
            dvalue_dx = dvalue_dx.T
            R_inv = torch.inverse(R)
            # u = np.random.rand(2, 1) * 20 - 10
            u = np.array([[np.sin(step/10)], [np.cos(step/10)]])
            u = torch.tensor(u, dtype=torch.float32)
            v_star = -0.5 * R_inv @ (F.T @ dvalue_dx + G @ u)
            x_now_raw = raw_sys(x_now_raw, u)
            x_now_raw = torch.tensor(x_now_raw, dtype=torch.float32)
            x_now_obs = obs_sys(x_now_obs, u, v_star)
            x_now_obs = torch.tensor(x_now_obs, dtype=torch.float32)
            # erro_x = (x_next_obs-x_next_raw)
            erro_x = (x_now_obs-x_now_raw)

            '''with u, v'''
            # hj_pred = erro_x.T @ erro_x - 0.25*u.T@G@R_inv@G@u - 0.25*u.T@G@R_inv.T@F.T@dvalue_dx + \
            #           dvalue_dx.T@(A@x_now_obs + B@u) - 0.25*dvalue_dx.T@F@R_inv@F.T@dvalue_dx - \
            #           0.25*dvalue_dx.T@F@R_inv@G@u
            '''without u, v'''
            # hj_pred = erro_x.T @ erro_x + \
            #           dvalue_dx.T @ A @ x_now_obs - 0.25 * dvalue_dx.T @ F @ R_inv @ F.T @ dvalue_dx
            '''use Utility'''
            hj_pred = erro_x.T @ erro_x

            reward = -hj_pred * time_interval
            episode_reward_sum += reward

            '''update'''
            adp.update(reward, x_now_obs)

        if episode % 100 == 0:
            print(episode_reward_sum)
    adp.save_model(model_path=save_path)