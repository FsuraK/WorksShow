import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


def loss_cal(obs, u, asdasdasd):
    u = u.T
    x_now_raw = obs[:4]
    x_now_obs = obs[4:]
    A = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=torch.float32)
    Q = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    F = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    G = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    R = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    R_inv = torch.inverse(R)
    x_next_raw = raw_sys(x_now_raw, u)
    x_next_obs = obs_sys(x_now_obs, asdasdasd, u)
    loss = (x_next_raw[:3] - x_next_obs[:3]).T @ (x_next_raw[:3] - x_next_obs[:3])
    return loss, x_next_raw.detach(), x_next_obs.detach()

def raw_sys(x_now_raw, u):
    t = 0.1
    A = torch.tensor([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [t, 0], [0, t]], dtype=torch.float32)
    F = torch.tensor([[t, 0], [0, t], [t, 0], [0, t]], dtype=torch.float32)
    # dis = torch.rand(2, 1)*2 - 1
    dis = torch.rand(2, 1) - 0.5
    dis = dis.type(torch.float32)

    x_next_raw = A @ x_now_raw + B @ u + F @ (dis / dis) * 0.5
    return x_next_raw


def obs_sys(x_now_obs, aaa, u):
    """ take: F= [[1,0],[0,1],[1,0],[0,1]]
        then: v[0]=v1, v[1]=v2, v[2]=v1, v[3]=v2
    """
    t = 0.01
    A = torch.tensor([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [t, 0], [0, t]], dtype=torch.float32)
    F = torch.tensor([[t, 0], [0, t], [t, 0], [0, t]], dtype=torch.float32)

    x_next_obs = A @ x_now_obs + F @ u
    return x_next_obs

def plot4(x_raw_lst, x_obs_lst, x_raw_without_dis_lst):
    # data
    x1_raw = [row[0] for row in x_raw_lst]
    x2_raw = [row[1] for row in x_raw_lst]
    x3_raw = [row[2] for row in x_raw_lst]
    x4_raw = [row[3] for row in x_raw_lst]
    x1_obs = [row[0] for row in x_obs_lst]
    x2_obs = [row[1] for row in x_obs_lst]
    x3_obs = [row[2] for row in x_obs_lst]
    x4_obs = [row[3] for row in x_obs_lst]
    x1_raw_without_dis = [row[0] for row in x_raw_without_dis_lst]
    x2_raw_without_dis = [row[1] for row in x_raw_without_dis_lst]
    x3_raw_without_dis = [row[2] for row in x_raw_without_dis_lst]
    x4_raw_without_dis = [row[3] for row in x_raw_without_dis_lst]

    # x-axis and plot init
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(0, len(x1_raw), 1)

    # plot 1
    axs[0, 0].plot(x, x1_raw, label='raw with dis')
    axs[0, 0].plot(x, x1_obs, label='observer')
    axs[0, 0].plot(x, x1_raw_without_dis, label='raw')
    axs[0, 0].set_title("x1")
    axs[0, 0].legend()
    # plot 2
    axs[0, 1].plot(x, x2_raw, label='raw with dis')
    axs[0, 1].plot(x, x2_obs, label='observer')
    axs[0, 1].plot(x, x2_raw_without_dis, label='raw')
    axs[0, 1].set_title("x2")
    axs[0, 1].legend()
    # plot 3
    axs[1, 0].plot(x, x3_raw, label='raw with dis')
    axs[1, 0].plot(x, x3_obs, label='observer')
    axs[1, 0].plot(x, x3_raw_without_dis, label='raw')
    axs[1, 0].set_title("x3")
    axs[1, 0].legend()
    # plot 4
    axs[1, 1].plot(x, x4_raw, label='raw with dis')
    axs[1, 1].plot(x, x4_obs, label='observer')
    axs[1, 1].plot(x, x4_raw_without_dis, label='raw')
    axs[1, 1].set_title("x4")
    axs[1, 1].legend()

    # plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Tracking effect", fontsize=20)
    plt.savefig(os.getcwd() + '/state_fig5.pdf')
    plt.show()


class net(nn.Module):
    def __init__(self, input_dim=8, output_dim=4, lr=0.001):
        super(net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2.weight.data.normal_(0, 0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ADP:
    def __init__(self, lr=0.01, model_path=None):
        self.model = net(input_dim=8, output_dim=2)

        self.model_path = model_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        if self.model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def train(self, x_now_raw, x_now_obs, u):
        self.optimizer.zero_grad()
        obs = torch.cat([x_now_raw, x_now_obs])
        dvalue_dx = self.model(obs.T)

        loss, x_next_raw, x_next_obs = loss_cal(obs, dvalue_dx, u)
        loss.backward()
        self.optimizer.step()
        return loss, x_next_raw, x_next_obs

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


if __name__ == '__main__':
    now_path = os.getcwd()
    save_path = now_path + '/USV_model.pt'
    model_path = None
    # model_path = now_path + '/ADP_model.pt'
    # super param
    episodes = 3000
    lr = 5e-6
    # init
    adp = ADP(lr=lr, model_path=model_path)
    time_interval = 0.1
    x_obs_lst = []
    x_raw_lst = []

    # train
    p_num = 0
    if model_path is None:
        for episode in range(episodes):
            # init
            x_now_raw = torch.zeros((4, 1), dtype=torch.float32)  # x_now.shape = (4,1)
            x_now_obs = torch.ones((4, 1), dtype=torch.float32) * 2

            episode_reward_sum = 0
            # v_star = np.zeros((4, 1))
            for step in range(200):
                u = torch.tensor([[np.sin(step / 10)], [np.cos(step / 10)]], dtype=torch.float32)
                loss, x_now_raw, x_now_obs = adp.train(x_now_raw, x_now_obs, u)

                if step % 120 == 0 and step > 100:
                    p_num += 1
                    print(loss, 'num = ', p_num)
    else:
        x_now_raw = torch.zeros((4, 1), dtype=torch.float32)
        x_now_obs = torch.ones((4, 1), dtype=torch.float32) * 5
        x_now_raw_without_dis = torch.zeros((4, 1), dtype=torch.float32)
        for step in range(200):
            x_raw_lst.append(x_now_raw.tolist())
            x_obs_lst.append(x_now_obs.tolist())

            obs = torch.cat([x_now_raw, x_now_obs])
            dvalue_dx = adp.model(obs.T)
            u = torch.tensor([[np.sin(step / 10)], [np.cos(step / 10)]], dtype=torch.float32)
            _, x_now_raw, x_now_obs = loss_cal(obs, dvalue_dx, u)

    adp.save_model(model_path=save_path)
