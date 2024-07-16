import numpy as np
import torch
import torch.nn as nn
from algorithms.utils.util import init, check
import os
import matplotlib.pyplot as plt


def loss_cal(obs, dvalue_dx, u):
    dvalue_dx = dvalue_dx.T
    x_now_raw, x_now_obs = torch.chunk(obs, 2, dim=0)
    A = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=torch.float32)
    Q = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    F = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    G = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    R = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    R_inv = torch.inverse(R)
    v_star = -0.5 * R_inv @ (F.T @ dvalue_dx + G @ u)
    x_next_raw, x_next_raw_without_dis = raw_sys(x_now_raw, u, x_now_raw_without_dis)
    x_next_obs = obs_sys(x_now_obs, u, v_star)
    loss = (x_next_raw - x_next_obs).T @ (x_next_raw - x_next_obs)
    return loss, x_next_raw.detach(), x_next_obs.detach(), x_next_raw_without_dis.detach()


def loss_cal_for_marl(obs, dvalue_dx, u):
    dvalue_dx = dvalue_dx.T
    x_now_raw, x_now_obs = torch.chunk(obs, 2, dim=0)
    F = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    G = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    R = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    R_inv = torch.inverse(R)
    v_star = -0.5 * R_inv @ (F.T @ dvalue_dx + G @ u)
    x_next_raw = raw_sys_for_marl(x_now_raw, u)
    x_next_obs = obs_sys(x_now_obs, u, v_star)
    loss = (x_next_raw - x_next_obs).T @ (x_next_raw - x_next_obs)
    return loss, x_next_raw.detach(), x_next_obs.detach()


def raw_sys(x_now_raw, u, x_now_raw_without_dis):
    t = 0.1
    A = torch.tensor([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [t, 0], [0, t]], dtype=torch.float32)
    F = torch.tensor([[t, 0], [0, t], [t, 0], [0, t]], dtype=torch.float32)
    # dis = torch.rand(2, 1)*2 - 1
    dis = torch.rand(2, 1) - 0.5
    dis = dis.type(torch.float32)

    x_next_raw = A @ x_now_raw + B @ u + F @ (dis / dis) * 0.5
    # x_next_raw = A @ x_now_raw + B @ u + F @ dis
    x_next_raw_without_dis = A @ x_now_raw_without_dis + B @ u
    return x_next_raw, x_next_raw_without_dis


def obs_sys(x_now_obs, u, v_star):
    """ take: F= [[1,0],[0,1],[1,0],[0,1]]
        then: v[0]=v1, v[1]=v2, v[2]=v1, v[3]=v2
    """
    t = 0.1
    A = torch.tensor([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [t, 0], [0, t]], dtype=torch.float32)
    F = torch.tensor([[t, 0], [0, t], [t, 0], [0, t]], dtype=torch.float32)

    x_next_obs = A @ x_now_obs + B @ u + F @ v_star
    return x_next_obs


def raw_sys_for_marl(x_now_raw, u):
    t = 0.1
    A = torch.tensor([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0], [t, 0], [0, t]], dtype=torch.float32)

    x_next_raw = A @ x_now_raw + B @ u
    return x_next_raw


def plotimgs(x_raw_lst, x_obs_lst):
    """
        1. import np, plt
        2. 功能：画布的一上一下画两个图，可以用来查看两个损失的曲线
        3. 需要在外面定义lst1, lst2:
           lst1 = []
           lst2 = []
           ......
           使用时直接append神经网络的loss输出，不需要中间处理
           lst1.append(data)
           lst1.append(data)
        """
    lst1 = [row[0] for row in x_raw_lst]
    lst2 = [row[0] for row in x_obs_lst]
    plt.close()
    x = np.arange(0, len(lst1), 1)
    plt.figure("Big Title")
    # 主要在这下一行，plt.plot中同时传入两个曲线
    plt.plot(x, np.array(lst1), "r", x, np.array(lst2), "g--")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title("Little Title")
    plt.show()


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


def plot4_for_marl(x_raw_lst, x_obs_lst):
    # data
    x1_raw = [row[0] for row in x_raw_lst]
    x2_raw = [row[1] for row in x_raw_lst]
    x3_raw = [row[2] for row in x_raw_lst]
    x4_raw = [row[3] for row in x_raw_lst]
    x1_obs = [row[0] for row in x_obs_lst]
    x2_obs = [row[1] for row in x_obs_lst]
    x3_obs = [row[2] for row in x_obs_lst]
    x4_obs = [row[3] for row in x_obs_lst]


    # x-axis and plot init
    fig, axs = plt.subplots(4, 1,figsize=(12, 8))
    x = np.arange(0, len(x1_raw), 1)

    # plot 1
    axs[0].plot(x, x1_raw, label='sub-system')
    axs[0].plot(x, x1_obs, label='sub-observer')
    axs[0].set_title("x1")
    axs[0].set_xlabel("Environment total steps")
    axs[0].set_ylabel("Value of state variable")
    axs[0].legend()
    # plot 2
    axs[1].plot(x, x2_raw, label='sub-system')
    axs[1].plot(x, x2_obs, label='sub-observer')
    axs[1].set_title("x2")
    axs[1].set_xlabel("Environment total steps")
    axs[1].set_ylabel("Value of state variable")
    axs[1].legend()
    # plot 3
    axs[2].plot(x, x3_raw, label='sub-system')
    axs[2].plot(x, x3_obs, label='sub-observer')
    axs[2].set_title("x3")
    axs[2].set_xlabel("Environment total steps")
    axs[2].set_ylabel("Value of state variable")
    axs[2].legend()
    # plot 4
    axs[3].plot(x, x4_raw, label='sub-system')
    axs[3].plot(x, x4_obs, label='sub-observer')
    axs[3].set_title("x4")
    axs[3].set_xlabel("Environment total steps")
    axs[3].set_ylabel("Value of state variable")
    axs[3].legend()

    plt.subplots_adjust(hspace=0.8)
    # fig.suptitle("Tracking effect", fontsize=20)
    plt.savefig('/home/lyy/Desktop/happo_imgs/observer_obs/result.eps', bbox_inches="tight")
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
        self.model = net(input_dim=8, output_dim=4)

        self.model_path = model_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        if self.model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def update(self, reward, x_now_obs):
        # reward = torch.tensor(reward, dtype=torch.float32)
        # hj_pred = reward + self.model(x_now_obs)
        hj_true = torch.tensor(0, dtype=torch.float32, requires_grad=False)

        loss = self.loss_fn(reward, hj_true)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, x_now_raw, x_now_obs, u):
        self.optimizer.zero_grad()
        obs = torch.cat([x_now_raw, x_now_obs])
        dvalue_dx = self.model(obs.T)

        loss, x_next_raw, x_next_obs, x_next_raw_without_dis = loss_cal(obs, dvalue_dx, u)
        loss.backward()
        self.optimizer.step()
        return loss, x_next_raw, x_next_obs, x_next_raw_without_dis

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
    save_path = now_path + '/USV_model.pt'
    # model_path = None
    model_path = now_path + '/ADP_model.pt'
    # super param
    episodes = 3000
    lr = 5e-6
    # init
    adp = ADP(lr=lr, model_path=model_path)
    time_interval = 0.1
    x_obs_lst = []
    x_raw_lst = []
    x_raw_without_dis_lst = []

    # x_now = np.zeros((4,1))  # x_now.shape = (4,1)

    # train
    p_num = 0
    if model_path is None:
        for episode in range(episodes):
            x_now_raw = torch.zeros((4, 1), dtype=torch.float32)  # x_now.shape = (4,1)
            x_now_obs = torch.ones((4, 1), dtype=torch.float32) * 2
            x_now_raw_without_dis = torch.zeros((4, 1), dtype=torch.float32)

            episode_reward_sum = 0
            # v_star = np.zeros((4, 1))
            for step in range(200):
                if episode == episodes - 1:
                    x_raw_lst.append(x_now_raw.tolist())
                    x_obs_lst.append(x_now_obs.tolist())
                    x_raw_without_dis_lst.append(x_now_raw_without_dis.tolist())

                # u = np.random.rand(2, 1) * 20 - 10
                u = torch.tensor([[np.sin(step / 10)], [np.cos(step / 10)]], dtype=torch.float32)
                loss, x_now_raw, x_now_obs, x_now_raw_without_dis = adp.train(x_now_raw, x_now_obs, u)

                '''with u, v'''
                # hj_pred = erro_x.T @ erro_x - 0.25*u.T@G@R_inv@G@u - 0.25*u.T@G@R_inv.T@F.T@dvalue_dx + \
                #           dvalue_dx.T@(A@x_now_obs + B@u) - 0.25*dvalue_dx.T@F@R_inv@F.T@dvalue_dx - \
                #           0.25*dvalue_dx.T@F@R_inv@G@u
                '''without u, v'''
                # hj_pred = erro_x.T @ erro_x + \
                #           dvalue_dx.T @ A @ x_now_obs - 0.25 * dvalue_dx.T @ F @ R_inv @ F.T @ dvalue_dx
                '''use Utility'''
                # hj_pred = erro_x.T @ erro_x

                '''update'''
                # adp.update(hj_pred, x_now_obs)
                # adp.optimizer.zero_grad()
                # hj_pred.backward()
                # adp.optimizer.step()

                # episode_reward_sum += hj_pred
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
            x_raw_without_dis_lst.append(x_now_raw_without_dis.tolist())

            obs = torch.cat([x_now_raw, x_now_obs])
            dvalue_dx = adp.model(obs.T)
            u = torch.tensor([[np.sin(step / 10)], [np.cos(step / 10)]], dtype=torch.float32)
            _, x_now_raw, x_now_obs, x_now_raw_without_dis = loss_cal(obs, dvalue_dx, u)

    adp.save_model(model_path=save_path)
    plot4(x_raw_lst, x_obs_lst, x_raw_without_dis_lst)
