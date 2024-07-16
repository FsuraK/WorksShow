import os, math, rvo2
from util.NetUtils import net
import torch
from env.UsvBaseEnvTensorM1 import USV
import numpy as np
import matplotlib.pyplot as plt
from util.BaseUtils import rotate_points, cal_pref_velocity, cal_target_pos, all_arrive_target_pos


def plot4(pxy_lst, exy_lst):
    # data
    px = [row[0] for row in pxy_lst]
    py = [row[1] for row in pxy_lst]
    ex = [row[0] for row in exy_lst]
    ey = [row[1] for row in exy_lst]

    # x-axis and plot init
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig, axs = plt.subplots()

    # plot 1
    axs.plot(px, py, label='pxy')
    axs.plot(ex, ey, label='exy')
    axs.set_title("--traj--")
    axs.legend()
    axs.set_xlim([-5, 70])
    axs.set_ylim([-5, 70])

    # plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # fig.suptitle("Tracking effect", fontsize=20)
    plt.savefig(os.getcwd() + '/AC_track_result1.pdf')
    plt.show()


class Tracking:
    def __init__(self, lr=0.01, model_path=None, input_dim=4, output_dim=2):
        self.model = net(input_dim=input_dim, output_dim=output_dim)
        self.model_path = model_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        if self.model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # print(loss.grad)
        self.optimizer.step()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def cal_u(self, state):
        state = torch.tensor(state.T, dtype=torch.float32, requires_grad=True)
        u = self.model(state)
        return u


if __name__ == "__main__":
    """all args"""
    max_speed = 6
    num_agents = 4
    # num_obst = 1
    time_step = 1 / 100.
    rotate_speed_rad = math.pi / 15
    # radius=1 or 1.5 is look good
    sim = rvo2.PyRVOSimulator(timeStep=time_step, neighborDist=20, maxNeighbors=4, timeHorizon=2, timeHorizonObst=0.8,
                              radius=1.3, maxSpeed=max_speed)

    # init
    now_path = os.getcwd()
    save_path = now_path + '/AC_track_model0.pt'
    # model_path = save_path
    model_path = None
    env = USV((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), time_interval=0.01)
    trackNN = Tracking(lr=5e-6, model_path=model_path, input_dim=8, output_dim=2)
    steps = 200
    episodes = 5000

    # episode start
    if model_path is None:
        for episode in range(episodes):
            env.reset()
            usv_eta = env.eta
            usv_v = env.v
            exy = torch.zeros((2, 1), dtype=torch.float32)
            exy[0] = 10
            exy[1] = 10
            episode_loss = 0

            # steps start
            for _ in range(steps):
                state = torch.cat([usv_eta, usv_v, exy])
                u = trackNN.model(state.T) * 10
                x_next, _ = env.step(u)
                if torch.isnan(x_next).any():
                    print("--" * 15, "Nan", "--" * 15)
                    break
                # loss cal  &  update
                loss = (x_next[:2] - exy).T @ (x_next[:2] - exy)
                trackNN.train(loss)
                env.grad_free()

                # position update
                episode_loss = episode_loss + loss
                # exy[0] = exy[0] + 0.2
                # exy[1] = exy[1] + 0.2
                e_path_1 = 1
                if exy[1] <= 30 and (exy[0] <= 10):
                    e_path_1 = 0

                if e_path_1 == 1:
                    theta0 = np.arccos(exy[0] / 20 - 1.5)
                    theta = theta0 if exy[1] > 30 else -theta0
                    exy[0] = 30 + 20 * np.cos(theta - 0.6 / 20)
                    exy[1] = 30 + 20 * np.sin(theta - 0.6 / 20)
                else:
                    exy[1] = exy[1] + 0.6
                x_now = x_next.detach()
            print("episode =", episode+1, "  episode_loss =", episode_loss.tolist()[0][0])
            if (episode % 800 == 0) or (episode == episodes-1):
                trackNN.save_model(save_path)
    else:
        x_now = env.reset()
        exy = torch.ones((env_arg.pxy_dim, 1), dtype=torch.float32)
        exy[0] = 10
        exy[1] = 10
        pxy_lst = []
        exy_lst = []
        pxy_lst.append(x_now[:2].tolist())
        exy_lst.append(exy.tolist())
        for _ in range(steps):
            state = torch.cat([x_now, exy])
            u = trackNN.model(state.T) * 10
            x_next, done, infos = env.step(x_now, u)

            x_now = x_next
            # position update
            # exy[0] = exy[0] + 0.1
            # exy[1] = exy[1] + 0.1
            """evader 圆形轨迹"""
            e_path_1 = 1
            if exy[1] <= 30 and (exy[0] <= 10):
                e_path_1 = 0

            if e_path_1 == 1:
                theta0 = np.arccos(exy[0] / 20 - 1.5)
                theta = theta0 if exy[1] > 30 else -theta0
                exy[0] = 30 + 20 * np.cos(theta - 0.6 / 20)
                exy[1] = 30 + 20 * np.sin(theta - 0.6 / 20)
            else:
                exy[1] = exy[1] + 0.6

            # lst store for plt
            pxy_lst.append(x_now[:2].tolist())
            exy_lst.append(exy.tolist())
        plot4(pxy_lst, exy_lst)
