import random

import torch


class USV:
    def __init__(self, init_pos_, init_speed_=None, time_interval=0.1):
        self.init_pos = init_pos_
        self.init_speed = init_speed_
        self.vxy = torch.zeros((2, 1), dtype=torch.float) if init_speed_ is None \
            else torch.tensor(init_speed_).reshape(2, 1)

        self.pxy = torch.tensor(init_pos_).reshape(2, 1)
        self.time_interval = time_interval
        self.observation_dim = 4
        self.action_dim = 2

    def step(self, tau):
        if tau.size is not torch.Size([2, 1]):
            tau = tau.view(2, 1)

        # self.ca(tau)
        self.euler(tau)
        # self.rk3(tau)

    def euler(self, tau):
        self.vxy = self.vxy + tau * self.time_interval
        self.pxy = self.pxy + self.vxy * self.time_interval

    def ca(self, tau):
        self.pxy = self.pxy + self.vxy * self.time_interval + 0.5 * tau * self.time_interval * self.time_interval
        self.vxy = self.vxy + tau * self.time_interval

    def rk3(self, tau):
        v = self.vxy
        p = self.pxy

        k1_p = v
        k_v = tau

        v1 = v + 0.5 * self.time_interval * k_v
        k2_p = v1

        v2 = v - self.time_interval * k_v + 2 * self.time_interval * k_v
        k3_p = v2

        self.pxy = p + (self.time_interval / 6) * (k1_p + 4*k2_p + k3_p)
        self.vxy = v + self.time_interval * tau

    def rk4(self, tau):
        v_swap = self.vxy
        p_swap = self.pxy

        k1_v = tau * self.time_interval
        self.vxy = v_swap + self.time_interval / 2 * k1_v
        k1_p = self.vxy * self.time_interval
        self.pxy = p_swap + self.time_interval / 2 * k1_p

        k2_v = tau * self.time_interval
        self.vxy = v_swap + self.time_interval / 2 * k2_v
        k2_p = self.vxy * self.time_interval
        self.pxy = p_swap + self.time_interval / 2 * k2_p

        k3_v = tau * self.time_interval
        self.vxy = v_swap + self.time_interval * k3_v
        k3_p = self.vxy * self.time_interval
        self.pxy = p_swap + self.time_interval * k3_p

        k4_v = tau * self.time_interval
        self.vxy = v_swap + self.time_interval / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        k4_p = self.vxy * self.time_interval
        self.pxy = p_swap + self.time_interval / 6 * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)

    def cal_vxy(self):
        vx = self.vxy[0]
        vy = self.vxy[1]
        return vx, vy

    def grad_free(self):
        self.vxy = self.vxy.detach()
        self.pxy = self.pxy.detach()

    def reset(self):
        self.vxy = torch.zeros((2, 1), dtype=torch.float) if self.init_speed is None \
            else torch.tensor(self.init_speed).reshape(2, 1)
        self.pxy = torch.tensor(self.init_pos).reshape(2, 1)

    def get_state(self):
        state = torch.cat((self.pxy, self.vxy), dim=0)
        return state


