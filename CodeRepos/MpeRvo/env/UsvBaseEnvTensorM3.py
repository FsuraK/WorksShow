"""
    This USV model is a pre-discrete model, that comes from:
    --- <<Reinforcement Learning from Demonstrations by Novel Interactive Expert and
          Application to Automatic Berthing Control Systems for Unmanned Surface Vessel>>
    as pre-discrete, without using rk4 or euler method for solving numerical solutions.

"""
import torch


class USV:
    def __init__(self, init_pos_, init_speed_=None, time_interval=0.1):
        # position and velocity param
        self.x, self.y, self.psi = init_pos_
        self.u, self.upsilon, self.r = (0.0, 0.0, 0.0) if init_speed_ is None else init_speed_
        self.init_speed = torch.zeros((3, 1), dtype=torch.float) if init_speed_ is None \
            else torch.tensor(init_speed_).reshape(3, 1)
        self.init_pos = torch.tensor(init_pos_).reshape(3, 1)

        # control param
        self.tau_u, self.tau_upsilon, self.tau_r = 0, 0, 0

        # init eta, v, tau ---vector
        self.eta = torch.tensor([self.x, self.y, self.psi]).reshape(3, 1)
        self.v = torch.tensor([self.u, self.upsilon, self.r]).reshape(3, 1)
        self.tau = torch.tensor([self.tau_u, self.tau_upsilon, self.tau_r]).reshape(3, 1)

        # other param
        self.eta_d, self.v_d = None, None
        self.time_interval = time_interval
        self.time = 0
        self.d = torch.zeros((3, 1), dtype=torch.float)

        # param setting
        self.m = 17.6
        self.m11 = 19.0
        self.m22 = 35.2
        self.m33 = 4.2
        self.d11 = 4.0
        self.d22 = 10.0
        self.d33 = 1.0

        # matrices
        self.M, self.C, self.D, self.R = None, None, None, None

        self.reset()

    def reset(self):
        self.tau = torch.zeros((3, 1), dtype=torch.float)
        self.v = self.init_speed.clone().detach()
        self.eta = self.init_pos.clone().detach()
        self.update_state()

    def step(self, tau):
        """tau = [tau_u, 0, tau_r] --- tensor"""
        v_swap = self.v.clone()
        eta_swap = self.eta.clone()

        v_new = torch.zeros_like(self.v)
        v_new[0] = v_swap[0] + (self.m22 / self.m11 * v_swap[1] * v_swap[2]
                                - self.d11 / self.m11 * v_swap[0] + 1 / self.m11 * tau[0]) * 0.01
        v_new[1] = v_swap[1] - (self.m11 / self.m22 * v_swap[0] * v_swap[2]
                                - self.d22 / self.m22 * v_swap[1] ) * 0.01
        v_new[2] = v_swap[2] + ((self.m11 - self.m22) / self.m33 * v_swap[0] * v_swap[1]
                                - self.d33 / self.m33 * v_swap[2] + 1 / self.m33 * tau[2]) * 0.01

        eta_new = torch.zeros_like(self.eta)
        eta_new[0] = eta_swap[0] + (torch.cos(v_swap[2]) * v_swap[0]
                                    - torch.sin(v_swap[2]) * v_swap[1]) * 0.01
        eta_new[1] = eta_swap[1] + (torch.sin(v_swap[2]) * v_swap[0]
                                    + torch.cos(v_swap[2]) * v_swap[1]) * 0.01
        eta_new[2] = eta_swap[2] + (v_swap[2]) * 0.01

        self.v = v_new
        self.eta = eta_new

        self.update_state()

        return self.eta, self.v

    def return_v_d(self, tau, v_swap=None):
        if v_swap is not None:
            self.v = v_swap
            self.update_state()
        self.v_d = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        return self.v_d

    def update_state(self, init_pos_=None, init_speed_=None, return_=False):

        if init_speed_ is not None:
            self.v = torch.tensor(init_speed_).reshape(3, 1)
        elif init_pos_ is not None:
            self.eta = torch.tensor(init_pos_).reshape(3, 1)
        # update position, speed, control single param from vector
        self.x, self.y, self.psi = [i for i in self.eta]
        self.u, self.upsilon, self.r = [i for i in self.v]
        self.tau_u, self.tau_upsilon, self.tau_r = [i for i in self.tau]

        # # selfC intermediate param
        # c13 = -self.m22 * self.v[1]
        # c23 = self.m11 * self.v[0]
        # c31 = self.m11 * self.v[1]
        # c32 = -self.m11 * self.v[0]
        #
        # # self.R intermediate param
        r11, r12 = torch.cos(self.eta[2]), -torch.sin(self.eta[2])
        r21, r22 = torch.sin(self.eta[2]), torch.cos(self.eta[2])
        #
        # # matrices
        # self.M = torch.tensor([[self.m11, 0, 0],
        #                        [0, self.m22, 0],
        #                        [0, 0, self.m33]])
        # self.C = torch.tensor([[0, 0, c13],
        #                        [0, 0, c23],
        #                        [c31, c32, 0]])
        # self.D = torch.tensor([[d11, 0, 0],
        #                        [0, d22, d23],
        #                        [0, d32, d33]])
        self.R = torch.tensor([[r11, r12, 0],
                               [r21, r22, 0],
                               [0, 0, 1]])
        if return_:
            return self.C, self.D

    def cal_vxy(self):
        # self.update_state()
        v_x_y_psi = self.R @ self.v
        vx = v_x_y_psi[0]
        vy = v_x_y_psi[1]
        return (vx, vy), v_x_y_psi[:2]

    def grad_free(self):
        self.v = self.v.detach()
        self.eta = self.eta.detach()
