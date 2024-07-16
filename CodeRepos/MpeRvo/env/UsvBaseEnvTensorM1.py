import math
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
        self.m = 23.8
        self.I_z = 1.76
        self.X_u = -0.7225
        self.Y_v = -0.8612
        self.Y_r = 0.1079
        self.N_v = 0.1052
        self.N_r = -1.9
        self.X_uu = -1.3274
        self.Y_vv = -36.2832
        self.Y_rv = -0.805
        self.Y_vr = -0.845
        self.Y_rr = -0.345
        self.N_rr = -0.75
        self.N_vv = 5.0437
        self.N_rv = 0.13
        self.N_vr = 0.08
        self.X_v_d = 0.046
        self.Y_r_d = 0.0
        self.N_r_d = -1.0
        self.Y_v_d = -10.0
        self.N_v_d = 0.0
        self.X_u_d = -2.0
        # matrices
        self.M, self.C, self.D, self.R = None, None, None, None

        self.reset()

    def reset(self):
        self.tau = torch.zeros((3, 1), dtype=torch.float)
        self.v = self.init_speed.clone().detach()
        self.eta = self.init_pos.clone().detach()
        self.update_state()

    def step(self, tau, init_pos_=None, init_speed_=None):
        if tau.size() == torch.Size([1, 2]):
            tau = tau.reshape(2, )
            zero = torch.tensor([0.0])
            tau = torch.cat((tau[:1], zero, tau[1:]))

        tau = tau.reshape(3, 1)

        # update position and speed using rk4 or euler method
        # self.rk4(tau)
        self.euler_test(tau)

        return self.eta, self.v

    def euler_test(self, tau):
        dv = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = self.v + dv * self.time_interval
        self.update_state()
        deta = self.R @ self.v
        self.eta = self.eta + deta * self.time_interval
        self.update_state()

    def euler(self, tau):
        deta = self.R @ self.v
        # 计算v的导数
        dv = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        # 计算eta和v在time_interval时刻的数值解
        self.eta = self.eta + deta * self.time_interval
        self.v = self.v + dv * self.time_interval
        self.update_state()
        # return dv

    def rk4(self, tau):
        """ using Runge-Kutta 4th order
        rk4 will update speed --- self.v = [u, upsilon, r]
        """
        v_swap = self.v
        eta_swap = self.eta

        k1_eta = self.R @ self.v
        k1_v = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = v_swap + self.time_interval / 2 * k1_v
        self.eta = eta_swap + self.time_interval / 2 * k1_eta
        self.update_state()

        k2_eta = self.R @ self.v
        k2_v = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = v_swap + self.time_interval / 2 * k2_v
        self.eta = eta_swap + self.time_interval / 2 * k2_eta
        self.update_state()

        k3_eta = self.R @ self.v
        k3_v = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = v_swap + self.time_interval * k3_v
        self.eta = eta_swap + self.time_interval * k3_eta
        self.update_state()

        k4_eta = self.R @ self.v
        k4_v = torch.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)

        # update self.v == new v
        self.eta = eta_swap + self.time_interval / 6 * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        self.v = v_swap + self.time_interval / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        self.update_state()

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

        # self.M intermediate param
        m11 = self.m - self.X_u_d
        m22 = self.m - self.Y_v_d
        m23 = self.m * self.X_v_d - self.Y_r_d
        m32 = m23
        m33 = self.I_z - self.N_r_d
        # self.C intermediate param
        c13 = m22 * self.v[1] + 0.5 * (m23 + m32) * self.v[2]
        c31 = -c13
        c23 = m11 * self.v[0]
        c32 = -c23
        # self.D intermediate param
        d11 = -self.X_u - self.X_uu * abs(self.v[0])
        d22 = -self.Y_v - self.Y_vv * abs(self.v[1]) - self.Y_rv * abs(self.v[2])
        d23 = -self.Y_r - self.Y_vr * abs(self.v[1]) - self.Y_rr * abs(self.v[2])
        d32 = -self.N_v - self.N_vv * abs(self.v[1]) - self.N_rv * abs(self.v[2])
        d33 = -self.N_r - self.N_vr * abs(self.v[1]) - self.N_rr * abs(self.v[2])
        # self.R intermediate param
        # psi = torch.tensor(math.radians(self.eta[2]))
        psi = self.eta[2]
        r11, r12 = torch.cos(psi), -torch.sin(psi)
        r21, r22 = torch.sin(psi), torch.cos(psi)

        # matrices
        self.M = torch.tensor([[m11, 0, 0],
                               [0, m22, m23],
                               [0, m32, m33]])
        self.C = torch.tensor([[0, 0, c13],
                               [0, 0, c23],
                               [c31, c32, 0]])
        self.D = torch.tensor([[d11, 0, 0],
                               [0, d22, d23],
                               [0, d32, d33]])
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
