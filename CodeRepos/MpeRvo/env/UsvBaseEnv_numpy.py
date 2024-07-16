import numpy as np
# from scipy.integrate import RK45


class USV:
    def __init__(self, init_pos_, init_speed_=None, time_interval=0.1):
        # position and velocity param
        self.x, self.y, self.psi = init_pos_
        self.u, self.upsilon, self.r = (0.0, 0.0, 0.0) if init_speed_ is None else init_speed_

        # control param
        self.tau_u, self.tau_upsilon, self.tau_r = 0, 0, 0

        # init eta, v, tau ---vector
        self.eta = np.array([self.x, self.y, self.psi]).reshape(3, 1)
        self.v = np.array([self.u, self.upsilon, self.r]).reshape(3, 1)
        self.tau = np.array([self.tau_u, self.tau_upsilon, self.tau_r]).reshape(3, 1)

        # other param
        self.eta_d, self.v_d = None, None
        self.time_interval = time_interval
        self.time = 0
        self.d = np.zeros((3, 1), dtype=float)

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

        self.update_state()

    def reset(self, init_pos_=None, init_speed_=None):
        self.update_state(init_pos_, init_speed_)

    def step(self, tau, init_pos_=None, init_speed_=None):
        # is start in any state
        self.update_state(init_pos_, init_speed_)

        # update position and speed using rk4
        self.rk4(tau)

        return self.eta, self.v

    def rk4(self, tau):
        """ using Runge-Kutta 4th order
        rk4 will update speed --- self.v = [u, upsilon, r]
        """
        # self.eta_d = self.R @ self.v
        # self.v_d = np.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        v_swap = self.v
        eta_swap = self.eta

        k1_eta = self.R @ self.v
        k1_v = np.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = v_swap + self.time_interval / 2 * k1_v
        self.eta = eta_swap + self.time_interval / 2 * k1_eta
        self.update_state()

        k2_eta = self.R @ self.v
        k2_v = np.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = v_swap + self.time_interval / 2 * k2_v
        self.eta = eta_swap + self.time_interval / 2 * k2_eta
        self.update_state()

        k3_eta = self.R @ self.v
        k3_v = np.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        self.v = v_swap + self.time_interval * k3_v
        self.eta = eta_swap + self.time_interval * k3_eta
        self.update_state()

        k4_eta = self.R @ self.v
        k4_v = np.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)

        # update self.v == new v
        self.eta = eta_swap + self.time_interval / 6 * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        self.v = v_swap + self.time_interval / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        self.update_state()

    def return_v_d(self, tau, v_swap=None):
        if v_swap is not None:
            self.v = v_swap
            self.update_state()
        self.v_d = np.linalg.inv(self.M) @ (tau + self.d - self.C @ self.v - self.D @ self.v)
        return self.v_d

    def update_state(self, init_pos_=None, init_speed_=None, no_return=True):

        if init_speed_ is not None:
            init_speed_ = np.array(init_speed_).reshape(3, 1)
            self.v = init_speed_
        elif init_pos_ is not None:
            init_pos_ = np.array(init_pos_).reshape(3, 1)
            self.eta = init_pos_
        # update position, speed, control single param from vector
        self.x, self.y, self.psi = [i.item() for i in self.eta]
        self.u, self.upsilon, self.r = [i.item() for i in self.v]
        self.tau_u, self.tau_upsilon, self.tau_r = [i.item() for i in self.tau]

        # self.M intermediate param
        m11 = self.m - self.X_u_d
        m22 = self.m - self.Y_v_d
        m23 = self.m * self.X_v_d - self.Y_r_d
        m32 = m23
        m33 = self.I_z - self.N_r_d
        # self.C intermediate param
        c13 = m22 * self.upsilon + 0.5 * (m23 + m32) * self.r
        c31 = -c13
        c23 = m11 * self.u
        c32 = -c23
        # self.D intermediate param
        d11 = -self.X_u - self.X_uu * abs(self.u)
        d22 = -self.Y_v - self.Y_vv * abs(self.upsilon) - self.Y_rv * abs(self.r)
        d23 = -self.Y_r - self.Y_vr * abs(self.upsilon) - self.Y_rr * abs(self.r)
        d32 = -self.N_v - self.N_vv * abs(self.upsilon) - self.N_rv * abs(self.r)
        d33 = -self.N_r - self.N_vr * abs(self.upsilon) - self.N_rr * abs(self.r)
        # self.R intermediate param
        r11, r12 = np.cos(self.psi), -np.sin(self.psi)
        r21, r22 = np.sin(self.psi), np.cos(self.psi)

        # matrices
        self.M = np.array([[m11, 0, 0],
                           [0, m22, m23],
                           [0, m32, m33]])
        self.C = np.array([[0, 0, c13],
                           [0, 0, c23],
                           [c31, c32, 0]])
        self.D = np.array([[d11, 0, 0],
                           [0, d22, d23],
                           [0, d32, d33]])
        self.R = np.array([[r11, r12, 0],
                           [r21, r22, 0],
                           [0, 0, 1]])
        if not no_return:
            return self.C, self.D

    def cal_vxy(self):
        self.update_state()
        v_x_y_psi = self.R @ self.v
        vx = v_x_y_psi[0].item()
        vy = v_x_y_psi[1].item()
        return vx, vy

