import random
from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
from .multiagentenv import MultiAgentEnv
from collections import namedtuple


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class MusvLinerEnv:
    def __init__(self, args, seed=None):
        super(MusvLinerEnv).__init__()
        self.steps = 0
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        self.n_agents = 4
        self.action_dim = 2
        self.obs_dim = 4 + 2  # 暂定2个状态：Px, Py + evader(Px, Py) + nearest agent(Px, Py)
        self.share_obs_dim = 10
        self.vxy_dim = 2
        self.seed = seed
        self.time_interval = 0.1
        self.dis_time = 0
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,)) for _ in
                                        range(self.n_agents)]
        self.action_space = tuple([Box(low=-np.inf, high=+np.inf, shape=(self.action_dim,), dtype=np.float32) for _ in
                                   range(self.n_agents)])

        # {ndarray:{8,8,1}}
        self.obs = np.zeros((self.args.n_rollout_threads, self.n_agents, self.obs_dim))
        self.actions = np.zeros((self.args.n_rollout_threads, self.n_agents, self.action_dim))
        self.share_obs = self.obs.reshape(self.args.n_rollout_threads, -1)
        self.share_obs = np.expand_dims(self.share_obs, 1).repeat(self.n_agents, axis=1)
        # self.share_obs = np.delete(self.share_obs, [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 28, 29],
        #                           axis=-1)
        self.share_obs = np.delete(self.share_obs, [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23], axis=-1)
        self.vxy = np.zeros((self.args.n_rollout_threads, self.n_agents, self.vxy_dim))

    def seed(self, args):
        pass

    def reset(self):
        self.steps = 0
        self.obs = np.zeros((self.args.n_rollout_threads, self.n_agents, self.obs_dim))

        for env in range(self.args.n_rollout_threads):
            self.obs[env][0][0] = 7  # 0
            self.obs[env][0][1] = 0  # 10
            self.obs[env][1][0] = 9  # 1.34
            self.obs[env][1][1] = 0  # 5
            self.obs[env][2][0] = 11  # 5
            self.obs[env][2][1] = 0  # 1.34
            self.obs[env][3][0] = 13  # 10
            self.obs[env][3][1] = 0  # 0
            # self.obs[env][0][0] = 0
            # self.obs[env][0][1] = 30
            # self.obs[env][1][0] = 0
            # self.obs[env][1][1] = 15
            # self.obs[env][2][0] = 15
            # self.obs[env][2][1] = 0
            # self.obs[env][3][0] = 30
            # self.obs[env][3][1] = 0
            for agent_id in range(self.n_agents):
                self.obs[env][agent_id][2] = 10
                self.obs[env][agent_id][3] = 10

        self.share_obs = self.obs.reshape(self.args.n_rollout_threads, -1)
        self.share_obs = np.expand_dims(self.share_obs, 1).repeat(self.n_agents, axis=1)
        # self.share_obs = np.delete(self.share_obs, [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 28, 29],
        #                          axis=-1)
        self.share_obs = np.delete(self.share_obs, [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23], axis=-1)

        return self.obs, self.share_obs

    def step(self, actions):
        """make sure that actions of n_rollout_threads envs all be done"""
        self.steps += 1
        self.dis_time += 1
        done = np.zeros((self.args.n_rollout_threads, self.n_agents), dtype=bool)
        rewards = np.zeros((self.args.n_rollout_threads, 1))
        infos = {}

        # vxy = np.clip(actions, -10, 10)
        vxy = 7 * actions
        # axy = 80 * actions  # 圆轨迹80，直线47
        # vxy = np.zeros((self.args.n_rollout_threads, self.n_agents, self.action_dim))
        for env in range(self.args.n_rollout_threads):
            rewards_sum = 0

            for agent_id in range(self.n_agents):
                """obs...[0,1] with disturbance  [case 1]"""
                self.obs[env][agent_id][0] = self.obs[env][agent_id][0] + \
                                             (vxy[env][agent_id][0] + 0.3 * np.sin(
                                                 self.dis_time)) * self.time_interval + \
                                             0.1 * np.cos(self.dis_time)
                self.obs[env][agent_id][1] = self.obs[env][agent_id][1] + \
                                             (vxy[env][agent_id][1] + 0.3 * np.sin(
                                                 self.dis_time)) * self.time_interval + \
                                             0.1 * np.cos(self.dis_time)

                """obs...[0,1] with disturbance  [case 2]"""
                # dis1 = np.random.randint(-4, 5)/20
                # dis2 = np.random.randint(-5, 10)/20
                # self.obs[env][agent_id][0] = self.obs[env][agent_id][0] + \
                #                              (vxy[env][agent_id][0] + dis1) * self.time_interval + dis2
                # self.obs[env][agent_id][1] = self.obs[env][agent_id][1] + \
                #                              (vxy[env][agent_id][1] + dis1) * self.time_interval + dis2

                """obs...[0,1] without disturbance"""
                # self.obs[env][agent_id][0] = self.obs[env][agent_id][0] + vxy[env][agent_id][0] * self.time_interval
                # self.obs[env][agent_id][1] = self.obs[env][agent_id][1] + vxy[env][agent_id][1] * self.time_interval

                """obs...[0,1] -- use a not v -- without/with disturbance"""
                # d1, d2, d3, d4 = 0, 0, 0, 0
                # # d1, d2, d3, d4 = np.random.randint(-4, 5)/10, np.random.randint(-5, 10)/10, \
                # #                  np.random.randint(-5, 8) / 5, np.random.randint(-7, 7) / 5
                # vxy[env][agent_id][0] = vxy[env][agent_id][0] + axy[env][agent_id][0] * self.time_interval + d1
                # vxy[env][agent_id][1] = vxy[env][agent_id][1] + axy[env][agent_id][1] * self.time_interval + d2
                # self.obs[env][agent_id][0] = self.obs[env][agent_id][0] + vxy[env][agent_id][0] * self.time_interval + d3
                # self.obs[env][agent_id][1] = self.obs[env][agent_id][1] + vxy[env][agent_id][1] * self.time_interval + d4

                """# obs...[2,3]"""
                # if self.obs[env][agent_id][2] >= -15 and self.obs[env][agent_id][3] >= -15:
                #     self.obs[env][agent_id][2] = self.obs[env][agent_id][2] - a
                # elif (self.obs[env][agent_id][2] < -15) and self.obs[env][agent_id][3] > -15:
                #     self.obs[env][agent_id][3] = self.obs[env][agent_id][3] - a
                # elif (self.obs[env][agent_id][2] < -15) and self.obs[env][agent_id][3] < -15:
                #     self.obs[env][agent_id][2] = self.obs[env][agent_id][2] + a
                # else:
                #     self.obs[env][agent_id][3] = self.obs[env][agent_id][3] + a
                # if self.obs[env][agent_id][2] <= 40:
                #     self.obs[env][agent_id][2] = self.obs[env][agent_id][2] + e_e
                #     self.obs[env][agent_id][3] = self.obs[env][agent_id][3] + e_e
                # else:
                #     self.obs[env][agent_id][2] = self.obs[env][agent_id][2] + e_a
                #     self.obs[env][agent_id][3] = self.obs[env][agent_id][3] + e_b
                """evader 直线轨迹"""
                # self.obs[env][agent_id][2] = self.obs[env][agent_id][2] + 0.15
                # self.obs[env][agent_id][3] = self.obs[env][agent_id][3] + 0.15
                """evader 圆形轨迹"""
                e_path_1 = 1
                if self.obs[env][agent_id][3] <= 30 and (self.obs[env][agent_id][2] <= 10):
                    e_path_1 = 0

                if e_path_1 == 1:
                    theta0 = np.arccos(self.obs[env][agent_id][2] / 20 - 1.5)
                    theta = theta0 if self.obs[env][agent_id][3] > 30 else -theta0
                    self.obs[env][agent_id][2] = 30 + 20 * np.cos(theta - 0.6 / 20)
                    self.obs[env][agent_id][3] = 30 + 20 * np.sin(theta - 0.6 / 20)
                else:
                    self.obs[env][agent_id][3] = self.obs[env][agent_id][3] + 0.6

                """obs...[4,5]"""
                near_distance, near_id = 1e5, 0
                for n in range(self.n_agents):
                    if n != agent_id:
                        n_x = self.obs[env][agent_id][0] - self.obs[env][n][0]
                        n_y = self.obs[env][agent_id][1] - self.obs[env][n][1]
                        distance = ((n_x ** 2) + (n_y ** 2)) ** 0.5
                        if distance < near_distance:
                            near_distance = distance
                            near_id = n
                self.obs[env][agent_id][4] = self.obs[env][near_id][0]
                self.obs[env][agent_id][5] = self.obs[env][near_id][1]

                # """reward1 compute"""
                # epsilon, c, zero_len = 2, 4, 0.3
                # de0 = 5
                # die = ((self.obs[env][agent_id][0] - self.obs[env][agent_id][2]) ** 2 + (
                #         self.obs[env][agent_id][1] - self.obs[env][agent_id][3]) ** 2) ** 0.5
                #
                # if de0 - zero_len <= die <= de0 + zero_len:
                #     phi_p = 0
                # elif die > de0 + zero_len or epsilon <= die <= de0 - zero_len:
                #     phi_p = 4 * (die - de0) ** 2 / die  # +5
                # elif die <= 1:
                #     # phi_p = 300
                #     phi_p = 40
                # else:
                #     phi_p = -2 * die + 22
                #     # phi_p = -20 * die + 60
                #
                # """reward2 compute"""
                # if 2 <= near_distance <= c:
                #     phi_o = 2 * (near_distance - c) ** 2 / near_distance  # 2 *
                # elif 1 < near_distance < 2:
                #     phi_o = -20 * near_distance + 40.5
                #     # phi_o = -60 * near_distance + 135
                # elif near_distance <= 1 and near_distance != 0:
                #     # phi_o = 200
                #     phi_o = 40
                # else:
                #     phi_o = 0

                # reward1 compute
                # epsilon, c, zero_len = 2, 4, 0.1
                # de0 = 4
                # die = ((self.obs[env][agent_id][0] - self.obs[env][agent_id][2]) ** 2 + (
                #         self.obs[env][agent_id][1] - self.obs[env][agent_id][3]) ** 2) ** 0.5
                #
                # if 3 <= die <= de0:
                #     # phi_p = -1 / 1.1 * near_distance + 4 / 1.1
                #     phi_p = -1
                # elif die < 2:
                #     #phi_p = (die - de0) ** 2 / (die + 0.2)
                #     phi_p = 1
                # else:
                #     # phi_p = (die - de0) ** 2 / (die - 2)
                #     # phi_p = (die - de0) ** 3 / die
                #     phi_p = 1
                # # reward2 liner->unliner
                # if 2 <= near_distance <= c:
                #     # phi_o = -1 / 1.1 * near_distance + 4 / 1.1
                #     phi_o = 0.5
                # elif near_distance < 2 and near_distance != 0:
                #     # phi_o = (near_distance - c) ** 2 / (near_distance + 0.2)
                #     phi_o = 1
                # else:
                #     phi_o = 0

                # reward1
                # if (vxy[env][agent_id][0]) < (vxy[env][agent_id][1]):
                #     reward = 1
                # else:
                #     reward = -1
                '''论文中的reward-最原始的reward  OAcof=1  episolon=2  c=4 de0=5
                    但在直线跑图的时候，我把OAcof = 1.2'''
                """reward1 compute"""
                OAcof = 1
                epsilon, c, zero_len = 2, 4, 0.3
                de0 = 5
                die = ((self.obs[env][agent_id][0] - self.obs[env][agent_id][2]) ** 2 + (
                        self.obs[env][agent_id][1] - self.obs[env][agent_id][3]) ** 2) ** 0.5
                if die > epsilon:
                    phi_p = (die - de0) ** 2 / die
                else:
                    phi_p = OAcof * (epsilon - de0) ** 2 / epsilon

                """reward2 compute"""
                if 0.1 <= near_distance <= c:
                    phi_o = OAcof * (near_distance - c) ** 2 / near_distance  # 2 *
                elif near_distance < 0.1:
                    phi_o = OAcof * (0.1 - c) ** 2 / 0.1
                else:
                    phi_o = 0

                """reward 3"""
                reward_u = (vxy[env][agent_id][0] ** 2 + vxy[env][agent_id][1] ** 2) * self.time_interval

                reward = 0.1 * (-phi_p - 0.5 * phi_o) \
                    # - 0.1 * reward_u

                """reward sum"""
                rewards_sum += reward
            rewards[env][0] = rewards_sum
        rewards = np.expand_dims(rewards, 1).repeat(self.n_agents, axis=1)
        self.share_obs = self.obs.reshape(self.args.n_rollout_threads, -1)
        self.share_obs = np.expand_dims(self.share_obs, 1).repeat(self.n_agents, axis=1)
        # self.share_obs = np.delete(self.share_obs, [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 28, 29],
        #                            axis=-1)
        self.share_obs = np.delete(self.share_obs, [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23], axis=-1)
        return self.obs, self.share_obs, rewards, done, infos

    def render(self):
        pass

    def close(self):
        print("env close")
