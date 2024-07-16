import time
import numpy as np
import os
from functools import reduce
import torch
from runners.separated.base_runner import Runner
from matplotlib import pyplot as plt
from envs.musv.ADP0 import ADP, loss_cal_for_marl, plot4_for_marl
from plots.plot_MpeTraj import plot_traj as save_img_traj


def save_txt_traj(x, y, exy, save_path):
    k0, k1 = 0, 0
    for i in x:
        k0 += 1
        np.savetxt(save_path + f'x_{k0}.txt', i)
    for i in y:
        k1 += 1
        np.savetxt(save_path + f'y_{k1}.txt', i)

    np.savetxt(save_path + f'x_e.txt', exy[0])
    np.savetxt(save_path + f'y_e.txt', exy[1])


def _t2n(x):
    return x.detach().cpu().numpy()


class MusvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(MusvRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        store_std = []
        store_aver_episode_rewards = []

        # for eval

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
            done_episodes_rewards = []
            obs, share_obs = self.envs.reset()

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                # actions = actions + np.random.randint(-1, 1)/100
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            aver_episode_rewards = np.mean(train_episode_rewards)
            std = np.std(train_episode_rewards)
            # store_aver_episode_rewards.append(aver_episode_rewards)
            store_aver_episode_rewards.append(train_episode_rewards[0])
            store_std.append(std)
            if episode % self.log_interval == 0 and episode != 0:
                np.savetxt('/home/lyy/Desktop/store_aver_episode_rewards.txt', store_aver_episode_rewards)
                np.savetxt("/home/lyy/Desktop/store_std.txt", store_std)
            # save model
            if (episode + 1 % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario: {} Algo: {} Exp: {} updates: {}/{} episodes, total num timesteps: {}/{}, FPS {}.\n"
                      .format('Musv-MPE',
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                                             total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            if episode > 500 and episode % 20 == 0:
                # if True:
                """plot here"""
                x, y = [], []
                fig, ax = plt.subplots()
                obs, share_obs = self.envs.reset()
                x.append(obs[0][:, 0].tolist())
                y.append(obs[0][:, 1].tolist())
                x[0].append(obs[0][:, 2][0].tolist())
                y[0].append(obs[0][:, 3][0].tolist())
                for step in range(self.episode_length):
                    _, actions, _, _, _ = self.collect(step)
                    obs, share_obs, rewards, _, _ = self.envs.step(actions)
                    x.append(obs[0][:, 0].tolist())
                    y.append(obs[0][:, 1].tolist())
                    x[step + 1].append(obs[0][:, 2][0].tolist())
                    y[step + 1].append(obs[0][:, 3][0].tolist())
                for t in range(len(x)):
                    # 在每个时间步骤画出所有智能体的位置
                    for i in range(self.num_agents + 1):
                        ax.scatter(x[t][i], y[t][i], c='b' if i < self.num_agents else 'r')
                    # 设置坐标轴范围
                    ax.set_xlim([-5, 70])
                    ax.set_ylim([-5, 70])
                    # plot
                    plt.show(block=False)
                    if t == 0:
                        plt.pause(5)
                    plt.pause(0.05)
                    ax.clear()
                plt.close()

            if (episode == episodes - 2) or (episode % 10 == 0):
                x, y = [], []
                obs, share_obs = self.envs.reset()
                x.append(obs[0][:, 0].tolist())
                y.append(obs[0][:, 1].tolist())
                x[0].append(obs[0][:, 2][0].tolist())
                y[0].append(obs[0][:, 3][0].tolist())
                for step in range(self.episode_length):
                    _, actions, _, _, _ = self.collect(step)
                    obs, share_obs, rewards, _, _ = self.envs.step(actions)
                    x.append(obs[0][:, 0].tolist())
                    y.append(obs[0][:, 1].tolist())
                    x[step + 1].append(obs[0][:, 2][0].tolist())
                    y[step + 1].append(obs[0][:, 3][0].tolist())

                x_1 = [row[0] for row in x]
                x_2 = [row[1] for row in x]
                x_3 = [row[2] for row in x]
                x_4 = [row[3] for row in x]
                # x_5 = [row[4] for row in x]
                # x_e = [row[5] for row in x]
                x_e = [row[4] for row in x]
                y_1 = [row[0] for row in y]
                y_2 = [row[1] for row in y]
                y_3 = [row[2] for row in y]
                y_4 = [row[3] for row in y]
                # y_5 = [row[4] for row in y]
                # y_e = [row[5] for row in y]
                y_e = [row[4] for row in y]
                save_root_path = '/home/lyy/Desktop/happo_imgs/MpeTraj_plot/'
                save_txt_traj([x_1, x_2, x_3, x_4], [y_1, y_2, y_3, y_4], [x_e, y_e], save_root_path + '0/')

                '''为了防止中间数据丢失，将每一次的数据都存储并绘制'''
                sub_folders = [int(name) for name in os.listdir(save_root_path) \
                               if os.path.isdir(os.path.join(save_root_path, name))]
                sub_folders.sort()
                max_folder_num = sub_folders[-1]
                missing_folder_num = None
                if len(sub_folders) != 1:
                    for i in range(len(sub_folders)-1):
                        if sub_folders[i+1]-sub_folders[i] > 1:
                            missing_folder_num = sub_folders[i]+1
                            break
                expected_folder_num = missing_folder_num if (missing_folder_num is not None) else (max_folder_num+1)
                new_sub_folder = save_root_path + str(expected_folder_num) + '/'
                os.mkdir(new_sub_folder)
                save_txt_traj([x_1, x_2, x_3, x_4], [y_1, y_2, y_3, y_4], [x_e, y_e], new_sub_folder)
                save_img_traj(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, new_sub_folder)


                # plot
                fig, ax = plt.subplots()
                ax.plot(x_e, y_e, color='red', label='Evader')
                ax.plot(x_1, y_1, color='blue', linestyle='--', label='Pursuer1')
                ax.plot(x_2, y_2, color='green', linestyle='--', label='Pursuer2')
                ax.plot(x_3, y_3, color='orange', linestyle='--', label='Pursuer3')
                ax.plot(x_4, y_4, color='purple', linestyle='--', label='Pursuer4')
                # ax.plot(x_5, y_5, color='grey',linestyle='--',label='Pursuer5')

                ax.set_xlim([-5, 70])
                ax.set_ylim([-5, 70])
                ax.legend()
                plt.savefig('/home/lyy/Desktop/HAPPO-HATRPO/mpe_a1.pdf')

            if (episode == episodes - 1) and episode > 50:
                # if episode == 0 :
                x, y = [], []
                obs, share_obs = self.envs.reset()
                x.append(obs[0][:, 0].tolist())
                y.append(obs[0][:, 1].tolist())
                x[0].append(obs[0][:, 2][0].tolist())
                y[0].append(obs[0][:, 3][0].tolist())
                for step in range(self.episode_length):
                    _, actions, _, _, _ = self.collect(step)
                    obs, share_obs, rewards, _, _ = self.envs.step(actions)
                    x.append(obs[0][:, 0].tolist())
                    y.append(obs[0][:, 1].tolist())
                    x[step + 1].append(obs[0][:, 2][0].tolist())
                    y[step + 1].append(obs[0][:, 3][0].tolist())

                x_1 = [row[0] for row in x]
                x_2 = [row[1] for row in x]
                x_3 = [row[2] for row in x]
                x_4 = [row[3] for row in x]
                # x_5 = [row[4] for row in x]
                # x_e = [row[5] for row in x]
                x_e = [row[4] for row in x]
                y_1 = [row[0] for row in y]
                y_2 = [row[1] for row in y]
                y_3 = [row[2] for row in y]
                y_4 = [row[3] for row in y]
                # y_5 = [row[4] for row in y]
                # y_e = [row[5] for row in y]
                y_e = [row[4] for row in y]

                # plot
                fig, ax = plt.subplots()
                ax.plot(x_e, y_e, color='red', label='Evader')
                ax.plot(x_1, y_1, color='blue', linestyle='--', label='Pursuer1')
                ax.plot(x_2, y_2, color='green', linestyle='--', label='Pursuer2')
                ax.plot(x_3, y_3, color='orange', linestyle='--', label='Pursuer3')
                ax.plot(x_4, y_4, color='purple', linestyle='--', label='Pursuer4')
                # ax.plot(x_5, y_5, color='grey',linestyle='--',label='Pursuer5')

                ax.set_xlim([-5, 70])
                ax.set_ylim([-5, 70])
                ax.legend()
                plt.savefig('/home/lyy/Desktop/HAPPO-HATRPO/mpe_a1.pdf')

            '''plot with observer'''
            if episode > 1000 and episode % 20 == 0:
                # if episode == 0:
                # x1, x2, x3, x4 = [], [], [], []
                adp = ADP(lr=5e-6, model_path='/home/lyy/Desktop/HAPPO-HATRPO/envs/musv/ADP_model.pt')
                x_raw_lst = []
                obs, share_obs = self.envs.reset()
                x_raw_lst.append([obs[0][0][0].tolist(), 0, obs[0][0][1].tolist(), 0])
                x_now_raw = torch.tensor(x_raw_lst[0], dtype=torch.float32).reshape(4, 1)
                x_now_raw[1][0] = 15
                x_now_raw[3][0] = 4
                x_now_raw[0][0] = 1
                # x_now_raw = torch.zeros((4, 1), dtype=torch.float32)
                x_now_obs = torch.zeros((4, 1), dtype=torch.float32)
                x_raw_lst = []
                x_obs_lst = []
                u_lst = []

                for step in range(300):
                    x_raw_lst.append(x_now_raw.tolist())
                    x_obs_lst.append(x_now_obs.tolist())
                    _, actions, _, _, _ = self.collect(step)
                    u = torch.tensor(actions[0][0], dtype=torch.float32).unsqueeze(1)
                    # obs, share_obs, rewards, _, _ = self.envs.step(actions)
                    u_lst.append(u.tolist())

                    obs = torch.cat([x_now_raw, x_now_obs])
                    dvalue_dx = adp.model(obs.T)
                    _, x_now_raw, x_now_obs = loss_cal_for_marl(obs, dvalue_dx, u)
                    # x_now_raw = raw_sys_for_marl(x_now_raw, u)
                plot4_for_marl(x_raw_lst, x_obs_lst)
                np.save('/home/lyy/Desktop/happo_imgs/observer_obs/x_raw_lst.npy', np.array(x_raw_lst))
                np.save("/home/lyy/Desktop/happo_imgs/observer_obs/x_obs_lst.npy", np.array(x_obs_lst))
                np.save("/home/lyy/Desktop/happo_imgs/observer_obs/u_lst.txt", np.array(u_lst))

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_max_episode_rewards': [np.max(eval_episode_rewards)]}
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                break
