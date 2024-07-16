"""
base on MpeRvoLinearTrack.py
  -- add velocity prediction NN
"""

import os
import rvo2
import matplotlib.pyplot as plt
import random, math
import numpy as np
import torch
from tqdm import tqdm

from plot.plot import plot_traj, plot_traj_static, plot_state, plot_est_traj
from AC_track import Tracking
from env.LinearBaseEnv import USV


def rotate_points(now_pos, center, rotate_speed_rad_, time):
    # 计算每个点相对于圆心的极角
    angles = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in now_pos]
    # 计算每个点在经过time时间后的新极角
    new_angles = [angle + rotate_speed_rad_ * time for angle in angles]
    # 计算每个点在经过time时间后的新位置
    new_pos = [(center[0] + 15 * math.cos(angle), center[1] + 15 * math.sin(angle)) for angle in new_angles]
    return new_pos


def cal_pref_velocity(usv_now_pos_, target_pos_, max_speed_):
    dif = target_pos_ - usv_now_pos_
    distance = np.linalg.norm(dif)

    angle_rad = math.atan2(dif[1], dif[0])
    if distance > 0.1:
        vx = max_speed_ * math.cos(angle_rad)
        vy = max_speed_ * math.sin(angle_rad)
    else:
        vx = 0
        vy = 0
    return (vx, vy)


def all_arrive_target_pos(now_pos, target_pos, num_agents):
    for i in range(num_agents):
        x1, y1 = now_pos[i]
        x2, y2 = target_pos[i]
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if distance > 0.5:
            return False
    return True


def cal_target_pos(evader_, e_angle_, e_radius_):
    # pos = [(evader[0] + e_radius * math.cos(e_angle_4[0]), evader[1] + e_radius * math.sin(e_angle_4[0])),
    #               (evader[0] + e_radius * math.cos(e_angle_4[1]), evader[1] + e_radius * math.sin(e_angle_4[0])),
    #               (evader[0] + e_radius * math.cos(e_angle_4[2]), evader[1] + e_radius * math.sin(e_angle_4[0])),
    #               (evader[0] + e_radius * math.cos(e_angle_4[3]), evader[1] + e_radius * math.sin(e_angle_4[0]))]
    pos = [(evader_[0] + e_radius_ * math.cos(e_angle_[i]), evader_[1] + e_radius_ * math.sin(e_angle_4[i]))
           for i in range(len(e_angle_4))]
    pos = [np.array(i, dtype=float) for i in pos]
    return pos


def reset_sim_pos_speed():
    sim.setAgentPosition(usv0, (5, 5))
    sim.setAgentPosition(usv1, (55, 5))
    sim.setAgentPosition(usv2, (55, 55))
    sim.setAgentPosition(usv3, (5, 55))
    sim.setAgentVelocity(usv0, (0, 0))
    sim.setAgentVelocity(usv1, (0, 0))
    sim.setAgentVelocity(usv2, (0, 0))
    sim.setAgentVelocity(usv3, (0, 0))


"""param"""
max_speed = 6
num_agents = 4
# num_obst = 1
time_step = 1 / 10.
rotate_speed_rad = math.pi / 10

# radius=1 or 1.5 is look good
sim = rvo2.PyRVOSimulator(timeStep=time_step, neighborDist=20, maxNeighbors=4, timeHorizon=1.8, timeHorizonObst=0.8,
                          radius=1.3, maxSpeed=max_speed)

# INIT USV
usv0 = sim.addAgent((5, 5))
usv1 = sim.addAgent((55, 5))
usv2 = sim.addAgent((55, 55))
usv3 = sim.addAgent((5, 55))

# INIT evader and target pos
ex, ey = 30, 30
evader = np.array([ex, ey], dtype=float)
e_angle_4 = [3 * math.pi / 2, 0, math.pi / 2, math.pi]
e_radius = 15
target_pos = cal_target_pos(evader, e_angle_4, e_radius)

# INIT OBST
obst_pos = [(ex + 15 * math.cos(math.pi / 4), ey + 15 * math.sin(math.pi / 4)),
            (ex + 15 * math.cos(3 * math.pi / 4), ey + 15 * math.sin(math.pi / 4)),
            (10, 10),
            (15, 8),
            (18, 12),
            (10, 25),
            (20, 35),
            (53, 8),
            (50, 50),
            (49, 20),
            (55, 15),
            (10, 40),
            (20, 20.5),
            ]
num_obst = len(obst_pos)
for i in obst_pos:
    sim.addAgent(i)

usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]

# INIT Path_track NN and USV env
# if_train = True
if_train = False
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.dirname(os.getcwd()) + "/save/TrackModel/track_NN_linear.pt"
model_base_path = os.path.dirname(os.getcwd()) + "/save/TrackModel/track_NN_linear_base0.pt"
model_load_path = None if if_train else model_base_path
track_NN = Tracking(lr=1e-5, input_dim=4, output_dim=2, model_path=model_load_path)
# env0 = USV((10.0, 10.0, 0.0), (3.0, 3.0, 3.0), time_interval=time_step)
env0 = USV((5.0, 5.0), time_interval=time_step)
env1 = USV((55.0, 5.0), time_interval=time_step)
env2 = USV((55.0, 55.0), time_interval=time_step)
env3 = USV((5.0, 55.0), time_interval=time_step)

# INIT Speed_Estimation NN "first based on control u and now state=vel+pos for test"
if_train_est = False
# if_train_est = True
est_model_save_path = os.path.dirname(os.getcwd()) + "/save/VelEstModel/Est_NN_linear.pt"
est_model = None if if_train_est else est_model_save_path
est_NN = Tracking(lr=1e-5, input_dim=6, output_dim=2, model_path=est_model)

""" INIT list
x_lst, y_lst = time_steps * np.array([usv0, ..., usvn, evader])
x_lst_des, y_lst_des = time_steps * np.array([target_pos_1, ..., target_pos_n])
"""
px_lst_usv, py_lst_usv, px_lst_des, py_lst_des = [], [], [], []
vx_lst_usv, vy_lst_usv, vx_lst_des, vy_lst_des = [], [], [], []
x_swap, y_swap = [], []
for i in range(num_agents):
    x_swap.append(sim.getAgentPosition(i)[0])
    y_swap.append(sim.getAgentPosition(i)[1])
    if i == num_agents - 1:
        x_swap.append(evader[0])
        y_swap.append(evader[1])
px_lst_usv.append(x_swap)
py_lst_usv.append(y_swap)
px_lst_des = [[i[0] for i in target_pos]]
py_lst_des = [[i[1] for i in target_pos]]

obst_lst = []
for i in range(num_obst):
    obst_lst.append(sim.getAgentPosition(num_agents + i))
rotate_flag = False

"""here start episode"""
Episodes = 1000
steps = 400
reward_sum, episode_record, bf_reward = 0, [], 1e+5
tq_bar = tqdm(range(Episodes))
# if_train = False

# torch.autograd.set_detect_anomaly(True)
if if_train or if_train_est:
    for episode in tq_bar:
        env0.reset()
        reset_sim_pos_speed()
        tq_bar.set_description(f'Episode [ {episode + 1} / {Episodes} ]')

        for step in range(steps):

            for i in range(num_obst):
                sim.setAgentPosition(num_agents + i, obst_pos[i])

            for i in range(num_agents):
                PrefVelocity = cal_pref_velocity(np.array(sim.getAgentPosition(i), dtype=float), target_pos[i],
                                                 max_speed)
                sim.setAgentPrefVelocity(i, PrefVelocity)

            if not rotate_flag:
                usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
                rotate_flag = all_arrive_target_pos(usv_now_pos, target_pos, num_agents)
            else:
                target_pos = rotate_points(target_pos, evader, rotate_speed_rad, time_step)

            sim.doStep()
            ideal_speed = sim.getAgentVelocity(usv0)
            true_speed = env0.vxy
            state = torch.cat((torch.tensor(ideal_speed), true_speed.reshape(2, )))
            # tau = track_NN.cal_tau(state)
            tau = track_NN.cal_tau(state)

            """Est_NN set"""
            vxy_tmp = env0.vxy.detach().T[0]
            pxy_tmp = env0.pxy.detach().T[0]

            env0.step(tau)
            true_speed_next = tuple([i.item() for i in env0.vxy.detach()])
            true_pos_next = tuple([i.item() for i in env0.pxy.detach()])
            sim.setAgentPosition(usv0, true_pos_next)
            sim.setAgentVelocity(usv0, true_speed_next)
            if if_train:
                loss = ((env0.vxy[0] - state[0]) ** 2 + (env0.vxy[1] - state[1]) ** 2) ** 0.5
                track_NN.train(loss)
                reward_sum = reward_sum + round(loss[0].item(), 5)

            """Est_NN train"""
            # # use now tau / now vxy / now pxy ---done
            # est_input = torch.cat((tau.detach(), env0.vxy.detach().T[0], env0.pxy.detach().T[0]))
            # est_input = torch.tensor(est_input.T, dtype=torch.float32, requires_grad=True)
            # est_output = est_NN.model(est_input)
            # if if_train_est:
            #     loss_tmp = (est_output - env0.vxy.detach().T[0])
            #     loss_Est = loss_tmp.T @ loss_tmp
            #     est_NN.train(loss_Est)
            #     reward_sum = reward_sum + round(loss_Est.item(), 5)

            # use now vxy / now pxy / next pxy ---done
            est_input = torch.cat((vxy_tmp, pxy_tmp, env0.pxy.detach().T[0]))
            est_input = torch.tensor(est_input.T, dtype=torch.float32, requires_grad=True)
            est_output = est_NN.model(est_input)
            if if_train_est:
                loss_tmp = (est_output - env0.vxy.detach().T[0])
                loss_Est = loss_tmp.T @ loss_tmp
                est_NN.train(loss_Est)
                reward_sum = reward_sum + round(loss_Est.item(), 5)

            env0.grad_free()
            # reward_sum = reward_sum + round(loss[0].item(), 5)
            # print(loss)
            if step == steps - 1:
                if if_train_est:
                    last_step_loss = round(loss_Est.item(), 5)
                if if_train:
                    last_step_loss = round(loss[0].item(), 5)

        episode_record.append(reward_sum)
        score = 0
        ep_mean_reward = np.mean(episode_record[-10:])
        # Init_reward = ep_mean_reward if episode == 10 else 0
        # bf_reward = episode_reward if episode_reward <= bf_reward else bf_reward
        if ep_mean_reward <= bf_reward:
            bf_reward = ep_mean_reward
            if if_train and episode >= 10:
                track_NN.save_model(model_save_path)
            if if_train_est and episode >= 10:
                est_NN.save_model(est_model_save_path)
        tq_bar.set_postfix({'lastMeanRewards': f'{ep_mean_reward:.2f}', 'BEST': f'{bf_reward:.2f}',
                            'last_step_loss': f'{last_step_loss:.2f}', 'steps': f'{steps}'})
        reward_sum = 0
else:
    env0.reset()
    env1.reset()
    env2.reset()
    env3.reset()
    reset_sim_pos_speed()
    vxy_est_lst, vxy_env_lst = [], []
    for step in range(steps):
        px_swap, py_swap, px_swap_des, py_swap_des = [], [], [], []
        vx_swap, vy_swap, vx_swap_des, vy_swap_des = [], [], [], []

        for i in range(num_obst):
            sim.setAgentPosition(num_agents + i, obst_pos[i])

        for i in range(num_agents):
            PrefVelocity = cal_pref_velocity(np.array(sim.getAgentPosition(i), dtype=float), target_pos[i],
                                             max_speed)
            sim.setAgentPrefVelocity(i, PrefVelocity)

        if not rotate_flag:
            usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
            rotate_flag = all_arrive_target_pos(usv_now_pos, target_pos, num_agents)
        else:
            target_pos = rotate_points(target_pos, evader, rotate_speed_rad, time_step)

        sim.doStep()
        """ here """
        for j in range(num_agents):
            env = globals()[f'env{j}']
            ideal_speed = sim.getAgentVelocity(j)
            true_speed = env.vxy
            state = torch.cat((torch.tensor(ideal_speed), true_speed.reshape(2, )))
            # tau = track_NN.cal_tau(state)
            tau = track_NN.cal_tau(state)

            """Est_NN set"""
            if j == 0:
                vxy_tmp = env.vxy.detach().T[0]
                pxy_tmp = env.pxy.detach().T[0]
            env.step(tau)
            if j == 0:
                est_input = torch.cat((vxy_tmp, pxy_tmp, env0.pxy.detach().T[0]))
                est_input = torch.tensor(est_input.T, dtype=torch.float32, requires_grad=True)
                vxy_est = est_NN.model(est_input)
                vxy_est_lst.append(vxy_est.detach().tolist())
                vxy_env_lst.append(env.vxy.detach().T[0].tolist())
                # loss_tmp = (est_output - env0.vxy.detach().T[0])
                # loss_Est = loss_tmp.T @ loss_tmp
                # est_NN.train(loss_Est)
                # reward_sum = reward_sum + round(loss_Est.item(), 5)

        for i in range(num_agents):
            env = globals()[f'env{i}']
            px_swap.append(env.pxy[0].detach().item())
            py_swap.append(env.pxy[1].detach().item())
            px_swap_des.append(target_pos[i][0])
            py_swap_des.append(target_pos[i][1])

            vx_swap.append(env.vxy[0].detach().item())
            vy_swap.append(env.vxy[1].detach().item())
            vx_swap_des.append(sim.getAgentVelocity(i)[0])
            vy_swap_des.append(sim.getAgentVelocity(i)[1])

            # vx_swap
            if i == num_agents - 1:
                px_swap.append(evader[0])
                py_swap.append(evader[1])

        for j in range(num_agents):
            env = globals()[f'env{j}']
            true_speed_next = tuple([i.item() for i in env.vxy.detach()])
            true_pos_next = tuple([i.item() for i in env.pxy.detach()])
            sim.setAgentPosition(j, true_pos_next)
            sim.setAgentVelocity(j, true_speed_next)

        px_lst_usv.append(px_swap)
        py_lst_usv.append(py_swap)
        px_lst_des.append(px_swap_des)
        py_lst_des.append(py_swap_des)

        vx_lst_usv.append(vx_swap)
        vy_lst_usv.append(vy_swap)
        vx_lst_des.append(vx_swap_des)
        vy_lst_des.append(vy_swap_des)
    # plot_traj(px_lst_usv, py_lst_usv, px_lst_des, py_lst_des, obst_lst, num_agents=num_agents, circle_radius=e_radius)
    # plot_traj_static(px_lst_usv, py_lst_usv, obst_lst, num_agents=num_agents, circle_radius=e_radius, xy_lim=[0, 70],
    #                  background_img=r'/home/liuyangyang/MARL/MpeRvo/plot/2.jpg')
    plot_est_traj(vxy_est_lst, vxy_env_lst)
    # plot_state(px_lst_usv, px_lst_des, py_lst_usv, py_lst_des,
    #            vx_lst_usv, vx_lst_des, vy_lst_usv, vy_lst_des,
    #            num_agents)
