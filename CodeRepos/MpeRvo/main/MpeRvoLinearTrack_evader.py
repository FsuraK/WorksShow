"""
base on MpeRvoLinearTrack.py
  -- add evader moving
"""

import os
import rvo2
import matplotlib.pyplot as plt
import random, math
import numpy as np
import torch
from tqdm import tqdm

from plot.plot import plot_traj, plot_traj_static, plot_state
from AC_track import Tracking
from env.LinearBaseEnv import USV


def rotate_points(center, rotate_speed_rad_, time, rotate_radiu=10, now_pos=None, angles=None):
    if now_pos is not None:
        # 计算每个点相对于圆心的极角
        angles = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in now_pos]
        # 计算每个点在经过time时间后的新极角
    angles = [angle + rotate_speed_rad_ * time for angle in angles]
    # 计算每个点在经过time时间后的新位置
    new_pos = [(center[0] + rotate_radiu * math.cos(angle), center[1] + rotate_radiu * math.sin(angle)) for angle in angles]
    return new_pos, angles


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
        if distance > 2:
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
usv1 = sim.addAgent((20, 5))
usv2 = sim.addAgent((5, 35))
usv3 = sim.addAgent((5, 20))

# INIT evader and target pos
ex, ey = 20, 20
evader = np.array([ex, ey], dtype=float)
e_angle_4 = [3 * math.pi / 2, 0, math.pi / 2, math.pi]
e_radius = 12
target_pos = cal_target_pos(evader, e_angle_4, e_radius)

# INIT OBST
obst_pos = [(ex + 15 * math.cos(math.pi / 4), ey + 15 * math.sin(math.pi / 4)),
            (ex + 15 * math.cos(3 * math.pi / 4), ey + 15 * math.sin(math.pi / 4)),
            (10, 10),
            (25, 10),
            (18, 12),
            (10, 25),
            (20, 35),
            (53, 8),
            (50, 55),
            (49, 20),
            (55, 15),
            (10, 40),
            (45, 43),
            ]
num_obst = len(obst_pos)
for i in obst_pos:
    sim.addAgent(i)

usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]

# INIT NN and USV env
# if_train = True
if_train = False
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.dirname(os.getcwd()) + "/save/TrackModel/track_NN_linear_evader.pt"
model_base_path = os.path.dirname(os.getcwd()) + "/save/TrackModel/track_NN_linear_base0.pt"
# model_load_path = None if if_train else model_save_path
model_load_path = None if if_train else model_base_path
track_NN = Tracking(lr=1e-5, input_dim=4, output_dim=2, model_path=model_load_path)
# env0 = USV((10.0, 10.0, 0.0), (3.0, 3.0, 3.0), time_interval=time_step)
env0 = USV((5.0, 5.0), time_interval=time_step)
env1 = USV((20.0, 5.0), time_interval=time_step)
env2 = USV((5.0, 35.0), time_interval=time_step)
env3 = USV((5.0, 20.0), time_interval=time_step)

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
if if_train:
    for episode in tq_bar:
        env0.reset()
        env1.reset()
        env2.reset()
        env3.reset()
        reset_sim_pos_speed()
        angles = [math.atan2(point[1] - evader[1], point[0] - evader[0]) for point in target_pos]
        tq_bar.set_description(f'Episode [ {episode + 1} / {Episodes} ]')

        for step in range(steps):

            for i in range(num_obst):
                sim.setAgentPosition(num_agents + i, obst_pos[i])

            for i in range(num_agents):
                PrefVelocity = cal_pref_velocity(np.array(sim.getAgentPosition(i), dtype=float), target_pos[i],
                                                 max_speed)
                sim.setAgentPrefVelocity(i, PrefVelocity)

            rotate_flag = True
            if not rotate_flag:
                usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
                rotate_flag = all_arrive_target_pos(usv_now_pos, target_pos, num_agents)
            else:
                target_pos, angles = rotate_points(evader, rotate_speed_rad, time_step, e_radius, angles=angles)

            sim.doStep()
            ideal_speed = sim.getAgentVelocity(usv0)
            true_speed = env0.vxy
            state = torch.cat((torch.tensor(ideal_speed), true_speed.reshape(2, )))
            # tau = track_NN.cal_tau(state)
            tau = track_NN.cal_tau(state)
            # usv and evader move
            env0.step(tau)
            # evader action
            if rotate_flag:
                evader[0] += 0.13
                evader[1] = 0.02 * (evader[0]) ** 2 - 0.8 * evader[0] + 28

            true_speed_next = tuple([i.item() for i in env0.vxy.detach()])
            true_pos_next = tuple([i.item() for i in env0.pxy.detach()])
            sim.setAgentPosition(usv0, true_pos_next)
            sim.setAgentVelocity(usv0, true_speed_next)

            loss = ((env0.vxy[0] - state[0]) ** 2 + (env0.vxy[1] - state[1]) ** 2) ** 0.5
            track_NN.train(loss)
            env0.grad_free()
            reward_sum = reward_sum + round(loss[0].item(), 5)
            # print(loss)
            if step == steps - 1:
                last_step_loss = round(loss[0].item(), 5)

        episode_record.append(reward_sum)
        score = 0
        ep_mean_reward = np.mean(episode_record[-10:])
        # Init_reward = ep_mean_reward if episode == 10 else 0
        # bf_reward = episode_reward if episode_reward <= bf_reward else bf_reward
        if ep_mean_reward <= bf_reward:
            bf_reward = ep_mean_reward
            if episode >= 10:
                track_NN.save_model(model_save_path)
        tq_bar.set_postfix({'lastMeanRewards': f'{ep_mean_reward:.2f}', 'BEST': f'{bf_reward:.2f}',
                            'last_step_loss': f'{last_step_loss:.2f}', 'steps': f'{steps}'})
        reward_sum = 0
else:
    env0.reset()
    env1.reset()
    env2.reset()
    env3.reset()
    reset_sim_pos_speed()
    angles = [math.atan2(point[1] - evader[1], point[0] - evader[0]) for point in target_pos]
    track_performance_time, first_arrive_flag, arrive_steps = 0, False, 0
    for step in range(steps):
        px_swap, py_swap, px_swap_des, py_swap_des = [], [], [], []
        vx_swap, vy_swap, vx_swap_des, vy_swap_des = [], [], [], []

        for i in range(num_obst):
            sim.setAgentPosition(num_agents + i, obst_pos[i])

        for i in range(num_agents):
            PrefVelocity = cal_pref_velocity(np.array(sim.getAgentPosition(i), dtype=float), target_pos[i],
                                             max_speed)
            sim.setAgentPrefVelocity(i, PrefVelocity)

        rotate_flag = True
        if not rotate_flag:
            usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
            rotate_flag = all_arrive_target_pos(usv_now_pos, target_pos, num_agents)
        else:
            target_pos, angles = rotate_points(evader, rotate_speed_rad, time_step, e_radius, angles=angles)

        """kuai su xing"""
        usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
        if all_arrive_target_pos(usv_now_pos, target_pos, num_agents):
            # print(step)
            track_performance_time += 1
            first_arrive_flag = True
        if first_arrive_flag:
            arrive_steps += 1


        sim.doStep()
        for j in range(num_agents):
            env = globals()[f'env{j}']
            ideal_speed = sim.getAgentVelocity(j)
            true_speed = env.vxy
            state = torch.cat((torch.tensor(ideal_speed), true_speed.reshape(2, )))
            # tau = track_NN.cal_tau(state)
            tau = track_NN.cal_tau(state)
            env.step(tau)
        if rotate_flag:
            evader[0] += 0.12
            evader[1] = 0.02*(evader[0])**2 - 0.8*evader[0] + 28

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
    print(track_performance_time)
    print(track_performance_time / 400)
    print(track_performance_time / arrive_steps)
    print(arrive_steps)
    # plot_traj(px_lst_usv, py_lst_usv, px_lst_des, py_lst_des, obst_lst, num_agents=num_agents, circle_radius=e_radius,
    #           background_img=r'/home/liuyangyang/MARL/MpeRvo/plot/2.jpg')
    plot_traj_static(px_lst_usv, py_lst_usv, obst_lst, num_agents=num_agents, circle_radius=e_radius, xy_lim=[0, 90],
                     background_img=r'/home/liuyangyang/MARL/MpeRvo/plot/2.jpg',
                     save_path=os.path.dirname(os.getcwd()) + "/save/ImgSave/MpeTraj_evader.svg")
    # plot_state(px_lst_usv, px_lst_des, py_lst_usv, py_lst_des,
    #            vx_lst_usv, vx_lst_des, vy_lst_usv, vy_lst_des,
    #            num_agents, save_path=os.path.dirname(os.getcwd()) + "/save/ImgSave/")


