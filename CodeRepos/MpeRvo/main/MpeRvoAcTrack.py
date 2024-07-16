import os
import rvo2
import random, math
import numpy as np
import torch
from tqdm import tqdm

from plot.plot import plot_traj
from env.UsvBaseEnvTensorM1 import USV
from util.NetUtils import ActorCritic


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


def reset_sim_pos_speed(sim):
    sim.setAgentPosition(usv0, (10, 10))
    sim.setAgentPosition(usv1, (10, 20))
    sim.setAgentPosition(usv2, (10, 30))
    sim.setAgentPosition(usv3, (10, 40))
    sim.setAgentVelocity(usv0, (0, 0))
    sim.setAgentVelocity(usv1, (0, 0))
    sim.setAgentVelocity(usv2, (0, 0))
    sim.setAgentVelocity(usv3, (0, 0))


"""-------------------------------------------main--------------------------------------------"""
"""param"""
max_speed = 6
num_agents = 4
# num_obst = 1
time_step = 1 / 100.
rotate_speed_rad = math.pi / 15
# radius=1 or 1.5 is look good
sim = rvo2.PyRVOSimulator(timeStep=time_step, neighborDist=20, maxNeighbors=4, timeHorizon=2, timeHorizonObst=0.8,
                          radius=1.3, maxSpeed=max_speed)

# INIT USV
usv0 = sim.addAgent((10, 10))
usv1 = sim.addAgent((10, 20))
usv2 = sim.addAgent((10, 30))
usv3 = sim.addAgent((10, 40))

# INIT evader and target pos
evader = np.array([55, 55], dtype=float)
e_angle_4 = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
e_radius = 15
target_pos = cal_target_pos(evader, e_angle_4, e_radius)

# INIT OBST
obst_pos = [(55 + 15 * math.cos(math.pi / 4), 55 + 15 * math.sin(math.pi / 4)),
            (15, 25),
            (25, 35), ]
num_obst = len(obst_pos)
for i in obst_pos:
    sim.addAgent(i)

usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]

# INIT NN and USV env
env0 = USV((10, 10, 0), time_interval=time_step)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.dirname(os.getcwd()) + "/save/TrackModel/track_NN.pt"
model = ActorCritic(input_dim=4, output_dim=2)
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-5)
critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-4)

""" INIT list
x_lst, y_lst = time_steps * np.array([usv0, ..., usvn, evader])
x_lst_des, y_lst_des = time_steps * np.array([target_pos_1, ..., target_pos_n])
"""
x_lst, y_lst, x_lst_tar, y_lst_tar = [], [], [], []
x_swap, y_swap = [], []
for i in range(num_agents):
    x_swap.append(sim.getAgentPosition(i)[0])
    y_swap.append(sim.getAgentPosition(i)[1])
    if i == num_agents - 1:
        x_swap.append(evader[0])
        y_swap.append(evader[1])
x_lst.append(x_swap)
y_lst.append(y_swap)
x_lst_tar = [[i[0] for i in target_pos]]
y_lst_tar = [[i[1] for i in target_pos]]

obst_lst = []
for i in range(num_obst):
    obst_lst.append(sim.getAgentPosition(num_agents + i))
rotate_flag = False


"""here start episode"""
Episodes = 200
steps = 400
speed_erro_sum, episode_record, bf_reward = 0, [], 1e+5
tq_bar = tqdm(range(Episodes))
# if_train = False
if_train = True
loss_func = torch.nn.MSELoss()
if if_train:
    for episode in tq_bar:
        env0.reset()
        reset_sim_pos_speed(sim)
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
        _, true_speed = env0.cal_vxy()
        state = torch.cat((torch.tensor(ideal_speed), true_speed.reshape(2, )))

        tq_bar.set_description(f'Episode [ {episode + 1} / {Episodes} ]')

        for step in range(steps):

            action = model.get_action(state)
            zero = torch.tensor([0.0])
            tau = torch.cat((action[:1], zero, action[1:]))
            value = model.get_q_value(state, action)
            r = torch.norm(state[:2] - state[2:])

            env0.step(tau.reshape(3, 1))
            env0.grad_free()
            true_speed_next_tuple, true_speed_next = env0.cal_vxy()
            true_pos_next = tuple([i.item() for i in env0.eta[:2].detach()])

            ideal_speed = sim.getAgentVelocity(usv0)
            sim.setAgentPosition(usv0, true_pos_next)
            sim.setAgentVelocity(usv0, true_speed_next_tuple)
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
            ideal_speed_next = sim.getAgentVelocity(usv0)
            true_speed_next = true_speed_next
            state_next = torch.cat((torch.tensor(ideal_speed_next), true_speed_next.reshape(2, )))
            action_next = model.get_action(state_next)
            value_next = model.get_q_value(state_next, action_next)
            target = r + 0.9 * value_next
            loss_critic = loss_func(target, value)
            # update critic
            critic_optimizer.zero_grad()
            loss_critic.backward(retain_graph=True)
            critic_optimizer.step()

            # update actor
            env0.grad_free()
            value_ = model.get_q_value(state_next, action_next)
            loss_actor = -torch.mean(value_.squeeze())
            actor_optimizer.zero_grad()
            loss_actor.backward(retain_graph=True)
            actor_optimizer.step()
            state = state_next

            speed_erro_sum = speed_erro_sum + round(r.item(), 5)
            # print(loss)
            if step == steps - 1:
                last_step_erro = round(r.item(), 5)

            if episode == Episodes - 1:
                x_swap, y_swap = [], []
                x_swap_tar, y_swap_tar = [], []
                for i in range(num_agents):
                    x_swap.append(sim.getAgentPosition(i)[0])
                    y_swap.append(sim.getAgentPosition(i)[1])
                    x_swap_tar.append(target_pos[i][0])
                    y_swap_tar.append(target_pos[i][1])
                    if i == num_agents - 1:
                        x_swap.append(evader[0])
                        y_swap.append(evader[1])
                if step % 5 == 0:
                    x_lst.append(x_swap)
                    y_lst.append(y_swap)
                    x_lst_tar.append(x_swap_tar)
                    y_lst_tar.append(y_swap_tar)

        episode_record.append(speed_erro_sum)
        score = 0
        ep_mean_erro = np.mean(episode_record[-10:])
        # Init_reward = ep_mean_reward if episode == 10 else 0
        # bf_reward = episode_reward if episode_reward <= bf_reward else bf_reward
        if ep_mean_erro <= bf_reward:
            bf_reward = ep_mean_erro
            # if episode >= 10:
            #     track_NN.save_model(model_save_path)
        tq_bar.set_postfix({'lastMeanRewards': f'{ep_mean_erro:.2f}', 'BEST': f'{bf_reward:.2f}',
                            'last_step_loss': f'{last_step_erro:.2f}', 'critic_loss': f'{loss_critic}'})
        speed_erro_sum = 0
else:
    track_NN = Tracking(lr=1e-4, input_dim=6, output_dim=2, model_path= \
                        os.path.dirname(os.getcwd()) + "/save/TrackModel/track_NN_base.pt")
    env0.reset()
    reset_sim_pos_speed()

    for step in range(steps):
        for i in range(num_obst):
            sim.setAgentPosition(num_agents + i, obst_pos[i])

        for i in range(num_agents):
            PrefVelocity = cal_pref_velocity(np.array(sim.getAgentPosition(i), dtype=float),
                                             target_pos[i], max_speed)
            sim.setAgentPrefVelocity(i, PrefVelocity)

        if not rotate_flag:
            usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
            rotate_flag = all_arrive_target_pos(usv_now_pos, target_pos, num_agents)
        else:
            target_pos = rotate_points(target_pos, evader, rotate_speed_rad, time_step)

        sim.doStep()
        ideal_speed = sim.getAgentVelocity(usv0)
        state = torch.cat((torch.tensor(ideal_speed), env0.eta[2].reshape(1, ), env0.v.reshape(3, )))
        tau = track_NN.cal_u_2dim(state)
        env0.step(tau.reshape(3, 1))
        true_speed_next = env0.cal_vxy()
        true_pos_next = tuple([i.item() for i in env0.eta[:2].detach()])
        sim.setAgentPosition(usv0, true_pos_next)
        sim.setAgentVelocity(usv0, true_speed_next)

        x_swap, y_swap = [], []
        x_swap_tar, y_swap_tar = [], []
        for i in range(num_agents):
            x_swap.append(sim.getAgentPosition(i)[0])
            y_swap.append(sim.getAgentPosition(i)[1])
            x_swap_tar.append(target_pos[i][0])
            y_swap_tar.append(target_pos[i][1])
            if i == num_agents - 1:
                x_swap.append(evader[0])
                y_swap.append(evader[1])
        if step % 10 == 0:
            x_lst.append(x_swap)
            y_lst.append(y_swap)
            x_lst_tar.append(x_swap_tar)
            y_lst_tar.append(y_swap_tar)
plot_traj(x_lst, y_lst, x_lst_tar, y_lst_tar, obst_lst, num_agents=num_agents)


