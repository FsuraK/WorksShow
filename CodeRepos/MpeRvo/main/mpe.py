import rvo2
import matplotlib.pyplot as plt
import random, math
import numpy as np
from plot.plot import plot_traj


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


"""param"""
max_speed = 5
num_agents = 4
# num_obst = 1
time_step = 1 / 10.
rotate_speed_rad = math.pi / 15
# radius=1 or 1.5 is look good
sim = rvo2.PyRVOSimulator(timeStep=1 / 10., neighborDist=20, maxNeighbors=4, timeHorizon=2, timeHorizonObst=0.8,
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
            (25, 35),
            (25, 45)]
num_obst = len(obst_pos)
for i in obst_pos:
    sim.addAgent(i)

usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]

# Obstacles
# o1 = sim.addObstacle([(20, 10), (55+15*math.cos(math.pi/4), 55+15*math.sin(math.pi/4))])
# sim.processObstacles()
# obst_lst = [[20, 10], [55+15*math.cos(math.pi/4), 55+15*math.sin(math.pi/4)]]


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
for step in range(400):
    sim.doStep()
    for i in range(num_obst):
        sim.setAgentPosition(num_agents + i, obst_pos[i])

    for i in range(num_agents):
        PrefVelocity = cal_pref_velocity(np.array(sim.getAgentPosition(i), dtype=float), target_pos[i], max_speed)
        sim.setAgentPrefVelocity(i, PrefVelocity)

    if not rotate_flag:
        usv_now_pos = [np.array(sim.getAgentPosition(i), dtype=float) for i in range(num_agents)]
        rotate_flag = all_arrive_target_pos(usv_now_pos, target_pos, num_agents)
    else:
        target_pos = rotate_points(target_pos, evader, rotate_speed_rad, time_step)

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

    x_lst.append(x_swap)
    y_lst.append(y_swap)
    x_lst_tar.append(x_swap_tar)
    y_lst_tar.append(y_swap_tar)

plot_traj(x_lst, y_lst, x_lst_tar, y_lst_tar, obst_lst, num_agents=num_agents)
