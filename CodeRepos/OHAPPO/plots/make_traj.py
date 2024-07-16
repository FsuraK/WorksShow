import numpy as np
import matplotlib.pyplot as plt
from plots.plot_MpeTraj import plot_traj

# init
time_interval = 0.1
obs = np.zeros((1, 4, 4))
obs[0][0][0] = 0
obs[0][0][1] = 30
obs[0][1][0] = 0
obs[0][1][1] = 15
obs[0][2][0] = 15
obs[0][2][1] = 0
obs[0][3][0] = 30
obs[0][3][1] = 0
x, y = [], []
x.append(obs[0][:, 0].tolist())
y.append(obs[0][:, 1].tolist())
x[0].append(obs[0][:, 2][0].tolist())
y[0].append(obs[0][:, 3][0].tolist())
for step in range(230):
    # obs update
    for agent_id in range(4):
        c = 4
        if agent_id < 2:
            exexp = obs[0][agent_id][2] - c
        else:
            exexp = obs[0][agent_id][2] + c
        if agent_id == 0 or agent_id == 2:
            eyexp = obs[0][agent_id][3] - c
        else:
            eyexp = obs[0][agent_id][3] + c

        dpe_x = obs[0][agent_id][0] - exexp
        dpe_y = obs[0][agent_id][1] - eyexp
        if dpe_x >= 1:
            vx = -2
        elif 0 <= dpe_x < 1:
            vx = -1
        elif -1 <= dpe_x < 0:
            vx = 1
        else:
            vx = 2

        if dpe_y >= 1:
            vy = -2
        elif 0 <= dpe_y < 1:
            vy = -1
        elif -1 <= dpe_y < 0:
            vy = 1
        else:
            vy = 2

        obs[0][agent_id][0] = obs[0][agent_id][0] + vx * time_interval + np.random.randint(-5, 8) / 20
        obs[0][agent_id][1] = obs[0][agent_id][1] + vy * time_interval + np.random.randint(-5, 8) / 20
        obs[0][agent_id][2] = obs[0][agent_id][2] + 0.2
        obs[0][agent_id][3] = obs[0][agent_id][3] + 0.2

    # x y e
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
plot_traj(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, '/home/lyy/Desktop/HAPPO-HATRPO/plots/imgs/make.eps')

