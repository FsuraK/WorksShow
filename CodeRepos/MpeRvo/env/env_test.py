from env.UsvBaseEnvTensorM1 import USV
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

init_pos = [0.0, 0.0, 0.0]
init_speed = [0.0, 0.0, 0.0]
time_interval = 0.1
usv = USV(init_pos, init_speed, time_interval=time_interval)
tau = torch.zeros((3, 1), dtype=torch.float)

t = 0
x_lst = []
y_lst = []
v0_lst = []
for i in range(400):
    t += time_interval
    # tau[0] = np.cos(t) * 10
    # tau[2] = np.sin(t) * 10
    tau[0] = random.uniform(-5, 10)
    tau[2] = random.uniform(-5, 10)
    # print(usv.euler(tau))
    pos, speed = usv.step(tau)
    x_lst.append([pos[0].item()])
    y_lst.append([pos[1].item()])

fig, ax = plt.subplots()
ax.plot(x_lst, y_lst, color='blue', linestyle='-.', label='Pursuer')
ax.scatter(x_lst[-1], y_lst[-1], marker='o', color='red', s=50)
ax.set_xlim([-5, 70])
ax.set_ylim([-5, 70])
ax.legend()
plt.show()