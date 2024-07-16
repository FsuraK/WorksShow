from plot.plot import plot_traj_static
import numpy as np
import math, os


px_lst_usv, py_lst_usv = [], []
px0 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/px_0.txt")
py0 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/py_0.txt")
px1 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/px_1.txt")
py1 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/py_1.txt")
px2 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/px_2.txt")
py2 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/py_2.txt")
px3 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/px_3.txt")
py3 = np.loadtxt(rf"/home/liuyangyang/MARL/MpeRvo/SOTA/PPO-Clip/data/1/py_3.txt")

for i in range(len(px0)):
    swap_x, swap_y = [], []

    swap_x.append(px0[i][0])
    swap_x.append(px1[i][0])
    swap_x.append(px2[i][0])
    swap_x.append(px3[i][0])
    swap_x.append(px3[i][1])

    swap_y.append(py0[i][0])
    swap_y.append(py1[i][0])
    swap_y.append(py2[i][0])
    swap_y.append(py3[i][0])
    swap_y.append(py3[i][1])

    px_lst_usv.append(swap_x)
    py_lst_usv.append(swap_y)


e_radius = 12
obst_pos = [(30.606601717798213, 30.606601717798213),
            (9.393398282201789, 30.606601717798213),
            (10., 10.),
            (25., 10.),
            (18., 12.),
            (10., 25.),
            (20., 35.),
            (53., 8.),
            (50., 55.),
            (49., 20.),
            (55., 15.),
            (10., 40.),
            (45., 43.),
            ]

plot_traj_static(px_lst_usv, py_lst_usv, obst_pos, num_agents=4, circle_radius=e_radius, xy_lim=[0, 90],
                         background_img=r'/home/liuyangyang/MARL/MpeRvo/plot/2.jpg',
                         save_path=os.path.dirname(os.getcwd()) + "/ImgSave/Sota2Rvo.svg")