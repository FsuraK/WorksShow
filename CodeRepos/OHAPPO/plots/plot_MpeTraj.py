import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'/usr/share/fonts/simsun.ttc', size=13)
from matplotlib.patches import Polygon, PathPatch, Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.markers import MarkerStyle
import matplotlib.image as mpimg
import svgpathtools
import matplotlib.transforms as transforms
import svgpath2mpl


def load_data(save_root_path):
    # load x
    x_1 = np.loadtxt(save_root_path + 'x_1.txt')
    x_2 = np.loadtxt(save_root_path + 'x_2.txt')
    x_3 = np.loadtxt(save_root_path + 'x_3.txt')
    x_4 = np.loadtxt(save_root_path + 'x_4.txt')
    # load y
    y_1 = np.loadtxt(save_root_path + 'y_1.txt')
    y_2 = np.loadtxt(save_root_path + 'y_2.txt')
    y_3 = np.loadtxt(save_root_path + 'y_3.txt')
    y_4 = np.loadtxt(save_root_path + 'y_4.txt')
    # load exy
    x_e = np.loadtxt(save_root_path + 'x_e.txt')
    y_e = np.loadtxt(save_root_path + 'y_e.txt')
    return x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e


def plot_traj_lin_raw(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, save_root_path):
    fig, ax = plt.subplots()
    ax.plot(x_e, y_e, color='red', linestyle='solid', label='Evader')
    ax.plot(x_1, y_1, color='blue', linestyle='--', label='Pursuer1')
    ax.plot(x_2, y_2, color='green', linestyle='--', label='Pursuer2')
    ax.plot(x_3, y_3, color='orange', linestyle='--', label='Pursuer3')
    ax.plot(x_4, y_4, color='purple', linestyle='--', label='Pursuer4')
    # ax.plot(x_5, y_5, color='grey',linestyle='--',label='Pursuer5')

    ax.scatter(x_1[-1], y_1[-1], marker='o', color='blue', s=30)
    ax.scatter(x_2[-1], y_2[-1], marker='o', color='green', s=30)
    ax.scatter(x_3[-1], y_3[-1], marker='o', color='orange', s=30)
    ax.scatter(x_4[-1], y_4[-1], marker='o', color='purple', s=30)
    ax.scatter(x_e[-1], y_e[-1], marker='*', color='red', s=70)

    ax.set_xlim([-5, 40])
    ax.set_ylim([-5, 40])
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.legend()
    plt.savefig(save_root_path + 'mpe_lin.eps')
    plt.show()


def plot_traj(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, save_root_path):
    fig, ax = plt.subplots()
    ax.plot(x_e, y_e, color='red', linestyle='solid', label='Evader')
    ax.plot(x_1, y_1, color='blue', linestyle='--', label='Pursuer1')
    ax.plot(x_2, y_2, color='green', linestyle='--', label='Pursuer2')
    ax.plot(x_3, y_3, color='orange', linestyle='--', label='Pursuer3')
    ax.plot(x_4, y_4, color='purple', linestyle='--', label='Pursuer4')
    # ax.plot(x_5, y_5, color='grey',linestyle='--',label='Pursuer5')

    ax.scatter(x_1[-1], y_1[-1], marker='o', color='blue', s=30)
    ax.scatter(x_2[-1], y_2[-1], marker='o', color='green', s=30)
    ax.scatter(x_3[-1], y_3[-1], marker='o', color='orange', s=30)
    ax.scatter(x_4[-1], y_4[-1], marker='o', color='purple', s=30)
    ax.scatter(x_e[-1], y_e[-1], marker='*', color='red', s=70)

    ax.set_xlim([-5, 70])
    ax.set_ylim([-5, 70])
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.legend()
    plt.savefig(save_root_path + 'mpe_lin.eps')
    # plt.show()


def plot_traj_lin_polish(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, save_root_path):
    fig, ax = plt.subplots()
    ax.plot(x_e, y_e, color='red', linestyle='-.', label='逃逸者')
    ax.plot(x_1, y_1, color='blue', linestyle='--', label='追击者1')
    ax.plot(x_2, y_2, color='green', linestyle='--', label='追击者2')
    ax.plot(x_3, y_3, color='orange', linestyle='--', label='追击者3')
    ax.plot(x_4, y_4, color='purple', linestyle='--', label='追击者4')

    ax.scatter(x_1[-1], y_1[-1], marker='o', color='blue', s=50)
    ax.scatter(x_2[-1], y_2[-1], marker='o', color='green', s=50)
    ax.scatter(x_3[-1], y_3[-1], marker='o', color='orange', s=50)
    ax.scatter(x_4[-1], y_4[-1], marker='o', color='purple', s=50)
    ax.scatter(x_e[-1], y_e[-1], marker='*', color='red', s=80)

    points = [(x_1[-1], y_1[-1]), (x_2[-1], y_2[-1]), (x_3[-1], y_3[-1]), (x_4[-1], y_4[-1])]
    polygon = Polygon(points, closed=True, fill=True, linestyle=':', color='blue', linewidth=2,
                      alpha=0.1)
    ax.add_patch(polygon)

    ax.set_xlim([-4, 37])
    ax.set_ylim([-4, 37])
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.legend(prop=font)
    # plt.savefig(save_root_path + 'mpe_lin_sur.pdf')
    plt.savefig('/home/lyy/Desktop/mpe1_.pdf')
    plt.show()


def plot_traj_cir_polish(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, save_root_path):
    fig, ax = plt.subplots(3, 1, figsize=(5, 14))
    a, b, c, d = 0, 50, 120, 200
    # fig, ax = plt.subplots()
    # ax.plot(x_e[:b+1], y_e[:b+1], color='red', linestyle='-.')
    # ax.plot(x_1[:b+1], y_1[:b+1], color='blue', linestyle='--', linewidth=0.5)  # , label='Pursuer1'
    # ax.plot(x_2[:b+1], y_2[:b+1], color='green', linestyle='--', linewidth=0.5)
    # ax.plot(x_3[:b+1], y_3[:b+1], color='orange', linestyle='--', linewidth=0.5)
    # ax.plot(x_4[:b+1], y_4[:b+1], color='purple', linestyle='--', linewidth=0.5)

    total_step = [b, c, d]
    for step, i in zip(total_step, range(3)):

        ax[i].plot(x_e[:step + 1], y_e[:step + 1], color='red', linestyle='-.')
        ax[i].plot(x_1[:step + 1], y_1[:step + 1], color='blue', linestyle='--', linewidth=0.5)  # , label='Pursuer1'
        ax[i].plot(x_2[:step + 1], y_2[:step + 1], color='green', linestyle='--', linewidth=0.5)
        ax[i].plot(x_3[:step + 1], y_3[:step + 1], color='orange', linestyle='--', linewidth=0.5)
        ax[i].plot(x_4[:step + 1], y_4[:step + 1], color='purple', linestyle='--', linewidth=0.5)

        ax[i].scatter(x_e[0], y_e[0], marker='*', color='red', s=80, label='逃逸者')
        ax[i].scatter(x_1[0], y_1[0], marker='o', color='blue', s=50, label='追击者1')
        ax[i].scatter(x_2[0], y_2[0], marker='o', color='green', s=50, label='追击者2')
        ax[i].scatter(x_3[0], y_3[0], marker='o', color='orange', s=50, label='追击者3')
        ax[i].scatter(x_4[0], y_4[0], marker='o', color='purple', s=50, label='追击者4')

        ax[i].scatter(x_e[step], y_e[step], marker='*', color='red', s=80)
        ax[i].scatter(x_1[step], y_1[step], marker='o', color='blue', s=50)
        ax[i].scatter(x_2[step], y_2[step], marker='o', color='green', s=50)
        ax[i].scatter(x_3[step], y_3[step], marker='o', color='orange', s=50)
        ax[i].scatter(x_4[step], y_4[step], marker='o', color='purple', s=50)

        if i != 2:
            points = [(x_1[step], y_1[step]),
                      (x_2[step], y_2[step]),
                      (x_3[step], y_3[step]),
                      (x_4[step], y_4[step])]
        else:
            points = [(x_1[step], y_1[step]),
                      (x_4[step], y_4[step]),
                      (x_2[step], y_2[step]),
                      (x_3[step], y_3[step])]
        polygon = Polygon(points, closed=True, fill=True, linestyle=':', color='blue', linewidth=2,
                          alpha=0.1)
        ax[i].add_patch(polygon)

    for i in range(3):
        ax[i].set_xlim([-5, 75])
        ax[i].set_ylim([-5, 75])
        ax[i].set_xlabel("x(m)")
        ax[i].set_ylabel("y(m)")
        if i == 0:
            env_step = 50
        elif i == 1:
            env_step = 120
        else:
            env_step = 200
        # ax[i].set_title(f"{env_step} Env Steps")
        # ax[i].legend()
        ax[i].set_title(f"{env_step} 环境步长", fontproperties=font)
        ax[i].legend(prop=font)
    plt.subplots_adjust(hspace=0.3)
    # plt.savefig(save_root_path + 'mpe_cir_sur.pdf')
    plt.savefig('/home/lyy/Desktop/mpe2_.pdf')
    plt.show()


if __name__ == '__main__':
    # save_root_path = '/home/lyy/Desktop/happo_imgs/MpeTraj_plot/10/'
    save_root_path = '/home/lyy/Desktop/MpeTraj_good/分散/9/'
    # save_root_path = '/home/lyy/Desktop/MpeTraj_good/分散/state_3/9/'
    x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e = load_data(save_root_path)
    # plot_traj_lin_raw(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, save_root_path)
    plot_traj_lin_polish(0.5 * x_1, 0.5 * x_2, 0.5 * x_3, 0.5 * x_4, 0.5 * y_1, 0.5 * y_2, 0.5 * y_3, 0.5 * y_4,
                         0.5 * x_e,
                         0.5 * y_e, save_root_path)
    # plot_traj_cir_polish(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4, x_e, y_e, save_root_path)
