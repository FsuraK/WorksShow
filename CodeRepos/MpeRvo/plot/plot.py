import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mtick
from matplotlib import rcParams
import matplotlib
# matplotlib.use('agg')
import numpy as np
from util.BaseUtils import find_diff
from scipy.signal import savgol_filter as sf


def plot_traj(x, y, x_tar, y_tar, obst_lst=None, num_agents=1, circle_radius=15, xy_lim=None, data_filter=True,
              background_img=None):

    if data_filter:
        for i in range(num_agents):
            x_ = [row[i] for row in x]
            x_filter = sf(x_, 53, 5)
            for j in range(len(x)):
                x[j][i] = x_filter[j]

    if xy_lim is None:
        xy_lim = [-5, 90]
    fig, ax = plt.subplots()

    for t in range(len(x)):
        for i in range(num_agents):
            # agent pos and agent_target pos
            ax.scatter(x[t][i], y[t][i], c='b')
            ax.scatter(x_tar[t][i], y_tar[t][i], c='g')
            # evader pos
            if i == num_agents - 1:
                ax.scatter(x[t][i + 1], y[t][i + 1], c='r')
                # circle
                draw_circle = plt.Circle((x[t][num_agents], y[t][num_agents]),
                                         circle_radius, fill=False, linestyle='dashed')
                ax.set_aspect(1)
                ax.add_artist(draw_circle)
        if obst_lst is not None:
            for j in range(len(obst_lst)):
                ax.scatter(obst_lst[j][0], obst_lst[j][1], c='black', s=8)
    # if background_img is not None:
    #     bg = mpimg.imread(background_img)
    #     # ax.imshow(bg, extent=[-20, 70, -20, 70], aspect='auto')  # for MpeRvoLinearTrack.py
    #     ax.imshow(bg, extent=[-10, 90, -10, 90], aspect='auto')  # for MpeRvoLinearTrack_evader.py

        # 设置坐标轴范围

        ax.set_xlim(xy_lim)
        ax.set_ylim(xy_lim)
        # plot
        plt.show(block=False)
        # if t == 0:
        #     plt.pause(2)
        plt.pause(0.00005)
        ax.clear()
    plt.close()


def plot_traj_static(x, y, obst_lst=None, save_path=None, num_agents=1, circle_radius=15, xy_lim=None,
                     double_circle=False, background_img=None):
    if xy_lim is None:
        xy_lim = [10, 40]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    # fig, ax = plt.subplots(figsize=(10, 8))
    # color = ['#ffc000', '#aa23ff', '#c5e0b4', '#0cff0c']
    color = ['#c5e0b4', '#0cff0c', '#ffc000', '#aa23ff']

    ex = [row[num_agents] for row in x]
    ey = [row[num_agents] for row in y]

    for i in range(num_agents):
        px = [row[i] for row in x]
        py = [row[i] for row in y]
        if i==2:
            py = sf(py, 23, 4)
        if i==0:
            py = sf(py, 13, 4)
        # px = sf(px, 13, 4)
        # py = sf(py, 53, 5)
        # px = sf(px, 53, 5)
        ax.plot(px, py, color=color[i], linestyle='-.', linewidth=1)
        ax.scatter(px[-1], py[-1], marker='o', color=color[i], s=10, label=f'Pursuer {i + 1}')

    ax.plot(ex, ey, color='red', linestyle='-.')
    ax.scatter(ex[-1], ey[-1], marker='o', color='red', s=10, label='Evader')

    for j in range(len(obst_lst)):
        if j == 0:
            ax.scatter(obst_lst[j][0], obst_lst[j][1], c='black', s=10, label='Obstacle')
        else:
            ax.scatter(obst_lst[j][0], obst_lst[j][1], c='black', s=10)
    # draw circle
    draw_circle = plt.Circle((x[-1][num_agents], y[-1][num_agents]),
                             circle_radius, fill=False, linestyle='dashed')
    ax.set_aspect(1)
    ax.add_artist(draw_circle)
    if double_circle:
        draw_circle_start = plt.Circle((x[0][num_agents], y[0][num_agents]),
                                       circle_radius, fill=False, linestyle='dashed')
        ax.set_aspect(1)
        ax.add_artist(draw_circle_start)

    ax.set_xlim(xy_lim)
    ax.set_ylim(xy_lim)
    # ax.set_ylim([0, 60])
    if background_img is not None:
        bg = mpimg.imread(background_img)
        # ax.imshow(bg, extent=[-20, 70, -20, 70], aspect='auto')  # for MpeRvoLinearTrack.py
        ax.imshow(bg, extent=[0, 90, 0, 90], aspect='auto')  # for MpeRvoLinearTrack_evader.py
    ax.legend()
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    # plot
    # plt.show(block=False)
    # plt.close()


def plot_state(px_lst_usv, px_lst_des, py_lst_usv, py_lst_des,
               vx_lst_usv, vx_lst_des, vy_lst_usv, vy_lst_des,
               num_agents=4, save_path=None):
    fig_vec, ax_vec = plt.subplots(2, 1, figsize=(8, 3.25 * 2))
    fig_pos, ax_pos = plt.subplots(2, 1, figsize=(8, 3.25 * 2))

    p_range = np.arange(0, len(px_lst_usv), 1)
    v_range = np.arange(0, len(vx_lst_usv), 1)

    # env = globals()[f'env{j}']
    ax_vec0_max, ax_vec0_min, ax_vec1_max, ax_vec1_min = 0, 0, 0, 0
    ax_pos0_max, ax_pos0_min, ax_pos1_max, ax_pos1_min = 0, 0, 0, 0
    for i in range(num_agents):
        px_erro = [px_lst_usv[j][i] - px_lst_des[j][i] for j in range(len(px_lst_usv))]
        py_erro = [py_lst_usv[j][i] - py_lst_des[j][i] for j in range(len(py_lst_usv))]
        vx_erro = [vx_lst_usv[j][i] - vx_lst_des[j][i] for j in range(len(vx_lst_usv))]
        vy_erro = [vy_lst_usv[j][i] - vy_lst_des[j][i] for j in range(len(vy_lst_usv))]
        v = 'v'
        ax_vec[0].plot(v_range, vx_erro, linestyle='dotted', label=f'$\\mathbf{{v}}_x^{i+1} - \\mathbf{{vd}}_x^{i+1}$')
        ax_vec[1].plot(v_range, vy_erro, linestyle='dotted', label=f'$\\mathbf{{v}}_y^{i+1} - \\mathbf{{vd}}_y^{i+1}$')
        ax_pos[0].plot(p_range, px_erro, linestyle='-', label=f'$\\mathbf{{p}}_x^{i+1}-\\mathbf{{pd}}_x^{i+1}$')
        ax_pos[1].plot(p_range, py_erro, linestyle='-', label=f'$\\mathbf{{p}}_y^{i+1}-\\mathbf{{pd}}_y^{i+1}$')

        # baseline max and min data
        ax_vec0_max = max(vx_erro) if max(vx_erro) > ax_vec0_max else ax_vec0_max
        ax_vec1_max = max(vy_erro) if max(vy_erro) > ax_vec1_max else ax_vec1_max
        ax_vec0_min = min(vx_erro) if min(vx_erro) < ax_vec0_min else ax_vec0_min
        ax_vec1_min = min(vy_erro) if min(vy_erro) < ax_vec1_min else ax_vec1_min

    ax_vec[0].set_ylabel("Horizontal velocity error (m/s)")
    ax_vec[1].set_ylabel("Vertical velocity error (m/s)")
    ax_pos[0].set_ylabel("Horizontal position error (m/s)")
    ax_pos[1].set_ylabel("Vertical position error (m/s)")

    ax_vec[0].legend()
    ax_vec[1].legend()
    ax_pos[0].legend()
    ax_pos[1].legend()

    # y=0 base_line
    ax_vec[0].axhline(y=0, color='black', linestyle='--')
    ax_vec[1].axhline(y=0, color='black', linestyle='--')
    ax_pos[0].axhline(y=0, color='black', linestyle='--')
    ax_pos[1].axhline(y=0, color='black', linestyle='--')

    # data base_line and tex --- ax_vec
    ax_vec[0].axhline(y=ax_vec0_max, color='blue', linestyle='--', linewidth=0.9)
    ax_vec[0].axhline(y=ax_vec0_min, color='blue', linestyle='--', linewidth=0.9)
    ax_vec[1].axhline(y=ax_vec1_max, color='blue', linestyle='--', linewidth=0.9)
    ax_vec[1].axhline(y=ax_vec1_min, color='blue', linestyle='--', linewidth=0.9)

    ax_vec[0].text(100, ax_vec0_max, f'$maximum:{round(ax_vec0_max, 2)}$', color='black', va='bottom')
    ax_vec[0].text(100, ax_vec0_min, f'$minimum:{round(ax_vec0_min, 2)}$', color='black', va='top')
    ax_vec[1].text(100, ax_vec1_max, f'$maximum:{round(ax_vec1_max, 2)}$', color='black', va='bottom')
    ax_vec[1].text(100, ax_vec1_min, f'$minimum:{round(ax_vec1_min, 2)}$', color='black', va='top')

    # data base_line and tex --- ax_pos
    ax_pos[0].axvline(80, color='purple', linestyle='--', linewidth=0.9)
    ax_pos[0].axvline(170, color='purple', linestyle='--', linewidth=0.9)
    ax_pos[0].axvline(275, color='blue', linestyle='--', linewidth=0.9)
    ax_pos[0].axvline(360, color='blue', linestyle='--', linewidth=0.9)

    ax_pos[1].axvline(120, color='purple', linestyle='--', linewidth=0.9)
    ax_pos[1].axvline(210, color='purple', linestyle='--', linewidth=0.9)
    ax_pos[1].axvline(320, color='blue', linestyle='--', linewidth=0.9)
    ax_pos[1].axvline(400, color='blue', linestyle='--', linewidth=0.9)

    """arrow and text"""
    ax_pos[0].text(185, 15, '$Obstacle$ $effect$', color='black', va='bottom')
    ax_pos[0].annotate(
        '',
        xy=(154, 17),  # target point
        xytext=(184, 17),  # start point
        arrowprops=dict(arrowstyle='->')
    )
    ax_pos[0].annotate(
        '',
        xy=(291, 17),  # target point
        xytext=(261, 17),  # start point
        arrowprops=dict(arrowstyle='->')
    )

    ax_pos[1].text(225, 15, '$Obstacle$ $effect$', color='black', va='bottom')
    ax_pos[1].annotate(
        '',
        xy=(194, 17),  # target point
        xytext=(224, 17),  # start point
        arrowprops=dict(arrowstyle='->')
    )
    ax_pos[1].annotate(
        '',
        xy=(331, 17),  # target point
        xytext=(301, 17),  # start point
        arrowprops=dict(arrowstyle='->')
    )

    # set x,y lim
    ax_vec[0].set_ylim([-1.1, 1.1])
    ax_vec[1].set_ylim([-1.1, 1.1])
    ax_vec[0].grid(True)
    ax_vec[1].grid(True)
    ax_pos[0].grid(True)
    ax_pos[1].grid(True)

    # SAVE IMAGE
    fig_vec.text(0.5, 0.04, r'Time (s)', ha='center')
    fig_pos.text(0.5, 0.04, r'Time (s)', ha='center')
    fig_vec.savefig(save_path + r"vec_error.pdf")
    fig_pos.savefig(save_path + r"pos_error.pdf")
    plt.show()


def save_txt_traj(x, y, exy, save_path):
    """ input
    ( [x1, x2, x3, ...], [y1, y2, y3, ...], [ex, ey], save_path= /lyy/abc/  --(root path with /) )
    """
    k0, k1 = 0, 0
    for i in x:
        k0 += 1
        np.savetxt(save_path + f'x_{k0}.txt', i)
    for i in y:
        k1 += 1
        np.savetxt(save_path + f'y_{k1}.txt', i)

    np.savetxt(save_path + f'x_e.txt', exy[0])
    np.savetxt(save_path + f'y_e.txt', exy[1])


def plot_est_traj(vxy_est_lst, vxy_env_lst, save_path=None, xy_lim=None, background_img=None):
    v0_est = [row[0] for row in vxy_est_lst]
    v1_est = [row[1] for row in vxy_est_lst]
    v2_est = [row[2] for row in vxy_est_lst]
    v0_env = [row[0] for row in vxy_env_lst]
    v1_env = [row[1] for row in vxy_env_lst]
    v2_env = [row[2] for row in vxy_env_lst]
    max_diff_index_0, max_diff_0 = find_diff(v0_est, v0_env, "max")
    max_diff_index_1, max_diff_1 = find_diff(v1_est, v1_env, "max")
    max_diff_index_2, max_diff_2 = find_diff(v2_est, v2_env, "max")
    time = np.arange(0, len(v0_est), 1)

    fig, ax = plt.subplots(3, 1, figsize=(8, 3.25 * 2))
    plt.subplots_adjust(hspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
    # ax 0
    ax[0].plot(time, v0_est, color='red', linewidth=0.8, label='Predicted')
    ax[0].plot(time, v0_env, color='green', linestyle='-.', linewidth=0.8, label='True')
    ax[0].grid(True)
    # ax 1
    ax[1].plot(time, v1_est, color='red', linewidth=0.8, label='Predicted')
    ax[1].plot(time, v1_env, color='green', linestyle='-.', linewidth=0.8, label='True')
    ax[1].grid(True)
    # ax 2
    ax[2].plot(time, v2_est, color='red', linewidth=0.8, label='Predicted')
    ax[2].plot(time, v2_env, color='green', linestyle='-.', linewidth=0.8, label='True')
    ax[2].grid(True)

    # ax[0].set_ylabel("Surge Velocity $u (m/s)$")
    # ax[1].set_ylabel("Sway Velocity $v (m/s)$")
    # ax[2].set_ylabel("Yaw Angular Velocity  $r (m/s)$")

    ax[0].set_ylabel(r"$u$ (m/s)")
    ax[1].set_ylabel(r"$v$ (m/s)")
    ax[2].set_ylabel(r"$r$ (rad/s)")
    ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax[2].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.text(0.5, 0.04, r'Time (s)', ha='center')

    """in the max diff draw line and text"""
    interval0 = 30
    interval1 = 30
    interval2 = 30
    ax[0].hlines(y=[v0_est[max_diff_index_0], v0_env[max_diff_index_0]],
                 xmin=[max_diff_index_0 - interval0, max_diff_index_0 - interval0],
                 xmax=[max_diff_index_0 + interval0, max_diff_index_0 + interval0],
                 colors='black', linewidth=1, linestyle='--')
    ax[1].hlines(y=[v1_est[max_diff_index_1], v1_env[max_diff_index_1]],
                 xmin=[max_diff_index_1 - interval1, max_diff_index_1 - interval1],
                 xmax=[max_diff_index_1 + interval1, max_diff_index_1 + interval1],
                 colors='black', linewidth=1, linestyle='--')
    ax[2].hlines(y=[v2_est[max_diff_index_2], v2_env[max_diff_index_2]],
                 xmin=[max_diff_index_2 - interval2, max_diff_index_2 - interval2],
                 xmax=[max_diff_index_2 + interval2, max_diff_index_2 + interval2],
                 colors='black', linewidth=1, linestyle='--')
    ax[0].text(max_diff_index_0, min(v0_est[max_diff_index_0], v0_env[max_diff_index_0]) - 0.08,
               f'${round(max_diff_0, 2)}$', color='black', va='top')
    ax[1].text(max_diff_index_1, min(v1_est[max_diff_index_1], v1_env[max_diff_index_1]) - 0.03,
               f'${round(max_diff_1, 2)}$', color='black', va='top')
    ax[2].text(max_diff_index_2, min(v2_est[max_diff_index_2], v2_env[max_diff_index_2]) - 0.1,
               f'${round(max_diff_2, 2)}$', color='black', va='top')
    # ax[0].annotate(
    #     '',
    #     xy=(50, 2),  # target point
    #     xytext=(0, 0),  # start point
    #     arrowprops=dict(arrowstyle='->')
    # )
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
