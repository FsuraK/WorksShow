import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r'/usr/share/fonts/simsun.ttc',size=13)
# plt.rcParams['ax']

from envs.musv.ADP0 import ADP, loss_cal_for_marl


def plot4_for_marl(x_raw_lst, x_obs_lst):
    # data
    x1_raw = [row[0] for row in x_raw_lst]
    x2_raw = [row[1] for row in x_raw_lst]
    x3_raw = [row[2] for row in x_raw_lst]
    x4_raw = [row[3] for row in x_raw_lst]
    x1_obs = [row[0] for row in x_obs_lst]
    x2_obs = [row[1] for row in x_obs_lst]
    x3_obs = [row[2] for row in x_obs_lst]
    x4_obs = [row[3] for row in x_obs_lst]

    # x-axis and plot init
    fig, axs = plt.subplots(4, 1, figsize=(8, 13))
    x = np.arange(0, len(x1_raw), 1)

    """plot 1"""
    axs[0].plot(x, x1_raw, label='sub-system')
    axs[0].plot(x, x1_obs, label='sub-observer')
    axs[0].set_title("$x_1$")
    # axs[0].set_xlabel("Environment total steps")
    axs[0].set_ylabel("Value")  # , fontproperties=font
    axs[0].legend(loc='lower right')
    # axs[0].set_xticks(np.arange(min(x), max(x)+2, 50))
    # axs[0].set_yticks(np.arange(min(np.array(x1_raw)), max(np.array(x1_raw))+200, 100))
    # axs[0].xaxis.grid(True)
    # axs[0].yaxis.grid(True)
    axs[0].grid(True)
    # axs[0].legend(prop=font)
    # 创建放大区域
    axins = inset_axes(axs[0], width="100%", height="100%", loc='lower left', bbox_to_anchor=(0.05, 0.40, 90/300, 90/250),
                       bbox_transform=axs[0].transAxes)
    # 在放大区域中绘制图像
    axins.plot(x, x1_raw, label='sub-system')
    axins.plot(x, x1_obs, label='sub-observer')
    # 设置放大区域的范围
    x1, x2, y1, y2 = -5, 25, -10, 30  # 指定放大的范围
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # 添加连接线
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="#e5feff", ec="#6c5ce7")


    """plot 2"""
    axs[1].plot(x, x2_raw, label='sub-system')
    axs[1].plot(x, x2_obs, label='sub-observer')
    axs[1].set_title("$x_2$")
    # axs[1].set_xlabel("Environment total steps")
    axs[1].set_ylabel("Value")
    axs[1].legend(loc='lower right')
    axs[1].grid(True)
    axins1 = inset_axes(axs[1], width="100%", height="100%", loc='lower left', bbox_to_anchor=(0.05, 0.50, 8/30, 9/30),
                       bbox_transform=axs[1].transAxes)
    axins1.plot(x, x2_raw, label='sub-system')
    axins1.plot(x, x2_obs, label='sub-observer')
    axins1.set_xlim(-5, 35)
    axins1.set_ylim(-5, 40)
    mark_inset(axs[1], axins1, loc1=2, loc2=4, fc="#e5feff", ec="#6c5ce7")


    """plot 3"""
    axs[2].plot(x, x3_raw, label='sub-system')
    axs[2].plot(x, x3_obs, label='sub-observer')
    axs[2].set_title("$x_3$")
    # axs[2].set_xlabel("Environment total steps")
    axs[2].set_ylabel("Value")
    axs[2].legend(loc='lower right')
    axs[2].grid(True)
    # axs[2].set_yticks([0, 2, 4, 6, 8])
    axins3 = inset_axes(axs[2], width="100%", height="100%", loc='lower left',
                        bbox_to_anchor=(0.08, 0.53, 3 / 30, 3 / 8),
                        bbox_transform=axs[2].transAxes)
    axins3.plot(x, x3_raw, label='sub-system')
    axins3.plot(x, x3_obs, label='sub-observer')
    axins3.set_xlim(-5, 15)
    axins3.set_ylim(2., 4)
    mark_inset(axs[2], axins3, loc1=2, loc2=4, fc="#e5feff", ec="#6c5ce7")
    axins3_ = inset_axes(axs[2], width="100%", height="100%", loc='center',
                        bbox_to_anchor=(0.4, 0.158, 3*2 / 30, 5 / 8),
                        bbox_transform=axs[2].transAxes)
    axins3_.plot(x, x3_raw, label='sub-system')
    axins3_.plot(x, x3_obs, label='sub-observer')
    axins3_.set_xlim(220, 250)
    axins3_.set_ylim(5, 7.5)
    mark_inset(axs[2], axins3_, loc1=2, loc2=4, fc="#e5feff", ec="#6c5ce7")

    """plot 4"""
    axs[3].plot(x, x4_raw, label='sub-system')
    axs[3].plot(x, x4_obs, label='sub-observer')
    axs[3].set_title("$x_4$")
    axs[3].set_xlabel("Environment total steps")
    axs[3].set_ylabel("Value")
    axs[3].legend(loc='lower right')
    axs[3].grid(True)
    axins4 = inset_axes(axs[3], width="100%", height="100%", loc='lower left',
                        bbox_to_anchor=(0.23, 0.15, 5 / 30, 6 / 10),
                        bbox_transform=axs[3].transAxes)
    axins4.plot(x, x4_raw, label='sub-system')
    axins4.plot(x, x4_obs, label='sub-observer')
    axins4.set_xlim(200, 230)
    axins4.set_ylim(3, 7)
    mark_inset(axs[3], axins4, loc1=2, loc2=4, fc="#e5feff", ec="#6c5ce7")

    plt.subplots_adjust(hspace=0.35)
    axs[2].yaxis.set_major_formatter('{:.1f}'.format)
    axs[3].yaxis.set_major_formatter('{:.1f}'.format)
    # for ax in axs:
    #     plt.setp(ax.get_yticklabels(), rotation=45)
    # fig.suptitle("Tracking effect", fontsize=20)
    # plt.savefig('/home/lyy/Desktop/happo_imgs/observer_obs/result_test5.eps', bbox_inches="tight")
    plt.savefig('/home/lyy/Desktop/tracking_result.pdf', bbox_inches="tight")
    plt.show()


adp = ADP(lr=5e-6, model_path='/home/lyy/Desktop/HAPPO-HATRPO/envs/musv/ADP_model.pt')
x_now_raw = torch.zeros((4, 1), dtype=torch.float32)
x_now_obs = torch.zeros((4, 1), dtype=torch.float32)
x_raw_lst = []
x_obs_lst = []
# x_now_raw[1][0] = 20
# x_now_raw[3][0] = 0
# x_now_raw[2][0] = 3
x_now_raw[0][0] = 0 # 10   # 0
x_now_raw[1][0] = 10 #15  # 10
x_now_raw[2][0] = 3 #10   # 3
x_now_raw[3][0] = 3 # 5   # 3


u_lst = np.load('/home/lyy/Desktop/happo_imgs/observer_obs/u_lst.txt.npy')
for step in range(300):
    u = torch.tensor(u_lst[step], dtype=torch.float32)
    x_raw_lst.append(x_now_raw.tolist())
    x_obs_lst.append(x_now_obs.tolist())

    obs = torch.cat([x_now_raw, x_now_obs])
    dvalue_dx = adp.model(obs.T)
    _, x_now_raw, x_now_obs = loss_cal_for_marl(obs, dvalue_dx, u)
    # x_now_raw = raw_sys_for_marl(x_now_raw, u)
plot4_for_marl(x_raw_lst, x_obs_lst)
# np.save('/home/lyy/Desktop/happo_imgs/observer_obs/x_raw_lst.npy', np.array(x_raw_lst))
# np.save("/home/lyy/Desktop/happo_imgs/observer_obs/x_obs_lst.npy", np.array(x_obs_lst))
# np.save("/home/lyy/Desktop/happo_imgs/observer_obs/u_lst.txt", np.array(u_lst))

# x_raw_lst = np.load('/home/lyy/Desktop/happo_imgs/observer_obs/x_raw_lst.npy')
# x_obs_lst = np.load('/home/lyy/Desktop/happo_imgs/observer_obs/x_obs_lst.npy')

