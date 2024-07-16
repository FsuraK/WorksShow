#
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#font = FontProperties(fname=r'/usr/share/fonts/simsun.ttc',size=13)

# torch.manual_seed(0)
# print(torch.rand(1))
# print(torch.rand(1)) # 和numpy一样，生效一次
# class musv_test:
#     def __init__(self):
#         super(musv_test).__init__()
#         self.obs = np.random.randint(0, 255, size=(8,8,2))
#
#     def printx(self):
#         self.obs[1][2][1] = 888
#         return self.obs

# obs = np.zeros((4, 5, 10))
# share_obs = obs.reshape(4, -1)
# share_obs = np.expand_dims(share_obs, 1).repeat(5, axis=1)
# print()
# musv = musv_test()
# obs = musv.obs
# obs = np.clip(obs,-10,10)
# a = np.zeros((10,1))
# b = np.expand_dims(a,1).repeat(8,axis=1)
# arr = np.zeros((8,5),dtype=bool)
# share_obs = np.concatenate(obs, axis=-1)
# share_obs = np.expand_dims(share_obs, 1).repeat(5, axis=1)

# aver_episode_rewards1 = np.loadtxt('/home/lyy/Desktop/r_2_raw.txt')
# std1 = np.loadtxt('/home/lyy/Desktop/s_2_raw.txt')
# aver_episode_rewards2 = np.loadtxt('/home/lyy/Desktop/r_2_dis.txt')
# std2 = np.loadtxt('/home/lyy/Desktop/s_2_dis.txt')
arr1 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward1_raw.txt')
arr2 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward2_raw.txt')
arr3 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward3_raw.txt')
arr4 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward4_raw.txt')

ard1 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward1_dis.txt')
ard2 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward2_dis.txt')
ard3 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward3_dis.txt')
ard4 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward4_dis.txt')
ard5 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward5_dis.txt')
ard6 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward6_dis.txt')
ard7 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward7_dis.txt')
ard8 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward8_dis.txt')
ard9 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward9_dis.txt')
ard10 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/reward10_dis.txt')

sequential1 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/sequential1.txt')
sequential1_std = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/std/sequential1_std.txt')
sequential2 = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/sequential2.txt')
sequential2_std = np.loadtxt('/home/lyy/Desktop/happo_imgs/shadow_plot_muitl_run/std/sequential2_std.txt')

aver_episode_rewards1 = np.mean([arr1,arr2,arr3,arr4], axis=0)
std1 = np.std([arr1,arr2,arr3,arr4], axis=0)
aver_episode_rewards2 = np.mean([ard1,ard2,ard3,ard4,ard5,ard6,ard7,ard8,ard9,ard10], axis=0)
std2 = np.std([ard1,ard2,ard3,ard4,ard5,ard6,ard7,ard8,ard9,ard10], axis=0)
aver_episode_rewards3 = np.mean([sequential1, sequential2], axis=0)
std3 = np.std([sequential1, sequential2], axis=0)

x_axis = np.linspace(0, len(aver_episode_rewards1) * 200, num=len(aver_episode_rewards1), endpoint=False)
fig, ax = plt.subplots()
# 1
ax.plot(x_axis, aver_episode_rewards1, label="OHAPPO")
ax.fill_between(x_axis, aver_episode_rewards1 - std1, aver_episode_rewards1 + std1, alpha=0.33)
# 2
ax.plot(x_axis, aver_episode_rewards2, label="HAPPO")
ax.fill_between(x_axis, aver_episode_rewards2 - std2, aver_episode_rewards2 + std2, alpha=0.33)
# 3
ax.plot(x_axis, aver_episode_rewards3, label="Sequential")
ax.fill_between(x_axis, aver_episode_rewards3 - std3, aver_episode_rewards3 + std3, alpha=0.33)
# ax.plot(x_axis, sequential1, label="Sequential")
# ax.fill_between(x_axis, sequential1 - sequential1_std, sequential1 + sequential1_std, alpha=0.33)

ax.set_xlabel("Env total steps")
ax.set_ylabel("Average episode rewards")
ax.legend()
# ax.set_ylabel("Average episode rewards", fontproperties=font)
# ax.set_title("4P-1E")


# plt.savefig("/home/lyy/Desktop/happo_imgs/vs1.pdf", bbox_inches="tight")
plt.savefig("/home/lyy/Desktop/reward_.pdf", bbox_inches="tight")
plt.show()

