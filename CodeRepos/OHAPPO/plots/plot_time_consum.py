import numpy as np
import matplotlib.pyplot as plt


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

add = np.abs(np.min(aver_episode_rewards1)) + 300
aver_episode_rewards1 += add
aver_episode_rewards2 += add
aver_episode_rewards3 += add


idx1 = [[], [], []]
idx_std = [[], [], []]
value = [np.max(aver_episode_rewards1)*i for i in [0.25, 0.62, 0.8]]
idx1[0] = [np.abs(aver_episode_rewards1 - v).argmin() for v in value]
idx1[1] = [np.abs(aver_episode_rewards2 - v).argmin() for v in value]
idx1[2] = [np.abs(aver_episode_rewards3 - v).argmin() for v in value]
idx = np.array(idx1)
idx_std[0] = [std1[i] for i in idx[0]]
idx_std[1] = [std2[i] for i in idx[1]]
idx_std[2] = [std3[i] for i in idx[2]]

# idx = [list(i) for i in zip(*idx1)]
# 1
# 设置颜色和组标签
colors = ['b', '#ff7f0e', 'g']
group_labels = ['OHAPPO', 'HAPPO', 'OHAPPO$^-$']

# 创建柱状图
fig, ax = plt.subplots()

for i in range(3):
    ax.bar(np.arange(3) + i*0.25, idx[i]*0.9/60, width=0.2, color=colors[i], align='center',
           label=group_labels[i], alpha=0.6)
    # yerr=idx_std[i] / (add/100)

# 添加折线图
for i in range(3):
    ax.plot(np.arange(3) + i*0.25, idx[i]*0.9/60, color=colors[i], marker='D', linestyle='-.')

# 设置标题和标签
# ax.set_title('title')
ax.set_xlabel('Percentage of peak performance')
ax.set_ylabel('Time consumption (hour)')
ax.set_xticks(np.arange(3) + 0.25)
ax.set_xticklabels(['25%', '60%', '80%'])
ax.grid(True, which='both', color='gray', linewidth='0.5')
ax.legend()

# plt.savefig("/home/lyy/Desktop/happo_imgs/vs1.pdf", bbox_inches="tight")
plt.savefig("/home/lyy/Desktop/time_consum.pdf", bbox_inches="tight")
plt.show()
