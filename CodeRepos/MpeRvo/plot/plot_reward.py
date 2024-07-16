import numpy as np
import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import os


def plot_reward():
  now_path = os.getcwd()
  target_path = os.path.abspath(os.path.join(now_path, "../")) + \
                "/theta_offset/data/reward.txt"
  save_path = os.path.abspath(os.path.join(now_path, "../")) + \
                "/theta_offset/imgs/reward.pdf"

  reward = np.loadtxt(target_path)
  x_axis = np.linspace(0, len(reward) * 200, num=len(reward), endpoint=False)
  fig, ax = plt.subplots()
  # 1
  ax.plot(x_axis, reward, label="reward")
  ax.set_xlabel("Environment total steps")
  ax.set_ylabel("Episode Reward")
  ax.legend()
  ax.set_title("BiLSTM-DDPG-LinearUSVEnv")
  plt.savefig(save_path, bbox_inches="tight")
