import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards, path, save_path_png):
    rewards = np.array(rewards)

    fig = plt.figure()
    plt.plot(rewards)

    # set label
    plt.title('Rewards Convergence')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    # open grid
    plt.grid(True)

    # save to pdf
    fig.savefig(path, format='pdf')
    fig.savefig(save_path_png)
    plt.close(fig)
