#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append("../")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from configs.config import get_config
from envs.musv.MUSV_liner import MusvLinerEnv
from runners.separated.musv_runner import MusvRunner as Runner

def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='Hopper-v2', help="Which mujoco task to run on")
    parser.add_argument('--agent_conf', type=str, default='3x1')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--use_single_network", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.runing_id
    else:
        all_args.seed=np.random.randint(1000,10000)
    print("seed is :",all_args.seed)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name / str(all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = MusvLinerEnv(all_args)

    eval_envs = None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()

if __name__ == "__main__":
    # argv = ['--env_name', 'musv', '--algorithm_name', 'happo', '--experiment_name', 'mlp', '--scenario',
    #         'Ant-v2', '--agent_conf', '2x4', '--agent_obsk', '2', '--lr', '5e-6', '--critic_lr', '5e-3',
    #         '--std_x_coef', '1', '--std_y_coef', '5e-1', '--running_id', '1', '--n_training_threads', '8',
    #         '--n_rollout_threads', '4', '--num_mini_batch', '40', '--episode_length', '1000',
    #         '--num_env_steps', '10000000', '--ppo_epoch', '5', '--kl_threshold', '1e-4',
    #         '--use_value_active_masks', '--use_eval', '--add_center_xy', '--use_state_agent', '--share_policy']
    argv = ['--env_name', 'musv', '--algorithm_name', 'happo', '--experiment_name', 'mlp', '--scenario',
            'Ant-v2', '--agent_conf', '2x4', '--agent_obsk', '2', '--lr', '5e-6', '--critic_lr', '5e-3',
            '--std_x_coef', '1', '--std_y_coef', '5e-1', '--runing_id', '1', '--n_training_threads', '8',
            '--n_rollout_threads', '8', '--num_mini_batch', '40', '--episode_length', '230',
            '--num_env_steps', '2000000', '--ppo_epoch', '5', '--kl_threshold', '1e-4',
            '--use_value_active_masks', '--add_center_xy', '--use_state_agent', '--share_policy']
    # '--model_dir', '/home/lyy/Desktop/HAPPO-HATRPO/scripts/results/musv/happo/mlp/5/run1/models'
    # '--model_dir', '/home/lyy/Desktop/HAPPO-HATRPO/scripts/results/musv/happo/mlp/2/run1/models'
    # '--seed_specify'
    main(argv)
