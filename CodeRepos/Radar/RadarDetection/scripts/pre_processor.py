import setproctitle
import numpy as np
from pathlib import Path
import torch
from RadarDetection.configs.config import get_config
import sys
import os

sys_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(sys_path)


def parse_args(args, parser):
    parser.add_argument("--use_single_network", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]
    return all_args


def PreProcess(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed = all_args.specify_id
    else:
        all_args.seed = np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)
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
                       0] + "/results") / all_args.user_name / all_args.module_name / all_args.algorithm_name / str(
        all_args.seed)
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
    curr_run = 'run0' if all_args.specify_single_run else curr_run
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.module_name) + "-" + str(all_args.algorithm_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.use_csv_label is False:
        all_args.output_dim = 10  # level 2, otherwise level 1, target 10

    configs = {
        "all_args": all_args,
        "device": device,
        "run_dir": run_dir,
        "save_name": "mssf_net_all.pth"
    }
    return configs

