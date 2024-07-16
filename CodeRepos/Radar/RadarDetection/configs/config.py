import argparse
import warnings
warnings.filterwarnings('ignore')


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.
    """

    parser = argparse.ArgumentParser(description='onpolicy_algorithm',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--user_name",
                        type=str,
                        default='TargetDetecter',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")

    parser.add_argument("--module_name",
                        type=str,
                        default='RadarDetection',
                        help="specify the name of environment")

    parser.add_argument("--algorithm_name",
                        type=str,
                        default='MSSFNet',
                        choices=["MSSFNet","MssfNet-pro"])

    parser.add_argument("--experiment_name",
                        type=str,
                        default="detect",
                        help="an identifier to distinguish different experiment.")

    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Random seed for numpy/torch")

    parser.add_argument("--seed_specify",
                        action="store_true",
                        default=False,
                        help="Random or specify seed for numpy/torch")

    parser.add_argument("--specify_id",
                        type=int,
                        default=1,
                        help="the runing index of experiment")

    parser.add_argument("--specify_single_run",
                        action="store_true",
                        default=False,
                        help="if only use result path with '/run0' instead of '/run{i}' ")

    parser.add_argument("--cuda",
                        action="store_true",
                        default=False,
                        help="by default True, will use GPU to train; or else will use CPU;")

    parser.add_argument("--cuda_deterministic",
                        action='store_false',
                        default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")

    parser.add_argument("--n_training_threads",
                        type=int,
                        default=1,
                        help="Number of torch threads for training")

    parser.add_argument("--n_rollout_threads",
                        type=int,
                        default=32,
                        help="Number of parallel envs for training rollouts")

    parser.add_argument("--n_eval_rollout_threads",
                        type=int,
                        default=1,
                        help="Number of parallel envs for evaluating rollouts")

    parser.add_argument("--n_render_rollout_threads",
                        type=int,
                        default=1,
                        help="Number of parallel envs for rendering rollouts")

    parser.add_argument("--frame_len",
                        type=int,
                        default=1,
                        help="frame_len = 1 for default setting")

    parser.add_argument("--history_frame_len",
                        type=int,
                        default=1,
                        help="history_frame_len = 1 for default setting")

    parser.add_argument("--frame_buffer_capacity",
                        type=int,
                        default=10,
                        help="frame_buffer_capacity")

    # path param
    parser.add_argument("--src_data_path",
                        type=str,
                        default=None,
                        help="radar dataset's source data path")

    parser.add_argument("--tar_data_path",
                        type=str,
                        default=None,
                        help="radar dataset's labeled target data path")

    parser.add_argument("--tar_data_relative_path",
                        type=str,
                        default="/data/DataIn",
                        help="radar dataset's labeled target data path")

    parser.add_argument("--detect_num",
                        type=int,
                        default=10,
                        help="detect_num")

    # training param
    parser.add_argument("--Train",
                        action="store_true",
                        default=False,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--ContinueTrain",
                        action="store_true",
                        default=False,
                        help="If continue train")
    parser.add_argument("--RADAR_ON",
                        action="store_true",
                        default=False,
                        help="If use radar")

    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="batch_size")

    parser.add_argument("--channels",
                        type=int,
                        default=1,
                        help="channels")

    parser.add_argument("--weight",
                        type=int,
                        default=256,
                        help="weight")

    parser.add_argument("--height",
                        type=int,
                        default=256,
                        help="height")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="epochs")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-5,
                        help="learning_rate")

    # ConvLstm param
    parser.add_argument("--batch_first",
                        action="store_true",
                        default=False,
                        help="batch_first")

    parser.add_argument("--bias",
                        action="store_true",
                        default=False,
                        help="bias")

    parser.add_argument("--return_all_layers",
                        action="store_true",
                        default=False,
                        help="return_all_layers")

    parser.add_argument("--hist_frame_hid_dim1",
                        type=int,
                        default=32,
                        help="hist_frame_hid_dim1")

    parser.add_argument("--hist_frame_hid_dim2",
                        type=int,
                        default=32,
                        help="hist_frame_hid_dim2")

    parser.add_argument("--kernel_size",
                        type=int,
                        default=3,
                        help="kernel_size")

    parser.add_argument("--num_layers",
                        type=int,
                        default=3,
                        help="num_layers")

    # SwinTransformer param
    parser.add_argument("--sample_times",
                        type=int,
                        default=8,
                        help="sample_times")

    # DownSample param
    parser.add_argument("--down_sample_times",
                        type=int,
                        default=4,
                        help="down_sample_times")

    # MaxPool param
    parser.add_argument("--max_pool_times",
                        type=int,
                        default=2,
                        help="max_pool_times")

    # Conv2OneChannel param
    parser.add_argument("--frame_conv_2one_times",
                        type=int,
                        default=4,
                        help="conv_2one_times")

    parser.add_argument("--hist_frame_conv_2one_times",
                        type=int,
                        default=2,
                        help="hist_frame_conv_2one_times")

    # linear param
    parser.add_argument("--linear_dim2",
                        type=int,
                        default=128,
                        help="linear_dim2")

    parser.add_argument("--output_dim",
                        type=int,
                        default=20,
                        help="linear_dim3")

    # loss function param
    parser.add_argument("--ctype",
                        type=str,
                        default='MSE',
                        help="type of criterion")

    parser.add_argument("--RADAR",
                        type=list,
                        default=[],
                        help="use radar")

    parser.add_argument("--end",
                        type=bool,
                        default=False,
                        help="is end or not")

    parser.add_argument("--use_csv_label",
                        type=bool,
                        default=False,
                        help="use_csv_label")

    parser.add_argument("--combine_mode",
                        type=bool,
                        default=False,
                        help="combine_mode")

    parser.add_argument("--min_len",
                        type=int,
                        default=0,
                        help="min_len")

    parser.add_argument("--count",
                        type=int,
                        default=0,
                        help="count")

    parser.add_argument("--target_num",
                        type=int,
                        default=0,
                        help="count")

    parser.add_argument("--M",
                        type=int,
                        default=4,
                        help="count")

    parser.add_argument("--N",
                        type=int,
                        default=3,
                        help="count")

    parser.add_argument("--start_time",
                        type=float,
                        default=0,
                        help="count")

    return parser
