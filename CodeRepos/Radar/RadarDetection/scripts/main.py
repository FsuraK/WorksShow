from pre_processor import PreProcess
from RadarDetection.runners.runner import Runner
from RadarDetection.algorithms.mssf_net import MSSFNet
from RadarTracking.mtt_main import track_main
from RadarTracking.mtt_main import aijpda_based_mtt
from RadarTracking.New_MN_Logic import TrackInitiation


argv = [
    '--user_name', 'TargetDetecter', '--module_name', 'RadarDetection', '--algorithm_name', 'MSSFNet',
    # '--Train',
    # '--ContinueTrain',
    # '--RADAR_ON',
    '--combine_mode', 'True',
    '--cuda', '--batch_first', '--bias',
    '--seed_specify', '--specify_id', '5',
    '--specify_single_run',
    '--epochs', '50000',
    '--channels', '1',
    '--batch_size', '1',
    '--history_frame_len', '1',
    # '--use_csv_label', 'True',
    '--n_training_threads', '1',
    '--n_rollout_threads', '32',
    '--n_eval_rollout_threads', '1',
    '--n_render_rollout_threads', '1',
    # '--cuda_deterministic',
]


if __name__ == "__main__":
    # --- Radar Target Detection Init
    configs = PreProcess(argv)
    config = configs['all_args']
    runner = Runner(configs)
    # --- Radar Target Tracking Init
    mtt = aijpda_based_mtt()
    tracker = TrackInitiation(threshold=100, M=config.M, N=config.N)

    runner.OnCreate(model=MSSFNet)
    if config.Train or (not config.combine_mode):
        runner.Execute()
    else:
        while not config.end:
            # --------------------- Radar Target Detection ---------------------
            runner.Execute()
            # --------------------- Radar Target Tracking ----------------------
            track_main(config, mtt, tracker)

    print("========================== End ==========================")
