import argparse


def get_config():
    """
    Env parameters:
    """
    parser = argparse.ArgumentParser(description='ENVArgs',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--Env_name", type=str, default='USV')
    parser.add_argument("--n_agents", type=int, default=1)
    parser.add_argument("--u_dim", type=int, default=2)
    parser.add_argument("--x_dim", type=int, default=4)
    parser.add_argument("--pxy_dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_rollout_threads", type=int, default=8)
    parser.add_argument("--time_interval", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2, help="Random seed for numpy/torch")
    parser.add_argument("--seed_specify", action="store_true",
                        default=False, help="Random or specify seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false',
                        default=True, help="by default True, will use GPU to train; or else will use CPU;")

    return parser
