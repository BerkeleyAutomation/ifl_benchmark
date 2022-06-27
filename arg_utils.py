import argparse
'''
Util to compile command line arguments for core script to run experiments
for the IFL Benchmark (main.py)
'''


def get_parser():
    # Global Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                    description='IFL Benchmark Arguments')
    parser.add_argument('--env_name', default='Humanoid',
                        help='Choice of environment. Choices: [Humanoid, Anymal, AllegroHand]')
    parser.add_argument('--logdir', default='logs',
                        help='log directory')
    parser.add_argument('--logdir_suffix', default='',
                        help='log directory suffix')
    parser.add_argument('--cuda', action='store_true',
                        help='run on CUDA')
    parser.add_argument('--cnn', action='store_true',
                        help='visual observations')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 123456)')

    # Parallel experiment
    parser.add_argument('--agent', type=str, default="IL", help="Type of parallel agent; options are in iflb/agents/__init__.py")
    parser.add_argument('--supervisor', type=str, default="GymRL", help="Type of parallel supervisor; options are in iflb/supervisors/__init__.py")
    parser.add_argument('--allocation', type=str, default="CUR", help="Type of allocation strategy; options are in iflb/allocations/__init__.py")
    parser.add_argument('--num_envs', type=int, default=10, help="number of robots")
    parser.add_argument('--num_humans', type=int, default=1, help="number of humans")
    parser.add_argument('--min_int_time', type=int, default=1, help="minimum intervention time")
    parser.add_argument('--hard_reset_time', type=int, default=0, help="number of steps waited before reset completes")
    parser.add_argument('--log_freq', type=int, default=100, help="log frequency")
    parser.add_argument('--vec_env', action="store_true", help="whether or not to use vectorized Isaac Gym environments")
    parser.add_argument('--render', action="store_true", help="whether or not to render an Isaac Gym env")
    parser.add_argument('--resume', default='', help='if resuming a previous run, this is the logdir from which to load info')
    parser.add_argument('--num_steps', type=int, default=10000, help='maximum number of parallel timesteps (default: 10000)')
    parser.add_argument('--noise', type=float, default=0.0, help='standard deviation for independent zero-mean gaussian noise injection into actions')
    parser.add_argument('--discrete', action="store_true", help="if true, use a discrete action space")

    return parser
