import strikethree
from strikethree.training.td3.model import td3
from strikethree.training.td3.core import MLPActorCritic
import gym
from spinup.utils.run_utils import setup_logger_kwargs
import os


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pitcher-v1')
    parser.add_argument('--alg', type=str, default='td3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--log_dir', type=str, default='.')
    args = parser.parse_args()

    if args.log_dir == '.': 
        data_dir = os.path.join(os.getcwd(), 'data')
    else: 
        data_dir = os.path.join(args.log_dir, 'data')

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir)


    if args.alg == 'td3': 
        td3(lambda : gym.make(args.env), actor_critic=MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
            gamma=args.gamma, seed=args.seed, epochs=args.epochs,
            logger_kwargs=logger_kwargs)