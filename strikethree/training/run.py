import strikethree
from strikethree.training.td3.model import td3
from strikethree.training.td3.core import MLPActorCritic
from strikethree.envs.mujoco.pitcher import PitcherEnv
from strikethree.training.utils import RewardLoggerCallback
import gymnasium as gym
from spinup.utils.run_utils import setup_logger_kwargs
import os
import ray
from ray.rllib.algorithms import td3 as ray_td3
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print


def ray_env_creator(env_config):
    return PitcherEnv()  # return an env instance

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
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_gpu', type=int, default=0)
    args = parser.parse_args()

    if args.log_dir == '.': 
        data_dir = os.path.join(os.getcwd(), 'data')
    else: 
        data_dir = os.path.join(args.log_dir, 'data')

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir)

    #if args.num_cores > 1:
    ray.init()

    register_env("pitcher", ray_env_creator)
    algo = ray_td3.TD3Config().environment(env="pitcher").training(
        actor_hiddens=[args.hid],
        critic_hiddens=[args.hid]
    ).resources(
        num_gpus=args.num_gpu,
        num_learner_workers=args.num_workers,
    ).callbacks(
        callbacks_class=RewardLoggerCallback
    ).build()
    for i in range(1, args.epochs + 1):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    """
    else: 
        if args.alg == 'td3': 
            td3(lambda : gym.make(args.env), actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch, logger_kwargs=logger_kwargs)
    """