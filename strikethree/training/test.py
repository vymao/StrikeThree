import time
import joblib
import gym
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import strikethree


def load_policy(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    pytsave_path = osp.join(fpath, 'pyt_save')
    # Each file in this folder has naming convention 'modelXX.pt', where
    # 'XX' is either an integer or empty string. Empty string case
    # corresponds to len(x)==8, hence that case is excluded.
    saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

    itr = '%d'%max(saves) if len(saves) > 0 else ''

    get_action = load_pytorch_policy(fpath, itr, deterministic)

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render_info=None):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset()[0], 0, False, 0, 0, 0
    while n < num_episodes:
        if render_info['render'] and render_info['mode'] != 'human':
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _, info = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset()[0], 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--env', type=str, default='Pitcher-v1')
    parser.add_argument('--render_mode', type=str, default='human')
    args = parser.parse_args()
    
    render_info = {}
    render_info['render'] = not(args.norender)
    render_info['mode'] = args.render_mode

    if not args.norender: 
        env = gym.make(args.env, render_mode=args.render_mode)
    else: 
        env = gym.make(args.env)
    get_action = load_policy(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, render_info)
    #env.close()
