import strikethree
from pprint import pprint
import gym
import numpy as np

def main():
    p = gym.make('Pitcher-v1')
    print(len(p.get_data()))
    print(np.concatenate([p.action_space.sample(), np.array([np.random.uniform(-1.0, 1.0)])]))

    print(type(p.spec))

if __name__ == "__main__":
    main()
