import strikethree
from trajectory_module import compute_trajectory
from pprint import pprint
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse

BALL_START_POS = [0, 0, 1.2]
BALL_START_VELO = [2, 40, 0]
MAX_DISTANCE = 18.29 # m

def run_and_plot_trajectory(): 
    time_elapsed, final_pos, final_velo, trajectory = compute_trajectory(BALL_START_VELO, BALL_START_POS, store_vals=True)
    print("Time elapsed (s): ", time_elapsed)
    print("Final position: ", final_pos[0], ',', final_pos[1], ',', final_pos[2])
    print("Final velocity: ", final_velo[0], ',', final_velo[1], ',', final_velo[2])

    fig, ax = plt.subplots()
    ax.scatter(trajectory[1], trajectory[2])
    #ax.set(xlim=(0, MAX_DISTANCE))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Height')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    ax.scatter(trajectory[0], trajectory[1], trajectory[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return trajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pitcher-v1')
    parser.add_argument('--max_traj_iter', type=int, default=1000)
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    p = gym.make(args.env)
    if args.plot: 
        run_and_plot_trajectory()

if __name__ == "__main__":
    main()
