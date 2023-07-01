from gym.envs.registration import make, register

# Mujoco
# ----------------------------------------

register(
    id="Pitcher-v1",
    entry_point="strikethree.envs.mujoco:PitcherEnv",
    max_episode_steps=1000,
)