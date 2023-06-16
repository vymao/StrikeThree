from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points
_load_env_plugins()

# Mujoco
# ----------------------------------------

register(
    id="Pitcher",
    entry_point="gym.envs.mujoco.pitcher:PitcherEnv",
    max_episode_steps=1000,
)