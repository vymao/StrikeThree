import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict, Tuple


# Guide: https://discuss.ray.io/t/log-or-record-custom-env-data-via-rllib/4674/2
class RewardLoggerCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        episode.user_data = {
            'ForwardRew': [],
            'HealthyRew': [],
            'ProxRew': [],
            'ReleaseRew': [],
            'StrikeRew': [],
            'TimeRew': [],
            'ControlCost': [],
            'WindupCosat': [],
            'ball_in_hand': [],
            'last_hand_action': []
        }
        episode.has_ball = True

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Running metrics -> keep all values
        # Final metrics -> only keep the current value
        info = episode.last_info_for()
        for k in episode.user_data.keys(): 
            episode.user_data[k].append(info[k])

        #if not info['ball_in_hand'] and episode.has_ball: 
        #    episode.user_data['release_step'] = episode.total_agent_steps
        #    episode.has_ball = False
            

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        for name, value in episode.user_data.items():
            #episode.custom_metrics[name + "_avg"] = np.mean(value)
            episode.custom_metrics[name] = np.sum(value)
        
        episode.hist_data['ball_in_hand'] = episode.user_data['ball_in_hand']
        episode.hist_data['last_hand_action'] = episode.user_data['last_hand_action']