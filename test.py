import warnings
import os
warnings.filterwarnings('ignore')


from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from callback import CustomCallback
from env_wrapper import wrap_env
from buffer import Buffer

import gym


if __name__ == "__main__":
    env = wrap_env(gym.make('LunarLanderContinuous-v2'), Buffer(50000), delay=0)
    agent = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./test_tensorboard')
    agent.learn(total_timesteps=5000000, log_interval=10)