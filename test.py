import warnings
import os
warnings.filterwarnings('ignore')


from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from env_wrapper import wrap_env
from stable_baselines.common.vec_env import SubprocVecEnv

import gym


if __name__ == "__main__":
    env = SubprocVecEnv([ lambda:wrap_env(gym.make('LunarLanderContinuous-v2'))])
    agent = PPO2(MlpPolicy, env, n_steps=32, verbose=1, tensorboard_log='./test_tensorboard')
    agent.learn(total_timesteps=5000000, log_interval=10)