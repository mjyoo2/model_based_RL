import warnings

warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
import gym

from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = SAC(LnMlpPolicy, env, verbose=1, tensorboard_log='./test_tensorboard/')
    agent.learn(total_timesteps=1000000, log_interval=10)
