import warnings

warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
import gym

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    model_1 = PPO2(MlpPolicy, env, verbose=1, policy_kwargs={'layers': [2], 'act_fun': tf.nn.relu})
    model_2 = PPO2(MlpPolicy, env, verbose=1, policy_kwargs={'layers': [2], 'act_fun': tf.nn.relu})
    print(model_1.get_parameters())
    print(model_2.get_parameters())
    model_2.load_parameters(model_1.get_parameters())
    print(model_2.get_parameters())