import tensorflow as tf
import gym

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from learner import MB_learner


if __name__ == '__main__':
    algo = lambda env: PPO2(MlpPolicy, env, verbose=1, tensorboard_log='C:/model/model_based_RL/',
                            policy_kwargs={'layers': [256, 256, 256, 256], 'act_fun': tf.nn.relu})
    real_env = gym.make('Pendulum-v0')
    model = MB_learner(algo=algo, real_env=real_env, n_model=12)
    model.learn(total_timesteps=1000000)