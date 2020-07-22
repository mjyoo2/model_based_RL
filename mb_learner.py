import warnings
import os
import gym
import socket
import pickle as pkl

warnings.filterwarnings('ignore')

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from asyn_MB import AsynMB
from callback import MBCallback
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ =='__main__':
    if environment is None:
        env_data = env_info
    else:
        env = environment()
        env_data = {'observation_space': env.observation_space, 'action_space': env.action_space}

    if not os.path.isdir('./mb/'):
        os.mkdir('./mb/')

    env_list = [lambda: AsynMB(env_data=env_data, name='0', n_steps=128, socket_info=model_env_info[0]),
                lambda: AsynMB(env_data=env_data, name='1', n_steps=128, socket_info=model_env_info[1]),
                lambda: AsynMB(env_data=env_data, name='2', n_steps=128, socket_info=model_env_info[2]),
                lambda: AsynMB(env_data=env_data, name='3', n_steps=128, socket_info=model_env_info[3])]

    MB_env = SubprocVecEnv(env_list)
    mb_callback = MBCallback(MBRL_info=MBRL_info, real_RL_info=real_env_info)

    mb_agent = PPO2(MlpPolicy, MB_env, verbose=1, learning_rate=1e-4, tensorboard_log='./mb/')
    mb_agent.learn(total_timesteps=MB_TIMESTEPS, log_interval=10, callback=mb_callback)