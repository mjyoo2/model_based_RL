import warnings
import os
warnings.filterwarnings('ignore')

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from buffer import Buffer
from asyn_MB import AsynMB
from callback import MBCallback

import gym

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ =='__main__':
    real_env = gym.make('LunarLanderContinuous-v2')
    env_data = {'observation_space': real_env.observation_space, 'action_space': real_env.action_space}

    replay_buffer = Buffer(max_len=50000)
    MB_env = SubprocVecEnv([lambda: AsynMB(env_data=env_data, replay_buffer=replay_buffer, name='1', n_steps=256),
                            # lambda: AsynMB(env_data=env_data, replay_buffer=replay_buffer, name='2', n_steps=256),
                            # lambda: AsynMB(env_data=env_data, replay_buffer=replay_buffer, name='3', n_steps=256),
                            lambda: AsynMB(env_data=env_data, replay_buffer=replay_buffer, name='4', n_steps=256)])

    agent = PPO2(MlpPolicy, MB_env, verbose=1, learning_rate=1e-4, tensorboard_log='./mb_sub_tensorboard')
    agent.learn(total_timesteps=100000000, log_interval=10, callback=MBCallback())