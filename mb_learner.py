import warnings
import os
import gym

warnings.filterwarnings('ignore')

from env_wrapper import wrap_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from multiprocessing import Process
from asyn_MB import AsynMB
from callback import MBCallback, MainLearnerCallback

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

real_env_info = {'socket_info' : ('127.0.0.1', 10010)}
MBRL_info = {'socket_info' : ('127.0.0.1', 10020)}
model_env_info = [('127.0.0.1', 10001), ('127.0.0.1', 10002),
                  ('127.0.0.1', 10003), ('127.0.0.1', 10004)]

if __name__ =='__main__':
    real_env = gym.make('LunarLanderContinuous-v2')
    env_data = {'observation_space': real_env.observation_space, 'action_space': real_env.action_space}

    env_list = [lambda: AsynMB(env_data=env_data, name='0', n_steps=256, socket_info=model_env_info[0]),
                lambda: AsynMB(env_data=env_data, name='1', n_steps=256, socket_info=model_env_info[1]),
                lambda: AsynMB(env_data=env_data, name='2', n_steps=256, socket_info=model_env_info[2]),
                lambda: AsynMB(env_data=env_data, name='3', n_steps=256, socket_info=model_env_info[3])]
    MB_env = SubprocVecEnv(env_list)
    mb_callback = MBCallback(MBRL_info=MBRL_info, real_RL_info=real_env_info)

    mb_agent = PPO2(MlpPolicy, MB_env, verbose=1, learning_rate=1e-4, tensorboard_log='./mb/')
    mb_agent.learn(total_timesteps=100000000, log_interval=10, callback=mb_callback)