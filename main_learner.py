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
    main_callback = MainLearnerCallback(MBRL_info=MBRL_info, real_RL_info=real_env_info)
    env = SubprocVecEnv([lambda: wrap_env(gym.make('LunarLanderContinuous-v2'), delay=0.01,
                                          port=10080, model_env_info=model_env_info)])
    main_agent = PPO2(MlpPolicy, env, learning_rate=0, verbose=1, tensorboard_log='./main/')
    main_agent.learn(total_timesteps=10000000, log_interval=10, callback=main_callback)