import warnings
import os
warnings.filterwarnings('ignore')

from env_wrapper import wrap_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from callback import MainLearnerCallback
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def make_lambda_env(i):
    return lambda: wrap_env(environment(), env_port=i)


if __name__ =='__main__':
    if not os.path.isdir('./main'):
        os.mkdir('./main')

    env_list = []
    for i in range(MAIN_ENV_NUM):
        env_list.append(make_lambda_env(i))

    main_callback = MainLearnerCallback(MBRL_info=MBRL_info, real_RL_info=real_env_info)
    env = SubprocVecEnv(env_list)

    main_agent = PPO2(MlpPolicy, env, n_steps=32, verbose=1, tensorboard_log='./main/')
    main_agent.learn(total_timesteps=MAIN_TIMESTEPS, log_interval=10, callback=main_callback)
