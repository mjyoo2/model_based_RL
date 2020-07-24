import warnings
import os

warnings.filterwarnings('ignore')

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from asyn_MB import AsynMB
from callback import MBCallback
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def make_lambda_env(i, model_env_info):
    return lambda: AsynMB(env_data=env_info, name=str(i), n_steps=MB_EPI_LENGTH, socket_info=model_env_info[i])


if __name__ =='__main__':
    if not os.path.isdir('./mb/'):
        os.mkdir('./mb/')

    model_env_info = []
    for i in range(MB_ENV_NUM):
        model_env_info.append((model_env_ip, port + i))

    env_list = []
    for i in range(MB_ENV_NUM):
        env_list.append(make_lambda_env(i, model_env_info))

    MB_env = SubprocVecEnv(env_list)
    mb_callback = MBCallback(MBRL_info=MBRL_info, real_RL_info=real_env_info)

    mb_agent = PPO2(MlpPolicy, MB_env, verbose=1, learning_rate=1e-4, tensorboard_log='./mb/')
    mb_agent.learn(total_timesteps=MB_TIMESTEPS, log_interval=10, callback=mb_callback)