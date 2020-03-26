import warnings

warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
import gym

from asyn_MB import AsynMB
from stable_baselines.common.vec_env import SubprocVecEnv
from multiprocessing import Process
from callback import CustomCallback
from buffer import Buffer, data_pipe
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from learner import MBRL, RealAgent


def wrapper(func):
    func()


if __name__ == '__main__':
    replay_buffer = Buffer(max_len=50000)
    data = data_pipe({})
    data.put_data('MB_learning_start', False)

    real_env = gym.make('CartPole-v1')
    algo = lambda env: PPO2(MlpPolicy, env, verbose=1, tensorboard_log='C:/model/model_based_RL/',
                            policy_kwargs={'layers': [256, 256], 'act_fun': tf.nn.relu})

    real_model = RealAgent(algo=algo, real_env=real_env, replay_buffer=replay_buffer, data_pipe=data, verbose=1)
    real_model.learn(1000)

    MB_env = SubprocVecEnv([lambda: AsynMB(env_data=data.get_data('env_data'), replay_buffer=replay_buffer, n_steps=1000) for _ in range(1)])

    Callback = CustomCallback(data_pipe=data, get_data=real_model.learn, verbose=0)
    MB_model = MBRL(algo=algo, env=MB_env, replay_buffer=replay_buffer, data_pipe=data, callback=Callback)


    MB_model.learn(10000000)

    # real_proc = Process(target=wrapper, args=(real_model,))
    # real_proc.start()
    # MB_proc = Process(target=wrapper, args=(MB_model, ))
    # MB_proc.start()
    # MB_proc.join()
    # real_proc.join()