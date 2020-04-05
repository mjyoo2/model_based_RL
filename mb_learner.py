from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy
from buffer import Buffer
from asyn_MB import AsynMB
from callback import MBCallback

import gym

if __name__ =='__main__':
    real_env = gym.make('LunarLanderContinuous-v2')
    env_data = {'observation_space': real_env.observation_space, 'action_space': real_env.action_space}

    replay_buffer = Buffer(max_len=50000)
    env = AsynMB(env_data = env_data, replay_buffer=replay_buffer, n_steps=256)

    agent = SAC(LnMlpPolicy, env, learning_rate=1e-4, verbose=1, tensorboard_log='./mb_sub_tensorboard')
    agent.learn(total_timesteps=10000000, log_interval=10, callback=MBCallback())