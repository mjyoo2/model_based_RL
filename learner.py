import time
import numpy as np
import pickle as pkl

from asyn_MB import AsynMB
from stable_baselines.common.vec_env import SubprocVecEnv

class MBRL(object):
    def __init__(self, algo, env, replay_buffer, data_pipe, callback):
        self.replay_buffer = replay_buffer
        self.data_pipe = data_pipe

        self.callback = callback
        self.MB_agent = algo
        self.MB_env = env

        self.data_pipe.put_data('learning_done', False)

    def learn(self, total_timesteps):
        # while self.replay_buffer.length < 10000:
        #     time.sleep(1)

        self.MB_agent = self.MB_agent(self.MB_env)
        self.data_pipe.put_data('MB_agent_parameters', self.MB_agent.get_parameters())
        self.data_pipe.put_data('MB_learing_start', True)

        self.MB_agent.learn(total_timesteps, log_interval=32, callback=self.callback)
        self.data_pipe.put_data('learning_done', True)


class RealAgent(object):
    def __init__(self, algo, real_env, replay_buffer, data_pipe, verbose):
        self.replay_buffer = replay_buffer
        self.data_pipe = data_pipe
        self.verbose = 1

        self.real_env = real_env
        self.real_agent = algo(real_env)
        self.first_learn = True

        env_data = {'observation_space': real_env.observation_space, 'action_space': real_env.action_space}
        self.data_pipe.put_data('env_data', env_data)

    def learn(self, steps):
        if not self.first_learn:
            self.get_model()
            self.first_learn = False
        num_timesteps = 0
        epi_reward_list = []
        while True:
            reward_list = []
            done = False
            state = self.real_env.reset()
            while not done:
                num_timesteps += 1
                action, _ = self.real_agent.predict(state)
                next_state, reward, done, info = self.real_env.step(action)
                reward_list.append(reward)
                self.replay_buffer.add({'state': state, 'action': action, 'next_state': next_state, 'reward': [2/(1+np.exp(-reward/10))-1], 'done': done})
                state = next_state
            if self.verbose == 1:
                epi_reward_list.append(np.sum(reward_list))
            if num_timesteps > steps:
                break
        if self.verbose == 1:
            print('episode_reward: ', np.mean(epi_reward_list))
        self.replay_buffer.save()

    def get_model(self):
        with open('./network/parameters.pkl', 'rb') as f:
            parameters = pkl.load(f)
        self.real_agent.load_parameters(parameters)