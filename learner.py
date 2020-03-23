from asyn_MB import AsynMB
from multiprocessing import Process, current_process
from buffer import Buffer
from stable_baselines.common.vec_env import SubprocVecEnv

import copy

class MB_learner(object):
    def __init__(self, algo, real_env, n_model):
        self.replay_buffer = Buffer(max_len=50000)
        self.real_env = real_env
        env_data = {'observation_space': self.real_env.observation_space, 'action_space': self.real_env.action_space}
        MB_env = SubprocVecEnv([lambda: AsynMB(env_data=env_data, replay_buffer=self.replay_buffer) for _ in range(n_model)])
        self.real_agent = algo(real_env)
        self.MB_agent = algo(MB_env)
        self.learning_done = False

    def learn(self, total_timesteps):
        data_process = Process(target=self.get_data, args=())
        data_process.start()
        MB_process = Process(target=self.MB_learn, args=(total_timesteps, ))
        MB_process.start()
        MB_process.join()
        self.learning_done = True
        data_process.join()

    def get_data(self):
        while not self.learning_done:
            self.get_model()
            done = False
            state = self.real_env.reset()
            while not done:
                action = self.real_agent.predict(state)
                next_state, reward, done, info = self.real_env.step(action)
                self.replay_buffer.add({'state': state, 'action': action, 'next_state': next_state, 'reward': reward})

    def get_model(self):
        self.real_agent = copy.deepcopy(self.MB_agent)
        self.real_agent.env = self.real_env

    def MB_learn(self, total_timesteps):
        self.MB_agent.learn(total_timesteps)