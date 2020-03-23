from asyn_MB import AsynMB
from buffer import Buffer

import copy

class MB_learner(object):
    def __init__(self, algo, real_env, MB_env):
        self.real_env = real_env
        self.real_agent = algo()
        self.MB_agent = algo()
        self.replay_buffer = Buffer(max_len=50000)

    def learn(self, total_timesteps):
        self.MB_agent.learn()

    def get_data(self):
        while True:
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
