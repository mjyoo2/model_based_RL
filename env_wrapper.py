import numpy as np
import time
import pickle as pkl
import zmq

from config import *

class wrap_env(gym.Env):
    def __init__(self, env):
        self.env = env
        self.delay = MAIN_ENV_DELAY
        self.state = None
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        ctx = zmq.Context()
        self.send_sock = ctx.socket(zmq.PUB)
        self.send_sock.connect('tcp://{}:{}'.format(model_env_ip, port+31))

    def step(self, action):
        time.sleep(self.delay)
        state, reward, done, info = self.env.step(action)
        # reward shaping
        # reward = 2 / (1 + np.exp(-np.clip(reward, a_max=10, a_min=-10)/10)) - 1
        if np.isnan(self.state.any()):
            print("state is NAN!")
        if np.isnan(action.any()):
            print("action is NAN!")
        if np.isnan(reward):
            print("reward is NAN!")
        data = {'state': self.state, 'action': action, 'next_state': state - self.state, 'reward': [reward], 'done': done}
        self.send_sock.send(pkl.dumps(data))
        self.state = state
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.state = state
        return state

    def render(self, mode='human'):
        self.env.render()
