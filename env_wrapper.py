import gym
import numpy as np
import time
import socket
import pickle as pkl

class wrap_env(gym.Env):
    def __init__(self, env, delay, port, model_env_info):
        self.env = env
        self.delay = delay
        self.state = None
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', port))
        self.model_env_info = model_env_info

    def step(self, action):
        time.sleep(self.delay)
        state, reward, done, info = self.env.step(action)
        reward = np.clip(reward, a_max=10, a_min=-10)
        reward = 2 / (1 + np.exp(-reward/10)) - 1
        if np.isnan(self.state.any()):
            print("state is NAN!")
        if np.isnan(action.any()):
            print("action is NAN!")
        if np.isnan(reward):
            print("reward is NAN!")
        data = {'state': self.state, 'action': action, 'next_state': state, 'reward': [reward], 'done': done}
        for socket_info in self.model_env_info:
            self.socket.sendto(pkl.dumps(data), socket_info)
        self.state = state
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.state = state
        return state

    def render(self, mode='human'):
        self.env.render()
