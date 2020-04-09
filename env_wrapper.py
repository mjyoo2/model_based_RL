import gym
import numpy as np
import time

class wrap_env(gym.Env):
    def __init__(self, env, buffer, delay):
        self.env = env
        self.buffer = buffer
        self.delay = delay
        self.state = None

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        time.sleep(self.delay)
        state, reward, done, info = self.env.step(action)
        reward = 2/(1+np.exp(-reward/10))-1
        self.buffer.add({'state': self.state, 'action': action, 'next_state': state, 'reward': [reward], 'done': done})
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.state = state
        return state

    def render(self, mode='human'):
        self.env.render()

    def buffer_save(self):
        self.buffer.save()