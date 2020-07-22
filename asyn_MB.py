import numpy as np
import gym
import time
import os
from buffer import Buffer
from network import Network
from config import *

class AsynMB(gym.Env):
    def __init__(self, env_data, n_steps, socket_info, name, verbose=1):
        self.replay_buffer = Buffer(MAX_BUFFER_SIZE, socket_info)
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.state = None
        self.timesteps = 0
        self.verbose = verbose
        self.n_steps = n_steps

        if not os.path.isdir('./weights'.format(name)):
            os.mkdir('./weights'.format(name))

        if not os.path.isdir('./weights/{}'.format(name)):
            os.mkdir('./weights/{}'.format(name))

        action_shape = self.action_space.shape[0]
        state_shape = self.observation_space.shape[0]
        next_state_shape = self.observation_space.shape[0]
        self.reward_network = Network(layer_structure=MB_LAYER_STRUCTURE, action_shape=action_shape, state_shape=state_shape,
                                      output_shape=1, name='{}/reward_network'.format(name), last_layer='tanh')
        self.next_state_network = Network(layer_structure=MB_LAYER_STRUCTURE, action_shape=action_shape, state_shape=state_shape,
                                          output_shape=next_state_shape, name='{}/next_state_network'.format(name))

    def train_network(self, train):
        self.reward_network.reinit()
        self.next_state_network.reinit()
        state_data, action_data, next_state_data, reward_data = self.replay_buffer.get_dataset()
        self.replay_buffer.read_lock = False
        self.reward_network.train(state_data, action_data, reward_data, training_epochs=train)
        self.next_state_network.train(state_data, action_data, next_state_data, training_epochs=train)

    def step(self, action):
        self.timesteps += 1
        done = False
        reward = self.reward_network.predict(self.state, action)
        self.state = self.next_state_network.predict(self.state, action)
        for value in self.state.flatten():
            if np.isnan(value):
                print('NAN!')
        if np.isnan(reward):
            print('NAN!')
        info = {}
        if self.n_steps == self.timesteps:
            done = True
        return self.state.flatten(), reward[0][0], done, info

    def small_reset(self):
        self.state = self.init_state()
        self.timesteps = 0

        return self.state.flatten()

    def reset(self):
        while self.replay_buffer.length < MB_START:
            time.sleep(1)
        self.state = self.init_state()
        self.timesteps = 0
        return self.state.flatten()

    def render(self, mode='human'):
        pass

    def init_state(self):
        return self.replay_buffer.get_one()

