import tensorflow.compat.v1 as tf
import numpy as np
import gym
import os
from network import Network, DoneNetwork


def decode_reward(reward):
    return -100 * np.log(2*reward/(reward+1))

class AsynMB(gym.Env):
    def __init__(self, env_data, replay_buffer, n_steps, name, verbose=1):
        self.replay_buffer = replay_buffer
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.state = None
        self.timesteps = 0
        self.verbose = verbose
        self.n_steps = n_steps

        if not os.path.isdir('./weights/{}'.format(name)):
            os.mkdir('./weights/{}'.format(name))

        action_shape = self.action_space.shape[0]
        state_shape = self.observation_space.shape[0]
        next_state_shape = self.observation_space.shape[0]
        self.reward_network = Network(layer_structure=[256, 256], action_shape=action_shape, state_shape=state_shape,
                                      output_shape=1, name='{}/reward_network'.format(name))
        self.next_state_network = Network(layer_structure=[256, 256], action_shape=action_shape, state_shape=state_shape,
                                          output_shape=next_state_shape, name='{}/next_state_network'.format(name))

    def train_network(self, train):
        self.replay_buffer.load()
        self.reward_network.reinit()
        self.next_state_network.reinit()
        state_data, action_data, next_state_data, reward_data = self.replay_buffer.get_dataset()
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
        self.state = self.init_state()
        self.timesteps = 0

        return self.state.flatten()

    def render(self, mode='human'):
        pass

    def init_state(self):
        return self.replay_buffer.get_one()

