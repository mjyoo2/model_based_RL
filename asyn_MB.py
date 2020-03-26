import tensorflow.compat.v1 as tf
import numpy as np
import gym
import time

from network import Network

class AsynMB(gym.Env):
    def __init__(self, env_data, n_steps, replay_buffer, verbose=1, n_pretrain=500):
        self.replay_buffer = replay_buffer
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.state = None
        self.n_steps = n_steps
        self.timesteps = 0
        self.verbose = verbose

        input_shape = self.observation_space.shape[0] + 1
        next_state_shape = self.observation_space.shape[0]
        self.reward_network = Network(layer_structure=[64, 64], input_shape=input_shape, output_shape=1)
        self.next_state_network = Network(layer_structure=[64, 64], input_shape=input_shape, output_shape=next_state_shape)
        self.train_network(n_pretrain)

    def train_network(self, n_pretrain):
        state_action_data, next_state_data, reward_data = self.replay_buffer.get_dataset()
        print(state_action_data[0])
        self.reward_network.train(state_action_data, reward_data, training_epochs=5)
        self.next_state_network.train(state_action_data, next_state_data, training_epochs=5)

    def step(self, action):
        self.timesteps += 1
        done = False
        state_action = np.concatenate([self.state.flatten(), action.flatten()])
        reward = self.reward_network.predict(state_action)
        self.state = self.next_state_network.predict(state_action)
        info = {}
        if self.timesteps == self.n_steps:
            done = True
        return self.state.flatten(), reward[0][0], done, info

    def reset(self):
        self.train_network(100)
        self.state = self.init_state()
        self.timesteps = 0

        return self.state.flatten()

    def render(self, mode='human'):
        pass

    def init_state(self):
        return self.replay_buffer.get_one()['state']