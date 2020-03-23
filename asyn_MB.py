import tensorflow as tf
import gym

from network import Network

class AsynMB(gym.Env):
    def __init__(self, env_data, replay_buffer, update_interval, n_pretrain):
        self.replay_buffer = replay_buffer
        self.update_interval = update_interval
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.state = None

        input_shape = self.observation_space.shape[0] + self.action_space.shape[0]
        next_state_shape = self.observation_space.shape[0]
        self.reward_network = Network(layer_structure=[256, 256 ,256, 256], input_shape=input_shape, output_shape=1, mode='reward')
        self.next_state_network = Network(layer_structure=[256, 256, 256, 256], input_shape=input_shape, output_shape=next_state_shape)
        self.pretrain_network(n_pretrain)

    def pretrain_network(self, n_pretrain):
        for i in range(n_pretrain):
            replay_batch = self.replay_buffer.get_batch()
            self.reward_network.train(replay_batch)
            self.next_state_network.train(replay_batch)

    def step(self, action):
        reward = self.reward_network.predict(self.state, action)
        self.state = self.next_state_network.predict(self.state, action)

        return self.state, reward, done, info

    def reset(self):
        self.state =

    def render(self):