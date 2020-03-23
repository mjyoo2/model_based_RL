import tensorflow as tf
import gym

from network import Network

class AsynMB(gym.Env):
    def __init__(self, env_data, replay_buffer, n_pretrain=100):
        self.replay_buffer = replay_buffer
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.state = None
        self.done = False

        input_shape = self.observation_space.shape[0] + self.action_space.shape[0]
        next_state_shape = self.observation_space.shape[0]
        self.reward_network = Network(layer_structure=[256, 256 ,256, 256], input_shape=input_shape, output_shape=1, mode='reward')
        self.next_state_network = Network(layer_structure=[256, 256, 256, 256], input_shape=input_shape, output_shape=next_state_shape)
        self.pretrain_network(n_pretrain)

    def pretrain_network(self, n_pretrain):
        for i in range(n_pretrain):
            replay_batch = self.replay_buffer.get_batch(num=64)
            self.reward_network.train(replay_batch)
            self.next_state_network.train(replay_batch)

    def step(self, action):
        reward = self.reward_network.predict(self.state, action)
        self.state = self.next_state_network.predict(self.state, action)

        info = {}

        return self.state, reward, self.done, info

    def reset(self):
        replay_batch = self.replay_buffer.get_batch()
        self.reward_network.train(replay_batch)
        self.next_state_network.train(replay_batch)
        self.state = self.init_state()
        self.done = False

        return self.state

    def render(self, mode='human'):
        pass

    def init_state(self):
        pass

    def reset_signal(self):
        self.done = True