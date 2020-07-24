import os
import zmq
import pickle as pkl

from buffer import Buffer
from network import Network
from config import *


class model_env(object):
    def __init__(self, env_data, name, verbose=1):
        self.replay_buffer = Buffer(MAX_BUFFER_SIZE, (model_env_ip, port+50))
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.verbose = verbose
        self.reward_network_list = []
        self.next_state_network_list = []

        ctx = zmq.Context()
        self.send_sock = ctx.socket(zmq.PUB)
        self.send_sock.bind('tcp://*:{}'.format(MBRL_info[1]))

        if not os.path.isdir('./weights'.format(name)):
            os.mkdir('./weights'.format(name))
        if not os.path.isdir('./weights/{}'.format(name)):
            os.mkdir('./weights/{}'.format(name))

    def set_up_network(self):
        if isinstance(self.action_space, gym.spaces.box.Box):
            action_shape = self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
            action_shape = self.action_space.n
        else:
            print("action space isn't Box or Discrete")
            action_shape = None

        state_shape = self.observation_space.shape[0]
        next_state_shape = self.observation_space.shape[0]

        for name in range(MB_ENV_NUM):
            reward_network = Network(layer_structure=MB_LAYER_STRUCTURE, action_shape=action_shape, state_shape=state_shape,
                                          output_shape=1, name='{}/reward_network'.format(name), last_layer='tanh')
            self.reward_network_list.append(reward_network)
            next_state_network = Network(layer_structure=MB_LAYER_STRUCTURE, action_shape=action_shape, state_shape=state_shape,
                                          output_shape=next_state_shape, name='{}/next_state_network'.format(name))
            self.next_state_network_list.append(next_state_network)

    def learn(self):
        while True:
            for i in range(MB_ENV_NUM):
                self.train_network(MB_TRAINING_EPOCHS, i)

    def train_network(self, epochs, index):
        self.reward_network_list[index].reinit()
        self.next_state_network_list[index].reinit()
        state_data, action_data, next_state_data, reward_data = self.replay_buffer.get_dataset()
        self.replay_buffer.read_lock = False
        self.reward_network_list[index].train(state_data, action_data, reward_data, training_epochs=epochs)
        self.next_state_network_list[index].train(state_data, action_data, next_state_data, training_epochs=epochs)
        data = {'reward_network': self.reward_network_list[index].network.get_weights(),
                'next_state_network': self.next_state_network_list[index].network.get_weights()}
        self.send_sock.send(pkl.dumps(data))

    def send_init_state(self):
        data = {'type': 'init_state', 'data': self.replay_buffer.get_one()}
        self.send_sock.send(pkl.dumps(data))