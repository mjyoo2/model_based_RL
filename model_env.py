import os
import zmq
import pickle as pkl
import time

from buffer import ModelEnvBuffer
from network import Network
from config import *
from threading import Thread

class ModelEnv(object):
    def __init__(self, env_data, name, verbose=1):
        self.replay_buffer = ModelEnvBuffer(MAX_BUFFER_SIZE, port)
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.verbose = verbose
        self.reward_network_list = []
        self.next_state_network_list = []
        self.start = True

        ctx = zmq.Context()
        self.send_sock = ctx.socket(zmq.PUB)
        self.send_sock.bind('tcp://*:{}'.format(port + 50 + name))

        if not os.path.isdir('./weights'.format(name)):
            os.mkdir('./weights'.format(name))
        if not os.path.isdir('./weights/{}'.format(name)):
            os.mkdir('./weights/{}'.format(name))

        if isinstance(self.action_space, gym.spaces.box.Box):
            action_shape = self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
            action_shape = self.action_space.n
        else:
            print("action space isn't Box or Discrete")
            action_shape = None

        state_shape = self.observation_space.shape[0]
        next_state_shape = self.observation_space.shape[0]

        self.reward_network = Network(layer_structure=MB_LAYER_STRUCTURE, action_shape=action_shape, state_shape=state_shape,
                                 output_shape=1, name='{}/reward_network'.format(name), last_layer='tanh')
        self.next_state_network = Network(layer_structure=MB_LAYER_STRUCTURE, action_shape=action_shape, state_shape=state_shape,
                                     output_shape=next_state_shape, name='{}/next_state_network'.format(name))

        send_thread = Thread(target=self.send_init_state, args=())
        send_thread.start()

    def run(self):
        while True:
             if (self.start and self.replay_buffer.length >= MB_START) or self.replay_buffer.new_data >= MB_LEARN_INTERVAL:
                self.train_network(MB_TRAINING_EPOCHS)
                self.start = False


    def train_network(self, epochs):
        self.reward_network.reinit()
        self.next_state_network.reinit()
        state_data, action_data, next_state_data, reward_data = self.replay_buffer.get_dataset()
        self.replay_buffer.read_lock = False
        self.reward_network.train(state_data, action_data, reward_data, training_epochs=epochs)
        self.next_state_network.train(state_data, action_data, next_state_data, training_epochs=epochs)
        data = {'type': 'weights',
                'reward_network': self.reward_network.network.get_weights(),
                'next_state_network': self.next_state_network.network.get_weights()}
        self.send_sock.send(pkl.dumps(data))

    def send_init_state(self):
        while True:
            if self.replay_buffer.length > 100:
                data = {'type': 'init_state', 'data': self.replay_buffer.get_batch()}
                self.send_sock.send(pkl.dumps(data))
                time.sleep(1)