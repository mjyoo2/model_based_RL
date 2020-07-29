import numpy as np
import time
import os
import zmq
import pickle as pkl

from buffer import Buffer
from network import Network
from config import *
from threading import Thread

class AsynMB(gym.Env):
    def __init__(self, env_data, n_steps, name, verbose=1):
        self.replay_buffer = Buffer(INIT_STATE_BUFFER_SIZE)
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.state = None
        self.timesteps = 0
        self.verbose = verbose
        self.n_steps = n_steps
        self.updated = False
        self.first_updated = False
        self.read_lock = False
        self.write_lock = False
        self.recv_weights = None
        self.name = name

        ctx = zmq.Context()
        self.recv_sock = ctx.socket(zmq.SUB)
        self.recv_sock.connect('tcp://{}:{}'.format(model_env_ip, port + 50 + name))
        self.recv_sock.setsockopt_string(zmq.SUBSCRIBE, '')

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

        recv_thread = Thread(target=self.recv_data, args=())
        recv_thread.start()

    def recv_data_handle(self, data):
        if data['type'] == 'init_state':
            self.replay_buffer.add(data['data'])
        elif data['type'] == 'weights':
            self.recv_weights = data
            self.updated = True
            print("new weight setting")

    def recv_data(self):
        while True:
            data = pkl.loads(self.recv_sock.recv())
            
            while self.read_lock:
                pass
            self.write_lock = True
            self.recv_data_handle(data)
            self.write_lock = False

    def get_weight(self):
        if self.updated:
            self.read_lock = True
            while self.write_lock:
                pass
            self.reward_network.network.set_weights(self.recv_weights['reward_network'])
            self.next_state_network.network.set_weights(self.recv_weights['next_state_network'])
            self.read_lock = False
            self.updated = False
            self.first_updated = True
            return True
        return False

    def step(self, action):
        self.timesteps += 1
        done = False
        reward = self.reward_network.predict(self.state, action)
        self.state = self.state + self.next_state_network.predict(self.state, action)
        for value in self.state.flatten():
            if np.isnan(value):
                print('NAN!')
        if np.isnan(reward):
            print('NAN!')
        info = {}
        if self.n_steps == self.timesteps:
            done = True
        return self.state.flatten(), reward[0][0], done, info

    def reset(self):
        while not self.first_updated:
            self.get_weight()
        self.get_weight()
        self.state = self.init_state()
        self.timesteps = 0
        return self.state.flatten()

    def render(self, mode='human'):
        pass

    def init_state(self):
        return self.replay_buffer.get_one()
