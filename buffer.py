import numpy as np
import pickle as pkl
import zmq
import copy

from threading import Thread


class Buffer(object):
    def __init__(self, max_len, socket_info):
        self.max_len = max_len
        self.state_data = []
        self.action_data = []
        self.reward_data = []
        self.next_state_data = []
        self.new_data = 0
        self.read_lock = False
        self.write_lock = False

        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.SUB)
        self.socket.bind('tcp://{}:{}'.format(socket_info[0], socket_info[1]))
        getdata_thread = Thread(target=self.get_data, args=())
        getdata_thread.start()

    def get_data(self):
        while True:
            while self.read_lock:
                pass
            data = self.socket.recv()
            self.write_lock = True
            self.add(pkl.loads(data))
            self.new_data += 1
            self.write_lock = False

    def add(self, new_trans):
        self.reward_data.append(new_trans['reward'])
        self.next_state_data.append(new_trans['next_state'])
        self.state_data.append(new_trans['state'])
        self.action_data.append(new_trans['action'])
        if len(self.reward_data) > self.max_len:
            del self.reward_data[0]
            del self.next_state_data[0]
            del self.action_data[0]
            del self.state_data[0]

    def get_dataset(self):
        self.read_lock = True
        while self.write_lock:
            pass
        self.new_data = 0
        return copy.deepcopy(self.state_data), copy.deepcopy(self.action_data), \
               copy.deepcopy(self.next_state_data), copy.deepcopy(self.reward_data)

    def get_one(self):
        idx = np.random.randint(low=0, high=len(self.state_data)-1)
        return self.state_data[idx]

    @property
    def length(self):
        return len(self.state_data)

