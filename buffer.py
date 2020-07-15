import numpy as np
import pickle as pkl
import socket
import time
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
        self.locks = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(socket_info)
        self.socket.bind(socket_info)
        getdata_thread = Thread(target=self.get_data, args=())
        getdata_thread.start()

    def get_data(self):
        while True:
            while self.locks:
                time.sleep(0.01)
                print('?????')
            data = self.socket.recv(4096)
            self.add(pkl.loads(data))
            self.new_data += 1

    def add(self, new_trans):
        self.reward_data.append(new_trans['reward'])
        self.next_state_data.append(new_trans['next_state'])
        self.state_data.append(new_trans['state'])
        self.action_data.append(new_trans['action'])
        if len(self.reward_data)> self.max_len:
            del self.reward_data[0]
            del self.next_state_data[0]
            del self.action_data[0]
            del self.state_data[0]

    def get_dataset(self):
        self.locks = True
        time.sleep(0.1)
        self.new_data = 0
        return copy.deepcopy(self.state_data), copy.deepcopy(self.action_data), \
               copy.deepcopy(self.next_state_data), copy.deepcopy(self.reward_data)

    def get_one(self):
        idx = np.random.randint(low=0, high=len(self.state_data)-1)
        return self.state_data[idx]

    def save(self, num):
        save_dict = {'reward_data': self.reward_data, 'next_state_data': self.next_state_data, 'state_data': self.state_data,
                     'action_data': self.action_data}
        with open('./replay_data/data_{}.pkl'.format(num % 4), 'wb') as f:
            pkl.dump(save_dict, f)

    def load(self, name):
        while True:
            try:
                if name:
                    with open('./replay_data/data_{}.pkl'.format(name), 'rb') as f:
                        load_dict = pkl.load(f)
                else:
                    with open('./replay_data/data.pkl'.format(name), 'rb') as f:
                        load_dict = pkl.load(f)
                break
            except:
                pass
        print("buffer loading...")
        self.reward_data = load_dict['reward_data']
        self.next_state_data = load_dict['next_state_data']
        self.state_data = load_dict['state_data']
        self.action_data = load_dict['action_data']

    @property
    def length(self):
        return len(self.state_data)
