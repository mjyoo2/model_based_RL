import numpy as np
import pickle as pkl
import os
from multiprocessing import managers
from stable_baselines.common.buffers import ReplayBuffer

class CommReplayBuffer(ReplayBuffer):
    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if reward < -1:
            data = (obs_t, action, -1, obs_tp1, done)
        if reward > 1:
            data = (obs_t, action, 1, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def save(self):
        save_data = self._encode_sample(np.arange(len(self)))
        save_dict = {'state_data': save_data[0], 'action_data': save_data[1], 'reward_data': save_data[2],
                     'next_state_data': save_data[3]}
        file_list = os.listdir('D:/memory/replay_data/')
        with open('D:/memory/replay_data/data_{}.pkl'.format(len(file_list)), 'wb') as f:
            pkl.dump(save_dict, f)

class Buffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.state_data = []
        self.action_data = []
        self.reward_data = []
        self.next_state_data = []

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
        return self.state_data, self.action_data, self.next_state_data, self.reward_data

    def get_one(self):
        idx = np.random.randint(low=0, high=len(self.state_data)-1)
        return self.state_data[idx]

    def save(self):
        save_dict = {'reward_data': self.reward_data, 'next_state_data': self.next_state_data, 'state_data': self.state_data,
                     'action_data': self.action_data}
        with open('./replay_data/data.pkl', 'wb') as f:
            pkl.dump(save_dict, f)

    def load(self):
        while True:
            try:
                with open('./replay_data/data.pkl', 'rb') as f:
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

class data_pipe(object):
    def __init__(self, data):
        self.data = data

    def get_data(self, key):
        return self.data[key]

    def put_data(self, key, new_data):
        self.data[key] = new_data
