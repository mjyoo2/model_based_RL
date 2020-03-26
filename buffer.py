import numpy as np
from multiprocessing import managers

class Buffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.state_data = []
        self.state_action_data = []
        self.reward_data = []
        self.next_state_data = []

    def add(self, new_trans):
        self.reward_data.append(new_trans['reward'])
        self.next_state_data.append(new_trans['next_state'])
        self.state_data.append({'state': new_trans['state']})
        state_action = np.concatenate([new_trans['state'].flatten(), new_trans['action'].flatten()])
        self.state_action_data.append(state_action)
        if len(self.reward_data)> self.max_len:
            del self.reward_data[0]
            del self.next_state_data[0]
            del self.state_action_data[0]
            del self.state_data[0]

    def get_dataset(self):
        return self.state_action_data, self.next_state_data, self.reward_data

    def get_one(self):
        print(np.random.choice(self.s))
        return np.random.choice(self.state_data, 1)[0]

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
