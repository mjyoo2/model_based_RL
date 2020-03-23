import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.data = []

    def add(self, new_trans):
        self.data.append(new_trans)
        if len(self.data)> self.max_len:
            del self.data[0]

    def get_batch(self, num):
        return np.random.choice(self.data, num)
