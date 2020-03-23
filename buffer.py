import numpy as np

class Buffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.data = []

    def add(self, new_trans):
        self.data.append(new_trans)
        if len(self.data)> self.max_len:
            del self.data[0]

    def get_batch(self, num):
        return np.random.choice(self.data, num)

    def get_recent(self):
        return self.data[-1]

    @property
    def length(self):
        return len(self.data)
