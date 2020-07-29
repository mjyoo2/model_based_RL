import numpy as np
import zmq
import copy
import pickle as pkl

from threading import Thread

class Buffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.state_data = []
        self.action_data = []
        self.reward_data = []
        self.next_state_data = []
        self.new_data = 0
        self.read_lock = [False, False]
        self.write_lock = False

    def add(self, new_trans):
        while self.read_lock[0] or self.read_lock[1] or self.write_lock:
            pass
        self.write_lock = True
        self.reward_data.append(new_trans['reward'])
        self.next_state_data.append(new_trans['next_state'])
        self.state_data.append(new_trans['state'])
        self.action_data.append(new_trans['action'])
        if len(self.reward_data) > self.max_len:
            del self.reward_data[0]
            del self.next_state_data[0]
            del self.action_data[0]
            del self.state_data[0]
        self.write_lock = False

    def get_dataset(self):
        self.read_lock[0] = True
        while self.write_lock:
            pass
        self.new_data = 0
        state = copy.deepcopy(self.state_data)
        action = copy.deepcopy(self.action_data)
        next_state = copy.deepcopy(self.next_state_data)
        reward = copy.deepcopy(self.reward_data)
        self.read_lock[0] = False
        return state, action, next_state, reward

    def get_one(self):
        idx = np.random.randint(low=0, high=len(self.state_data)-1)
        return self.state_data[idx]

    def get_batch(self):
        idx = np.random.randint(low=0, high=len(self.state_data)-1)
        self.read_lock[1] = True
        while self.write_lock:
            pass
        data = {'reward': copy.deepcopy(self.reward_data[idx]), 'next_state': copy.deepcopy(self.next_state_data[idx]),
                'state': copy.deepcopy(self.state_data[idx]), 'action': copy.deepcopy(self.action_data[idx])}
        self.read_lock[1] = False
        return data

    @property
    def length(self):
        return len(self.state_data)


class ModelEnvBuffer(Buffer):
    def __init__(self, max_len, port):
        super().__init__(max_len)
        ctx = zmq.Context()
        self.recv_sock = ctx.socket(zmq.SUB)
        self.recv_sock.setsockopt_string(zmq.SUBSCRIBE, '')
        self.recv_sock.connect('tcp://{}:{}'.format('127.0.0.1', port + 30))

        recv_thread = Thread(target=self.recv_data, args=())
        recv_thread.start()

    def recv_data(self):
        while True:
            data = pkl.loads(self.recv_sock.recv())
            self.add(data)
            self.new_data += 1

class CenterBuffer(object):
    def __init__(self, port):
        # socket port is port + 30
        ctx = zmq.Context()
        self.send_sock = ctx.socket(zmq.PUB)
        self.send_sock.bind('tcp://*:{}'.format(port + 30))

        self.recv_sock = ctx.socket(zmq.SUB)
        self.recv_sock.bind('tcp://*:{}'.format(port + 31))
        self.recv_sock.setsockopt_string(zmq.SUBSCRIBE, '')

        self.lock = False
        self.sending_queue = []

    def run(self):
        send_thread = Thread(target=self.send_data, args=())
        send_thread.start()
        recv_thread = Thread(target=self.recv_data, args=())
        recv_thread.start()

    def recv_data(self):
        while True:
            data = pkl.loads(self.recv_sock.recv())
            self.sending_queue.append(data)

    def send_data(self):
        while True:
            if len(self.sending_queue) > 0:
                self.send_sock.send(pkl.dumps(self.sending_queue[0]))
                del self.sending_queue[0]
