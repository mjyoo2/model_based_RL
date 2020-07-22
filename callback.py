from stable_baselines.common.callbacks import BaseCallback
import pickle as pkl
import socket
from threading import Thread
from config import *

class MBCallback(BaseCallback):
    def __init__(self, real_RL_info, MBRL_info, verbose=0):
        super(MBCallback, self).__init__(verbose)
        self.callback_step = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(MBRL_info['socket_info'])
        self.real_RL_info = real_RL_info
        self.real_env_steps = 0
        self.num_updates = 0
        self.recv_parameters = None
        self.get_data = False
        self.read_lock = False
        self.write_lock = False
        recv_thread = Thread(target=self.recv_data, args=())
        recv_thread.start()

    def recv_data(self):
        while True:
            data = pkl.loads(self.socket.recv(SOCKET_QUEUE_SIZE))
            while self.read_lock:
                pass
            self.real_env_steps = data['steps']
            self.recv_parameters = data['parameters']
            self.write_lock = True
            self.write_lock = False
            self.get_data = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.real_env_steps >= MB_LEARN_INTERVAL * self.num_updates + MB_START:
            train = MB_TRAINING_EPOCHS
            print('training..')
            self.model.get_env().env_method(method_name='train_network', train=train)
            self.num_updates += 1

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_rollout_end(self) -> None:
        data = pkl.dumps(self.model.get_parameters())
        self.socket.sendto(data, self.real_RL_info['socket_info'])
        if self.get_data:
            self.read_lock = True
            while self.write_lock:
                pass
            self.model.load_parameters(self.recv_parameters)
            self.read_lock = False
            self.get_data = False

    def _on_training_end(self) -> None:
        pass

class MainLearnerCallback(BaseCallback):
    def __init__(self, real_RL_info, MBRL_info, verbose=0):
        super(MainLearnerCallback, self).__init__(verbose)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(real_RL_info['socket_info'])
        self.MBRL_info = MBRL_info
        self.recv_parameters = None
        self.get_data = False
        self.read_lock = False
        self.write_lock = False
        recv_thread = Thread(target=self.recv_data, args=())
        recv_thread.start()

    def recv_data(self):
        while True:
            data = pkl.loads(self.socket.recv(SOCKET_QUEUE_SIZE))
            while self.read_lock:
                pass
            self.write_lock = True
            self.recv_parameters = data
            self.write_lock = False
            self.get_data = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        data = pkl.dumps({'steps': self.num_timesteps, 'parameters': self.model.get_parameters()})
        self.socket.sendto(data, self.MBRL_info['socket_info'])

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.get_data:
            self.read_lock = True
            while self.write_lock:
                pass
            self.model.load_parameters(self.rollout())
            self.read_lock = False
            self.get_data = False
            data = pkl.dumps({'steps': self.num_timesteps, 'parameters': self.model.get_parameters()})
            self.socket.sendto(data, self.MBRL_info['socket_info'])
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def rollout(self):
        param_1 = self.recv_parameters
        param_2 = self.model.get_parameters()
        param_3 = {}
        key = param_1.keys()
        if IS_AGGREGATION_RATE_DECAY:
            alpha = AGGREGATION_RATE * ((5000000 - self.num_timesteps) / 5000000)
        else:
            alpha = AGGREGATION_RATE
        beta = 1 - alpha
        for layers in key:
            param_3[layers] = param_1[layers] * alpha + param_2[layers] * beta
        return param_3