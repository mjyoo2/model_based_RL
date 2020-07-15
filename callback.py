from stable_baselines.common.callbacks import BaseCallback

import os
import pickle as pkl
import socket

class MBCallback(BaseCallback):
    def __init__(self, real_RL_info, MBRL_info, verbose=0):
        super(MBCallback, self).__init__(verbose)
        self.callback_step = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(MBRL_info['socket_info'])
        self.real_RL_info = real_RL_info

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        data = pkl.loads(self.socket.recv(65536))
        self.model.load_parameters(data)


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

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

    def recv_data(self):
        while True:
            data = pkl.loads(self.socket.recv(4096))
            self.recv_parameters = data
            self.get_data = True


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        data = pkl.dumps(self.model.get_parameters())
        print(len(data))
        self.socket.sendto(data, self.MBRL_info['socket_info'])


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        if self.get_data:
            data = pkl.loads(self.socket.recv(4096))
            self.model.load_parameters(data)
            self.get_data = False

    def _on_training_end(self) -> None:
        pass