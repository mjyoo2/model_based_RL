from stable_baselines.common.callbacks import BaseCallback

import os
import pickle as pkl

class PPOCallback(BaseCallback):
    def __init__(self, get_data, verbose=0, get_replay=False):
        super(PPOCallback, self).__init__(verbose)
        self.get_replay = get_replay
        self.get_data = get_data
        self.callback_step = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

        self.model.env.env_method('train_network', 25)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        self.callback_step += 1
        if self.callback_step % 8196 == 0:
            parameters = self.model.get_parameters()
            with open('./network/mb_parameters.pkl', 'wb') as f:
                pkl.dump(parameters, f)

    def _on_training_end(self) -> None:
        pass


class MBCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MBCallback, self).__init__(verbose)
        self.callback_step = 0

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
        dir_list = os.listdir('./replay_data')
        if 'data.pkl' in dir_list:
            self.model.env.env_method('train_network', 15)
            os.remove('./replay_data/data.pkl')


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        parameters = self.model.get_parameters()
        with open('./network/mb_parameters.pkl', 'wb') as f:
            pkl.dump(parameters, f)
        dir_list = os.listdir('./network')
        if 'parameters.pkl' in dir_list:
            while True:
                try:
                    with open('./network/mb_parameters.pkl', 'rb') as f:
                        mb_parameters = pkl.load(f)
                    break
                except:
                    pass
            try:
                os.remove('./network/parameters.pkl')
            except:
                pass
        self.model.load_parameters(parameters)

    def _on_training_end(self) -> None:
        pass

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.data_save_param = 1

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.data_save_param * 10000 + 40000 < self.num_timesteps:
            self.model.env.env_method('buffer_save')
            self.data_save_param += 1


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        dir_list = os.listdir('./network')
        if 'mb_parameters.pkl' in dir_list:
            parameters = self.model.get_parameters()
            while True:
                try:
                    with open('./network/mb_parameters.pkl', 'rb') as f:
                        mb_parameters = pkl.load(f)
                    break
                except:
                    pass
            key = parameters.keys()
            for layer_key in key:
                parameters[layer_key] = 0.95 * parameters[layer_key] + 0.05 * mb_parameters[layer_key]
            self.model.load_parameters(parameters)
            while True:
                try:
                    with open('./network/parameters.pkl', 'wb') as f:
                        pkl.dump(parameters, f)
                    break
                except:
                    pass

    def _on_training_end(self) -> None:
        pass