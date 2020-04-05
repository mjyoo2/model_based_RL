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
        self.model.env.env_method('train_network', 25)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.callback_step += 1
        if self.callback_step % 4096 == 0:
            print('{} step! rollout!'.format(self.callback_step))
            parameters = self.model.get_parameters()
            with open('./network/parameters.pkl', 'wb') as f:
                pkl.dump(parameters, f)
            self.get_data(20000)
            self.model.setup_model()


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


class MBCallback(BaseCallback):
    def __init__(self, verbose=0, get_replay=False):
        super(MBCallback, self).__init__(verbose)
        self.get_replay = get_replay

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
        if self.num_timesteps % 256 == 0:
            file_list = os.listdir('D:/memory/mb_network/')
            parameters = self.model.get_parameters()
            with open('D:/memory/mb_network/mb_parameters_{}.pkl'.format(len(file_list)), 'wb') as f:
                pkl.dump(parameters, f)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps % 256 == 0:
            file_list = os.listdir('D:/memory/network/')
            if len(file_list) < 3:
                pass
            else:
                with open('D:/memory/network/parameters_{}.pkl'.format(len(file_list) - 1), 'rb') as f:
                    parameters = pkl.load(f)
            self.model.load_parameters(parameters)

    def _on_training_end(self) -> None:
        pass

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0, get_replay=False):
        super(CustomCallback, self).__init__(verbose)
        self.get_replay = get_replay

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
        if self.num_timesteps % 256 == 0:
            self.model.replay_buffer.save()
            parameters = self.model.get_parameters()
            file_list = os.listdir('D:/memory/network/')
            with open('D:/memory/network/parameters_{}.pkl'.format(len(file_list)), 'wb') as f:
                pkl.dump(parameters, f)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps % 256 == 0:
            file_list = os.listdir('D:/memory/mb_network/')
            if len(file_list) < 3:
                pass
            else:
                with open('D:/memory/mb_network/mb_parameters_{}.pkl'.format(len(file_list) - 1), 'rb') as f:
                    parameters = pkl.load(f)
                self.model.load_parameters(parameters)

    def _on_training_end(self) -> None:
        pass