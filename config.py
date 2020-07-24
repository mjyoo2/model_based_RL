import gym

model_env_ip = '127.0.0.1'
real_env_ip = '127.0.0.1'
port = 10001
environment = lambda: gym.make('LunarLanderContinuous-v2')
env_info = {'action_space': gym.spaces.box.Box(low=0, high=1, shape=(2, )),
            'observation_space': gym.spaces.box.Box(low=0, high=1, shape=(8, ))}

real_env_info = (real_env_ip, port + 10)
MBRL_info = (model_env_ip, port + 20)

SOCKET_QUEUE_SIZE = 65536
AGGREGATION_RATE = 0.05
IS_AGGREGATION_RATE_DECAY = True
MAX_BUFFER_SIZE = 200000
MB_START = 5000
MB_LEARN_INTERVAL = 50000
MB_LAYER_STRUCTURE = [64, 64]
MAIN_ENV_DELAY = 0
MAIN_ENV_NUM = 4
MB_ENV_NUM = 4

MB_EPI_LENGTH = 128
MB_TRAINING_EPOCHS = 20
MB_TIMESTEPS = 100000000
MAIN_TIMESTEPS = 5000000