import gym

'''
main_learner_port = port + 10
model_based_learner_port = port + 20
model_env_center_buffer_send_port = port + 30
model_env_center_buffer_recv_port = port + 31
model_env_learner = port + 50 + env_num
'''

model_RL_ip = '127.0.0.1'
real_env_ip = '127.0.0.1'
model_env_ip = '127.0.0.1'
port = 33132
environment = lambda: gym.make('LunarLanderContinuous-v2')
env_info = {'action_space': gym.spaces.box.Box(low=0, high=1, shape=(2, )),
            'observation_space': gym.spaces.box.Box(low=0, high=1, shape=(8, ))}

real_env_info = (real_env_ip, port + 10)
MBRL_info = (model_env_ip, port + 20)

SOCKET_QUEUE_SIZE = 65536
AGGREGATION_RATE = 0.05
IS_AGGREGATION_RATE_DECAY = True
MAX_BUFFER_SIZE = 200000
MB_START = 50000
MB_LEARN_INTERVAL = 100000
MB_LAYER_STRUCTURE = [64, 64, 64, 64] 
MAIN_ENV_DELAY = 0.005
MAIN_ENV_NUM = 1 
MB_ENV_NUM = 4
MB_EPI_LENGTH = 256
MB_TRAINING_EPOCHS = 50
MB_TIMESTEPS = 100000000
MAIN_TIMESTEPS = 5000000
INIT_STATE_BUFFER_SIZE = 10000

