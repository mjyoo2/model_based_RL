from model_env import ModelEnv
from buffer import CenterBuffer
from multiprocessing import Process
from config import *


def run_model_env(name, verbose=1):
    model_env = ModelEnv(env_info, name, verbose)
    model_env.run()


def run_center_buffer():
    center_buffer = CenterBuffer(port)
    center_buffer.run()


if __name__ == '__main__':
    process_list = [Process(target=run_center_buffer, args=())]
    for i in range(MB_ENV_NUM):
        process_list.append(Process(target=run_model_env, args=(i, )))
    for child_process in process_list:
        child_process.start()
    for child_process in process_list:
        child_process.join()