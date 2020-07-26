import zmq
import time
import pickle as pkl

from multiprocessing import Process

class test(object):
    def __init__(self):
        ctx = zmq.Context()
        self.data = [0, 0, 0, 0]
        time.sleep(1)

    def send(self):
        process_list = []
        for i in range(4):
            process_list.append(Process(target=self.send_msg, args=(i, )))
        for subprocess in process_list:
            subprocess.start()
        for subprocess in process_list:
            subprocess.join()

    def send_msg(self, index):
        self.data[index] = index


if __name__ =='__main__':
    test_class = test()
    test_class.send()
    print(test_class.data)
