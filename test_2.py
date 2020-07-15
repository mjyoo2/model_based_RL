import socket
from threading import Thread
import time
import pickle as pkl

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 11000))
    time.sleep(10)
    data = sock.recv(2048)
    print(pkl.loads(data))
