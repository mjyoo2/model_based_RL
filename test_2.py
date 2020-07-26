import zmq
import time
import pickle as pkl

if __name__ == '__main__':
    ctx = zmq.Context()
    send_sock = ctx.socket(zmq.SUB)
    send_sock.connect('tcp://127.0.0.1:21111')

    while True:
        msg = send_sock.recv()
        print(pkl.loads(msg))