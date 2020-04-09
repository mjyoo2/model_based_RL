import pickle as pkl

class FileLock(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self._lock = False

    def file_write(self):

    def file_read(self):

    def lock(self):
        self._lock = True

    def unlock(self):
        self._lock = False