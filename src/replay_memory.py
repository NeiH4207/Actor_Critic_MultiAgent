import numpy as np
import random
from collections import deque
import torch


class ReplayMemory():
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.maxSize = max_size
        self.len = 0
        self.batch_size = batch_size

    def add_all(self, rows):
        """
		adds a particular transaction in the memory buffer
        """
        size = len(rows[0])
        for i in range(size):
            transition = (rows[0][i], rows[1][i], rows[2][i])
            self.len += 1
            if self.len > self.maxSize:
                self.len = self.maxSize
            self.buffer.append(transition)

    def sample(self):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        batch_size = min(self.batch_size, self.len)
        batch = random.sample(self.buffer, batch_size)
        s_arr = np.array([arr[0] for arr in batch], dtype = np.float32)
        log_p_arr = np.array([arr[1] for arr in batch], dtype = np.float32)
        r_arr = np.array([arr[2] for arr in batch], dtype = np.float32)
        return s_arr, log_p_arr, r_arr