from common.base_memory import BaseMemory
import numpy as np


class ReplayMem(BaseMemory):
    def __init__(self, buffer):
        self.buffer = buffer
        self.memory = [None for _ in range(self.buffer)]
        self.position = 0

    def random_batch(self, batch_size):
        return np.random.choice(self.memory, batch_size)

    def append(self, trans):
        idx = self.position + 1 % self.buffer
        self.memory[idx] = trans

    def __len__(self):
        return len(self.memory)
