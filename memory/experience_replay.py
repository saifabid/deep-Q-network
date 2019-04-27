from common.base_memory import BaseMemory
import random
import numpy as np


class ReplayMem(BaseMemory):
    def __init__(self, buffer):
        self.buffer = buffer
        self.memory = []
        self.position = 0

    def random_batch(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size)
        return [self.memory[idx] for idx in idxs]

    def append(self, trans):
        self.memory.append(trans)
        if len(self.memory) > self.buffer:
            self.memory = self.memory[1:]

    def __len__(self):
        return len(self.memory)
