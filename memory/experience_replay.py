from common.base_memory import BaseMemory
import numpy as np


class ReplayMem(BaseMemory):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None for _ in range(self.capacity)]
        self.position = 0

    def add(self, trans):
        idx = self.position + 1 % self.capacity
        self.memory[idx] = trans

    def sample(self, num_samples):
        return np.random.choice(self.memory, num_samples)

    def __len__(self):
        return len(self.memory)
