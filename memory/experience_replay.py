import numpy as np

from common.base_memory import BaseMemory


class ReplayMem(BaseMemory):
    # TODO Make sure the return objects from memory is always numpy ndarrays
    def __init__(self, buffer):
        self.buffer = buffer
        # TODO a more efficient way to store the transitions
        self.memory = []
        self.position = 0

    def random_batch(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size)
        train_batch = [self.memory[idx] for idx in idxs]
        return map(lambda x: np.array(x), zip(*train_batch))

    def append(self, trans):
        self.memory.append(trans)
        if len(self.memory) > self.buffer:
            self.memory = self.memory[1:]

    def __len__(self):
        return len(self.memory)
