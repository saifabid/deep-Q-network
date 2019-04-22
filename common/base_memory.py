import abc
from collections import namedtuple

Transition = namedtuple("Transition", ('state', 'action', 'reward', 'next_state', 'done'))


class BaseMemory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def append(self, trans):
        pass

    @abc.abstractmethod
    def random_batch(self, batch_size):
        pass
