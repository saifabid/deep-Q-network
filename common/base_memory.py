import abc
from collections import namedtuple

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


class BaseMemory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def append(self, trans):
        pass

    @abc.abstractmethod
    def random_batch(self, batch_size):
        pass
