import abc
from collections import namedtuple

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


class BaseMemory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add(self, trans):
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass
