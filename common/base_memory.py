import abc


class BaseMemory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass
