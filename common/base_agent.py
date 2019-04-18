import abc


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass
