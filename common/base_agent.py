import abc


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        pass
