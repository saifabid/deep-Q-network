import abc

from common.utils import get_config


class BaseAgent(metaclass=abc.ABCMeta):

    def __init__(self, config_path):
        self.config = get_config(config_path)

    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass
