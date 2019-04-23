import abc

from common.utils import get_config


class BaseAgent(metaclass=abc.ABCMeta):

    def __init__(self,
                 seed,
                 config,
                 ob_space,
                 ac_space):
        self.config = config
        self.seed = seed
        self.ob_space = ob_space
        self.ac_space = ac_space

    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass
