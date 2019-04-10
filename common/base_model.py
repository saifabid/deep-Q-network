import abc


class BaseModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def saveModel(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loadModel(self, *args, **kwargs):
        pass
