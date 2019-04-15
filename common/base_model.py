import abc


class BaseModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, data):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def saveModel(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loadModel(self, *args, **kwargs):
        pass
