import abc


class BaseModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, data):
        pass

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset_metrics(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def metrics():
        pass
