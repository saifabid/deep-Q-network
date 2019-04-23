import abc
import numpy as np


class BasePolicy:
    @abc.abstractmethod
    def select_action(self, action_probs):
        pass

    @abc.abstractmethod
    def update(self, **kwargs):
        pass


class EpsGreedy(BasePolicy):
    def __init__(self, eps, decay, eps_end):
        self.eps = eps
        self.decay = decay
        self.eps_end = eps_end

    def select_action(self, action_values):
        n_actions = action_values.shape[0]
        if np.random.uniform() < self.eps:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(action_values)

    def update(self):
        self.eps = max(self.eps_end, self.decay * self.eps)
