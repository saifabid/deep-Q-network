import abc
import numpy as np

class BasePolicy:
    @abc.abstractmethod
    def select_action(self, action_probs):
        pass


class EpsGreedy(BasePolicy):
    def __init__(self, eps):
        self.eps = eps

    def select_action(self, action_values):
        n_actions = action_values.shape[0]
        if np.random.uniform() < self.eps:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(action_values)
