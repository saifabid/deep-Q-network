from common.base_agent import BaseAgent
from common.base_memory import Transition, BaseMemory
import numpy as np

# TODO ﻿Every C steps reset ^Q~Q -> Define Q prime
# TODO Double DQN vs DQN -> max(Q(s', a))  <-> Q(s', argmax_a(Q'(s', a))
# TODO according to paper we need to clip the error term -> We also found it helpful to clip the error term from the update to be between -1 and 1.

class DQN(BaseAgent):
    def __init__(self,
                 config,
                 seed,
                 ob_space,
                 ac_space,
                 av_model,
                 memory,
                 policy,
                 ):
        super().__init__(seed, config, ob_space, ac_space)

        self.memory = memory
        self.av_model = av_model
        self.policy = policy
        self.gamma = self.config['dq']['gamma']
        self.eps = self.config['train']['eps_start']
        self.t_step = 0

    def load(self, *args, **kwargs):
        pass

    def act(self, state, *args, **kwargs):
        action_values = self.av_model.forward(state)
        selected_action = self.policy.select_action(action_values)

        return selected_action

    def step(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

        self.t_step = (self.t_step + 1) % self.config['exp_replay']['update_every']
        if self.__time_to_learn__():
            self.__learn__()

    def __time_to_learn__(self):
        if self.t_step == 0 and len(self.memory) > self.config['exp_replay']['batch']:
            return True

        return False

    def __update_hyper_params__(self):
        self.policy.update()
        pass

    def __learn__(self):
        state, action, reward, next_state, done = self.memory.random_batch(self.config['exp_replay']['batch'])

        Q = self.av_model.forward(next_state)
        labels = reward + self.gamma * np.max(Q, axis=1) * (1 - done)
        self.av_model.train(zip(state, labels))

        self.__update_hyper_params__()

    def save(self, *args, **kwargs):
        pass
