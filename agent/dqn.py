from common.base_agent import BaseAgent
from common.base_memory import Transition
from common.base_policy import EpsGreedy
from memory.experience_replay import ReplayMem
from model.nn_tensorflow import SimpleNN


class DQN(BaseAgent):
    def __init__(self, config_path, seed, ob_space, ac_space):
        super().__init__(seed, config_path, ob_space, ac_space)

        self.memory = ReplayMem(buffer=self.config['exp_replay']['buffer'])
        self.av_model = SimpleNN(input_shape=ob_space, output_shape=ac_space)
        self.policy = EpsGreedy(eps=self.config['train']['eps_start'])
        self.gamma = self.config['dq']['gamma']

    def load(self, *args, **kwargs):
        pass

    def act(self, state, *args, **kwargs):
        action_values = self.av_model.forward(state)
        selected_action = self.policy.select_action(action_values)

        return selected_action

    def step(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state)
        self.memory.append(transition)

        self.t_step = (self.t_step + 1) % self.config['exp_replay']['update_every']
        if self.__time_to_train__():
                self.__learn__()

    def __time_to_train__(self):
        if self.t_step == 0 and len(self.memory) > self.config['exp_replay']['batch']:
            return True

        return False

    def __update_hyper_params__(self):
        # TODO if necessary
        pass

    def __learn__(self):
        train_batch = self.memory.random_batch(self.config['exp_replay']['batch'])

        Q = self.av_model.forward(train_batch.next_state)
        labels = train_batch.reward + self.gamma * Q * (1 - train_batch.done)
        self.av_model.train(zip(train_batch.state, labels))

        self.__update_hyper_params__()

    def save(self, *args, **kwargs):
        pass
