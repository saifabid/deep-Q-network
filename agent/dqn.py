from common.base_agent import BaseAgent
from memory.experience_replay import ReplayMem
from model.nn_tensorflow import SimpleNN


class DQN(BaseAgent):
    def __init__(self, config_path, seed, ob_space, ac_space):
        super().__init__(seed, config_path, ob_space, ac_space)

        self.memory = ReplayMem(buffer=self.config['exp_replay']['buffer'])
        self.av_model = SimpleNN(input_shape=ob_space, output_shape=ac_space)
        # TODO self.policy = create Policy

    def load(self, *args, **kwargs):
        pass

    def act(self, state, *args, **kwargs):
        action_probs = self.av_model.forward(state)
        # TODO get selected_action from the policy -> selection_action = self.policy.get_action(action_probs)
        # TODO time to learn? -> train the model
        # TODO adjust the parameters -> Gamma, learning rate, ...

        pass

    def save(self, *args, **kwargs):
        pass
