from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class Player:
    def __init__(self,
                 agent,
                 env,
                 config,
                 game_logger,
                 train_mode):
        self.agent = agent
        self.env = env
        self.glogger = game_logger
        self.config = config
        self.train_mode = train_mode

    def step(self, state):
        action = self.agent.act(state)
        next_state, reward, done, info = self.env.step(action)

        if self.train_mode:
            self.agent.step(state, action, reward, next_state, done)

        return reward, next_state, done

    def play_episode(self):
        state = self.env.reset()

        score = 0
        for t in range(self.config['train']['max_t']):
            reward, state, done = self.step(state)
            score += reward

            print('\r', 'Iteration', t, 'Score:', score, end='')
            if done:
                return score

        return score

    def play(self):
        for i_episode in range(1, 10):
            score = self.play_episode()
            self.glogger.log_score(score)

            if self.train_mode:
                solved = self.glogger.report_solved(i_episode)

                if solved:
                    # TODO Save with file name
                    self.agent.save(i_episode)

        self.glogger.plot_scores()


class GameLogger:
    # TODO Decouple visualization from logger
    # TODO Add Tensorboard support to GameLogger
    # TODO Create an interface for game logger if necessary

    def __init__(self, scores_window_size, winning_threshold):
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=scores_window_size)  # last 100 scores
        self.winning_threshold = winning_threshold

    def log_score(self, score):
        self.scores_window.append(score)
        self.scores.append(score)

    def plot_scores(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    def report_solved(self, i_episode):
        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(self.scores_window)))

        if i_episode % 500 == 0:
            self.plot_scores()

        if np.mean(self.scores_window) >= self.winning_threshold:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                       np.mean(self.scores_window)))
            return True

        return False
