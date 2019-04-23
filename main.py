#!/usr/bin/env python3
import gym
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import argparse
import common.utils as cutils
from agent.dqn import DQN
from common.base_policy import EpsGreedy
from memory.experience_replay import ReplayMem
from model.nn_tensorflow import SimpleNN



def main(config_path, env_name, train_mode=True, weights_path=None):
    """Load the environment, create an agent, and train it.
    """
    config = cutils.get_config(config_path)
    env, brain_name = cutils.load_environment(env_name)
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    memory = ReplayMem(buffer=config['exp_replay']['buffer'])
    av_model = SimpleNN(input_shape=state_size, output_shape=action_size)
    policy = EpsGreedy(eps=config['train']['eps_start'])

    agent = DQN(config,
                seed=0,
                ob_space=state_size,
                ac_space=action_size,
                av_model=av_model,
                memory=memory,
                policy=policy)

    if weights_path is not None:
        agent.load(weights_path)

    scores = play(env, brain_name, agent, config, train_mode, weights_path)
    env.close()

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Agent')
    parser.add_argument('--env', action="store", dest="env_name", type=str,
                        help='Name of the openai gym game to play')
    parser.add_argument('--train', action="store_true", default=False,
                        help='Train the DQN agent if used, else load the trained weights and play the game')
    parser.add_argument('--weights', action="store", dest="path", type=str,
                        help='path of .pth file with the trained weights')
    args = parser.parse_args()

    print("Train_mode: {}".format(args.train))
    main("model/config.yaml", train_mode=args.train, weights_path=args.path, env_name=args.env_name)
