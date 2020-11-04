# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np


class GenericMujocoEnv:
    """This class evaluates policy of OpenAI Gym environment.

    Parameters
    -----------
    env_name: str
        Gym environment name
    state_mean: list
        Average state values of multiple independent runs.
    state_std: list
        Standard deviation of state values of multiple independent runs.
    num_rollouts: int
        number of independent runs.
    random_state: int or None
        random state for reproducibility in Gym environment.
    """
    def __init__(self, env_name, state_mean, state_std, num_rollouts,
                 random_state):
        self.mean = np.array(state_mean)
        self.std = np.array(state_std)
        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        self.env.seed(random_state)

    def __call__(self, x, y=None):
        """Compute average cummulative reward of a given policy.
        Optionally, y is a second (neural) layer of weights.
        """
        returns = []
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            while not done:
                action = np.dot(x, (obs - self.mean) / self.std)
                # if y is not None:
                #     action = 0.1 * action
                #     if "Ant" in str(self.env):
                #         action = 0.1 * action
                #     if "Humanoid" in str(self.env):
                #         action = 0.01 * action
                if y is not None:
                    action = np.dot(y, np.tanh(action))
                obs, r, done, _ = self.env.step(action)
                totalr += r
            returns.append(totalr)

        return -np.mean(returns)
