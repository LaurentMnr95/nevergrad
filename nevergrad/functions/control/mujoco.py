# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np

# Test for mucojo installation
try:
    import mujoco_py
except ModuleNotFoundError:
    raise("Mujoco-py not installed.")


class GenericMujocoEnv:
    def __init__(self, env_name, state_mean, state_std, policy_dim, num_rollouts,
                 random_state):
        self.mean = np.array(state_mean)
        self.std = np.array(state_std)
        self.policy_dim = policy_dim
        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        self.env.seed(random_state)

    def __call__(self, x):
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_dim)

        returns = []
        for i in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            while not done:
                action = np.dot(M, (obs - self.mean) / self.std)
                obs, r, done, _ = self.env.step(action)
                totalr += r
            returns.append(totalr)

        return -np.mean(returns)
