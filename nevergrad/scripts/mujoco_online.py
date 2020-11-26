import gym
import numpy as np
from multiprocessing import Pool


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

    def __init__(self, env_name, num_rollouts,
                 random_state):
        # self.mean = np.array(state_mean)
        # self.std = np.array(state_std)
        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        self.env.seed(random_state)

    def __call__(self, x, state_mean, state_std):
        """Compute average cummulative reward of a given policy.
        """
        returns = []
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            while not done:
                action = np.dot(x, (obs - state_mean) / state_std)
                obs, r, done, _ = self.env.step(action)
                totalr += r
            returns.append(totalr)

        return -np.mean(returns), 0, 0  # todo: return stats of runs


def update_statistics(state_mean, state_std, stats_run):
    return ...


num_workers = 5
optimizer = ...
env_name = ...
num_rollouts = ...
random_state = ...
state_mean, state_std = 0, 0

while True:
    x = []
    for _ in range(num_workers):
        x.append((optimizer.ask(), state_mean, state_std))

    env = GenericMujocoEnv(env_name, num_rollouts, random_state)

    with Pool(num_workers) as p:
        y = p.map(env, x)

    for _, stats_run in y:
        state_mean, state_std = update_statistics(state_mean, state_std, stats_run)

    for u, v in zip(x, y):
        optimizer.tell(u, v[0])

