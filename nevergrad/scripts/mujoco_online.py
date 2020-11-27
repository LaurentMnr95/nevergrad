import gym
import numpy as np
from multiprocessing import Pool
import nevergrad as ng


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

    def __init__(self, env_name, num_rollouts, random_state):
        # self.mean = np.array(state_mean)
        # self.std = np.array(state_std)
        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        self.env.seed(random_state)
        self.global_stats = Stats()

    def __call__(self, x):
        """Compute average cummulative reward of a given policy.
        """
        x = x.value
        state_mean = self.global_stats.mean
        state_std = self.global_stats.std

        stats_run = Stats()
        returns = []
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.
            while not done:
                action = np.dot(x, (obs - state_mean) / (state_std + 1e-8))
                obs, r, done, _ = self.env.step(action)
                stats_run.push(obs)
                totalr += r
            returns.append(totalr)

        return -np.mean(returns), stats_run


class Stats:
    def __init__(self):
        self.n = 0
        self.sum_obs = 0
        self.sum_obs2 = 0

    def update(self, stats_run):
        self.n += stats_run.n
        self.sum_obs += stats_run.sum_obs
        self.sum_obs2 += stats_run.sum_obs2

    def push(self, obs):
        self.n += 1
        self.sum_obs += obs
        self.sum_obs2 += obs ** 2

    @property
    def mean(self):
        if self.n == 0:
            return 0
        return self.sum_obs / self.n

    @property
    def std(self):
        if self.n == 0:
            return 1
        a = self.sum_obs2 / self.n - (self.sum_obs / self.n) ** 2
        a = np.sqrt(a)
        a[a < 1e-7] = float("inf")
        return a


env_name = 'Ant-v2'
policy_dim = (8, 111)

num_workers = 1
budget = 10000
param = ng.p.Parameterng.p.Array(shape=(8, 111)).set_mutation(sigma=0.001)
optimizer = ng.optimizers.DiagonalCMA(parametrization=param, budget=budget)
num_rollouts = 1
random_state = 1

elapsed_budget = 0

env = GenericMujocoEnv(env_name, num_rollouts, random_state)
while elapsed_budget <= budget:
    x = []
    for _ in range(num_workers):
        x.append(optimizer.ask())

    with Pool(num_workers) as p:
        y = p.map(env, x)

    for _, stats_run in y:
        env.global_stats.update(stats_run)

    for u, v in zip(x, y):
        print(v[0])
        optimizer.tell(u, v[0])
        elapsed_budget += 1
        if elapsed_budget > budget:
            print("ok")
            break
