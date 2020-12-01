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

    def __init__(self, env_name, num_rollouts, random_state=None):
        # self.mean = np.array(state_mean)
        # self.std = np.array(state_std)
        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        if random_state is not None:
            self.env.seed(random_state)
        self.global_stats = Stats()

    def __call__(self, x):
        """Compute average cummulative reward of a given policy.
        """
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


class MujocoExperiment:
    def __init__(self, num_workers, budget, optimizer, rescaling, policy_dim, env_name, num_rollouts,
                 random_state=None):
        self.num_workers = num_workers
        self.budget = budget
        self.num_rollouts = num_rollouts
        self.random_state = random_state
        self.optimizer = optimizer
        self.policy_dim = policy_dim
        self.env_name = env_name
        self.rescaling = rescaling

    def __call__(self):
        param = ng.p.Array(shape=self.policy_dim).set_mutation(sigma=self.rescaling)
        optimizer = ng.optimizers.registry[self.optimizer](parametrization=param, budget=self.budget,
                                                           num_workers=self.num_workers*2)
        elapsed_budget = 0
        env = GenericMujocoEnv(self.env_name, self.num_rollouts, self.random_state)
        while elapsed_budget <= self.budget:
            x = []
            for _ in range(self.num_workers):
                x.append(optimizer.ask())

            x_values = [v.value for v in x]

            with Pool(self.num_workers) as p:
                y = p.map(env, x_values)

            for _, stats_run in y:
                env.global_stats.update(stats_run)

            for u, v in zip(x, y):
                elapsed_budget += 1
                if elapsed_budget > self.budget:
                    break
                if v[0] < 0:
                    print(f"[{elapsed_budget}/{budget}]: score = {v[0]}")
                else:
                    print(u.value)
                    print(f"[{elapsed_budget}/{budget}]: score = {v[0]}")
                optimizer.tell(u, v[0])

        recommendation = optimizer.provide_recommendation()
        score = env(recommendation.value)
        print(f"Final score = {score[0]}")

        with open('mujoco.txt', 'a') as f:
            f.write(f"{self.num_workers},{self.budget},{self.num_rollouts},{self.random_state},"
                    f"{self.optimizer},{self.policy_dim},{self.env_name},{self.rescaling},{score}\n")


if __name__ == "__main__":
    num_workers = 20
    budget = 10000
    optimizer = "DiagonalCMA"
    # rescaling = 0.0001
    # policy_dim = (17, 376)
    # env_name = "Humanoid-v2"
    rescaling = 0.001
    policy_dim = (8, 111)
    env_name = "Ant-v2"
    num_rollouts = 1
    exp = MujocoExperiment(num_workers, budget, optimizer, rescaling, policy_dim, env_name, num_rollouts)
    exp()


# import gym
# import numpy as np
# from multiprocessing import Pool
# import nevergrad as ng
#
#
# class GenericMujocoEnv:
#     """This class evaluates policy of OpenAI Gym environment.
#     Parameters
#     -----------
#     env_name: str
#         Gym environment name
#     state_mean: list
#         Average state values of multiple independent runs.
#     state_std: list
#         Standard deviation of state values of multiple independent runs.
#     num_rollouts: int
#         number of independent runs.
#     random_state: int or None
#         random state for reproducibility in Gym environment.
#     """
#
#     def __init__(self, env_name, num_rollouts, random_state=None):
#         # self.mean = np.array(state_mean)
#         # self.std = np.array(state_std)
#         self.env = gym.make(env_name)
#         self.num_rollouts = num_rollouts
#         if random_state is not None:
#             self.env.seed(random_state)
#         self.global_stats = Stats()
#
#     def __call__(self, x):
#         """Compute average cummulative reward of a given policy.
#         """
#         x = x.value
#         state_mean = self.global_stats.mean
#         state_std = self.global_stats.std
#
#         stats_run = Stats()
#         returns = []
#         for _ in range(self.num_rollouts):
#             obs = self.env.reset()
#             done = False
#             totalr = 0.
#             while not done:
#                 action = np.dot(x, (obs - state_mean) / (state_std + 1e-8))
#                 obs, r, done, _ = self.env.step(action)
#                 stats_run.push(obs)
#                 totalr += r
#             returns.append(totalr)
#
#         return -np.mean(returns), stats_run
#
#
# class Stats:
#     def __init__(self):
#         self.n = 0
#         self.sum_obs = 0
#         self.sum_obs2 = 0
#
#     def update(self, stats_run):
#         self.n += stats_run.n
#         self.sum_obs += stats_run.sum_obs
#         self.sum_obs2 += stats_run.sum_obs2
#
#     def push(self, obs):
#         self.n += 1
#         self.sum_obs += obs
#         self.sum_obs2 += obs ** 2
#
#     @property
#     def mean(self):
#         if self.n == 0:
#             return 0
#         return self.sum_obs / self.n
#
#     @property
#     def std(self):
#         if self.n == 0:
#             return 1
#         a = self.sum_obs2 / self.n - (self.sum_obs / self.n) ** 2
#         a = np.sqrt(a)
#         a[a < 1e-7] = float("inf")
#         return a
#
#
# class MujocoExperiment:
#     def __init__(self, num_workers, budget, optimizer, rescaling, policy_dim, env_name, num_rollouts,
#                  random_state=None):
#         self.num_workers = num_workers
#         self.budget = budget
#         self.num_rollouts = num_rollouts
#         self.random_state = random_state
#         self.optimizer = optimizer
#         self.policy_dim = policy_dim
#         self.env_name = env_name
#         self.rescaling = rescaling
#
#     def __call__(self):
#         param = ng.p.Array(shape=self.policy_dim).set_mutation(sigma=self.rescaling)
#         optimizer = ng.optimizers.registry[self.optimizer](parametrization=param, budget=self.budget,
#                                                            num_workers=self.num_workers * 2)
#         elapsed_budget = 0
#
#         env = GenericMujocoEnv(self.env_name, self.num_rollouts, self.random_state)
#         while elapsed_budget <= self.budget:
#             x = []
#             for _ in range(self.num_workers):
#                 x.append(optimizer.ask())
#
#             with Pool(self.num_workers) as p:
#                 y = p.map(env, x)
#
#             for _, stats_run in y:
#                 env.global_stats.update(stats_run)
#
#             for u, v in zip(x, y):
#                 print(elapsed_budget, v[0])
#                 optimizer.tell(u, v[0])
#                 elapsed_budget += 1
#                 if elapsed_budget >= self.budget:
#                     break
#
#         with open('mujoco.txt', 'a') as f:
#             f.write(f"{self.num_workers},{self.budget},{self.num_rollouts},{self.random_state},"
#                     f"{self.optimizer},{self.policy_dim},{self.env_name},{self.rescaling}")
#
#
# if __name__ == "__main__":
#     num_workers = 20
#     budget = 10000
#     optimizer = "DiagonalCMA"
#     rescaling = 0.001
#     policy_dim = (8, 111)
#     env_name = "Ant-v2"
#     num_rollouts = 1
#     exp = MujocoExperiment(num_workers, budget, optimizer, rescaling, policy_dim, env_name, num_rollouts)
#     exp()
