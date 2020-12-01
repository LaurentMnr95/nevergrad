import os
import numpy as np
import uuid
import submitit
from pathlib import Path
from mujoco_online import *

cases = [("Ant-v2", (8, 111), 0.001), ("Humanoid-v2", (17, 376), 0.0001)]
budgets = [1000, 10000, 100000, 500000]
optimizers = ["DiagonalCMA", "NGOpt8"]

executor = submitit.AutoExecutor(folder=f"logs")
executor.update_parameters(
    gpus_per_node=1,
    tasks_per_node=1,  # one task per GPU
    cpus_per_task=10,
    nodes=1,
    # Below are cluster dependent parameters
    slurm_partition="dev",
    # slurm_comment=comment,
    timeout_min=4320,
    mem_gb=256,
    name="mujoco_online",
    slurm_array_parallelism=100
)

jobs = []
with executor.batch():
    for optimizer in optimizers:
        for budget in budgets:
            for case in cases:
                exp = MujocoExperiment(num_workers=1, budget=budget, optimizer=optimizer, rescaling=case[2],
                                       policy_dim=case[1],
                                       env_name=case[0], num_rollouts=1)
                job = executor.submit(exp)
                jobs.append(job)

    print(f"Number of jobs = {len(jobs)}")
