#
# Adversarial input for our algorithm
#

from scheduler.mris import MRIS
from scheduler.pq import PriorityQueueScheduler, OnlinePriorityQueueScheduler
from scheduler.tetris import TetrisScheduler
from scheduler.bf_exec import BFEXECScheduler
import matplotlib.pyplot as plt
from tools import autolabel_h, DataGenerator
import multiprocessing
import gurobipy as gp
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from scheduler.job import Job
import argparse
from tqdm import tqdm
import time
import os

N = 2500
M = 1
R = 3
num_big = 1
num_small = N - num_big


def schedule(args):
    scheduler, dataset = args
    scheduler.jobs, scheduler.machines = dataset.get_copy()
    scheduler.process()
    return scheduler


def main():
    schedulers = [
        MRIS(sort='WSVF'),
        PriorityQueueScheduler(sort='WSJF', online=True),
        OnlinePriorityQueueScheduler(sort='WSJF'),
        TetrisScheduler(score_averaging='instantaneous'),
        BFEXECScheduler()
    ]

    dataset = DataGenerator(N=N, M=M, R=R, p_max=10, r_max=100)

    # Needed for repeated use of TQDM
    for i, scheduler in enumerate(schedulers):
        scheduler.id = i + 1

    for i, job in enumerate(dataset.jobs):
        if i < num_big:
            job.d = np.ones(R) * dataset.num_demand_levels - 0.1
            job.p = 14.
            job.r = 0.01
        else:
            job.p = np.random.ranf() * np.power(2, np.floor(i / N * 10)) / 100
            job.d = np.random.rand(R)
            job.r = 0.1



    args_list = [(scheduler, dataset) for scheduler in schedulers]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), initargs=(multiprocessing.RLock(),),
                                initializer=tqdm.set_lock)
    pool_results = list(pool.imap(schedule, args_list))
    pool.close()
    pool.join()

    # mip = GurobiScheduler()
    # mip.env = gp.Env()
    # mip.jobs, mip.machines = dataset.get_copy()
    # mip_result = mip.process()
    # schedulers = [mip] + schedulers
    # pool_results = [mip_result] + pool_results

    pool_results[0].plot_schedule(cumulative=False)
    # pool_results[2].plot_schedule(cumulative=False)
    # pool_results[-2].plot_schedule(cumulative=False)
    # pool_results[2].plot_schedule(cumulative=True)
    #
    # exit()

    weighted_completion_times = [sum([(job.S + job.p) * job.w for job in scheduler.jobs]) for scheduler in pool_results]
    print([str(scheduler) for scheduler in pool_results])
    weighted_completion_times = [ele / weighted_completion_times[0] for ele in weighted_completion_times]
    makespans = [max([job.S + job.p for job in scheduler.jobs]) for scheduler in pool_results]

    plt.figure(figsize=(10, 5), dpi=200)
    bars = plt.barh([str(scheduler) for scheduler in schedulers], weighted_completion_times)
    # autolabel_h(bars, scientific_notation=True)
    plt.margins(x=0.10)
    plt.title(f'One-Shot Comparison - $N={N}$, $M={M}$, Dataset={dataset.dataset}')
    plt.xlabel(r"Weighted Completion Times $(\sum_j w_jC_j)$")
    plt.ylabel("Scheduler")
    plt.tight_layout()
    # plt.savefig(f'images/experiment9_{dataset.dataset}_{num_jobs}.png')
    plt.show()

    # plt.figure(figsize=(10, 5), dpi=200)
    # bars = plt.barh([str(scheduler) for scheduler in schedulers], makespans)
    # # autolabel_h(bars, scientific_notation=True)
    # plt.margins(x=0.10)
    # plt.title(f'One-Shot Comparison - $N={num_jobs}$, $M={M}$, Dataset={dataset.dataset}')
    # plt.xlabel(r"Makespan $(\max_j\,C_j)$")
    # plt.ylabel("Scheduler")
    # plt.tight_layout()
    # plt.show()

    dfs = [scheduler.results_as_dataframe() for scheduler in pool_results]
    df = pd.concat(dfs)
    df.to_parquet(f'results/adversarial.parquet')



if __name__ == "__main__":
    main()
