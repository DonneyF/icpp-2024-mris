#
# Sweep the number of machines using a dilution factor
#

from scheduler.pq import PriorityQueueScheduler, OnlinePriorityQueueScheduler
from scheduler.tetris import TetrisScheduler
from scheduler.mris import MRIS, MRISGreedy
from scheduler.bf_exec import BFEXECScheduler
import matplotlib.pyplot as plt
from tools import autolabel_h, DataGenerator
from multiprocessing import Pool, RLock
import pickle
from tqdm import tqdm
from tools import autolabel_h, DataGenerator
import multiprocessing
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os

N = 4_096_000
skip = 64


def schedule(args):
    scheduler, dataset = args
    scheduler.jobs, scheduler.machines = dataset.get_copy()
    scheduler.process()
    return scheduler


def main(N_=None, skip_=None, M=None):
    num_jobs = N_ if N_ else N
    skip_factor = skip_ if skip_ else skip
    schedulers = [
        MRIS(sort='WSJF'),
        PriorityQueueScheduler(sort='WSJF', online=True),
        OnlinePriorityQueueScheduler(sort='WSJF'),
        TetrisScheduler(),
        BFEXECScheduler()
    ]

    dataset = DataGenerator(N=num_jobs, M=M, dataset="azure_packing_2020")

    dataset.jobs = dataset.jobs[::skip_factor]
    num_jobs = len(dataset.jobs)
    # Reindex id
    for i in range(num_jobs):
        dataset.jobs[i].id = i

    # Needed for repeated use of TQDM
    for i, scheduler in enumerate(schedulers):
        scheduler.id = i

    r_max = max([job.r for job in dataset.jobs])
    p_max = max([job.p for job in dataset.jobs])

    print(f"r_max: {r_max:.3f}, p_max: {p_max:.3f}")

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

    # pool_results[0].plot_schedule(cumulative=True)
    # pool_results[1].plot_schedule(cumulative=True)
    # pool_results[2].plot_schedule(cumulative=True)
    #
    # exit()

    # weighted_completion_times = [sum([(job.S + job.p) * job.w for job in scheduler.jobs]) for scheduler in pool_results]
    # print([str(scheduler) for scheduler in pool_results])
    # weighted_completion_times = [ele / weighted_completion_times[0] for ele in weighted_completion_times]
    # makespans = [max([job.S + job.p for job in scheduler.jobs]) for scheduler in pool_results]

    # plt.figure(figsize=(10, 5), dpi=200)
    # bars = plt.barh([str(scheduler) for scheduler in schedulers], weighted_completion_times)
    # # autolabel_h(bars, scientific_notation=True)
    # plt.margins(x=0.10)
    # plt.title(f'One-Shot Comparison - $N={num_jobs}$, $M={M}$, Dataset={dataset.dataset}')
    # plt.xlabel(r"Weighted Completion Times $(\sum_j w_jC_j)$")
    # plt.ylabel("Scheduler")
    # plt.tight_layout()
    # # plt.savefig(f'images/experiment9_{dataset.dataset}_{num_jobs}.png')
    # plt.show()

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
    df.to_parquet(f'results/machine_sweep_azure_packing_2020_{M}_{num_jobs}.parquet')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', dest="N", type=int, help='Number of Jobs to run')
    args = parser.parse_args()
    machines = [5, 10, 20, 40, 80]
    for j in range(12, -1, -1):
        for i in machines:
            main(skip_=int(np.power(2, j)), M=i)
            os.system('cls' if os.name == 'nt' else 'clear')
