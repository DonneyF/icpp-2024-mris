#
# Adversarial input for our algorithm
#

from scheduler.mris import MRIS
from scheduler.pq import PriorityQueue, OnlinePriorityQueue
from scheduler.tetris import TetrisScheduler
from scheduler.bf_exec import BFEXECScheduler
import argparse
import multiprocessing
from tools import DataGenerator
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

M = 1


def schedule(args):
    scheduler, dataset = args
    scheduler.jobs, scheduler.machines = dataset.get_copy()
    scheduler.process()
    return scheduler


def main(N, M, R, run):
    num_big = 1
    num_small = N - num_big

    schedulers = [
        MRIS(sort='WSVF'),
        OnlinePriorityQueue(sort='WSVF'),
        TetrisScheduler(),
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

    dfs = [scheduler.results_as_dataframe() for scheduler in pool_results]
    df = pd.concat(dfs)
    df.to_parquet(Path(__file__).parent / f'results/6_adversarial_{run}.parquet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest="N", type=int, help="Number of jobs", default=2500)
    parser.add_argument('-m', dest='M', type=int, help="Number of machines", default=1)
    parser.add_argument('-r', dest='R', type=int, help="Number of resources", default=3)
    parser.add_argument('--run', dest='run', type=int, help="Run number", default=1)
    args = parser.parse_args()
    main(N=args.N, M=args.M, R=args.R, run=args.run)
