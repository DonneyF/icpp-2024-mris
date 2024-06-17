#
# Compare the algorithms by sweeping the number of resources
#

from scheduler.mris import MRIS
from scheduler.tetris import TetrisScheduler
from scheduler.pq import PriorityQueue, OnlinePriorityQueue
from scheduler.bf_exec import BFEXECScheduler
from tools import DataGenerator
import multiprocessing
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

N = 4_096_000
M = 20


def schedule(args):
    scheduler, dataset = args
    scheduler.jobs, scheduler.machines = dataset.get_copy()
    scheduler.process()
    return scheduler


def main(downsample_factor, downsample_offset, run, R):
    schedulers = [
        MRIS(sort='WSJF'),
        PriorityQueue(sort='WSJF', collect=True),
        OnlinePriorityQueue(sort='WSJF'),
        TetrisScheduler(),
        BFEXECScheduler()
    ]

    dataset = DataGenerator(M=M, R=R, dataset="azure_packing_2020")

    dataset.jobs = dataset.jobs[downsample_offset::downsample_factor][:N // downsample_factor]
    num_jobs = len(dataset.jobs)
    # Reindex id
    for i in range(num_jobs):
        dataset.jobs[i].id = i

    # Needed for repeated use of TQDM
    for i, scheduler in enumerate(schedulers):
        scheduler.id = i + 1

    args_list = [(scheduler, dataset) for scheduler in schedulers]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), initargs=(multiprocessing.RLock(),),
                                initializer=tqdm.set_lock)
    pool_results = list(pool.imap(schedule, args_list))
    pool.close()
    pool.join()

    dfs = [scheduler.results_as_dataframe() for scheduler in pool_results]
    df = pd.concat(dfs)
    df.to_parquet(Path(__file__).parent / f'results/5_scheduler_benchmark_resources_{R}_{run}.parquet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample_factor', dest="downsample_factor", type=int, help='Downsampling factor', default=512)
    parser.add_argument('--downsample_offset', dest="downsample_offset", type=int, help="Offset before downsampling.", default=10)
    parser.add_argument('--run', dest='run', type=int, help="Run number")
    parser.add_argument('-r', dest='R', type=int, help="Number of resources", default=4)
    args = parser.parse_args()
    main(downsample_factor=args.downsample_factor, downsample_offset=args.downsample_offset, run=args.run, R=args.R)
