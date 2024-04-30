#
# One-Shot comparison of all schedulers
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

N = 1600
M = 2
R = 2
p_max = 2000
r_max = 1000
skip_factor = 256


def schedule(scheduler):
    scheduler.process()
    return scheduler


def main():
    schedulers = [
        MRIS(sort='WSVF'),
        PriorityQueueScheduler(sort='WSVF'),
        OnlinePriorityQueueScheduler(sort='WSVF'),
        TetrisScheduler(),
        BFEXECScheduler(),
    ]

    dataset = DataGenerator(N=N, M=M, R=R, p_max=p_max, r_max=r_max, integer=False)

    # dataset = DataGenerator(N=1_280_000, M=M, dataset="azure_packing_2020")
    #
    # dataset.jobs = dataset.jobs[::skip_factor]
    # num_jobs = len(dataset.jobs)
    # # Reindex id
    # for i in range(num_jobs):
    #     dataset.jobs[i].id = i
    #
    # # Needed for repeated use of TQDM
    # for i, scheduler in enumerate(schedulers):
    #     scheduler.id = i + 1
    #
    with open('test.dat', 'wb') as f:
        pickle.dump(dataset, f)
        # dataset = pickle.load(f)

    for scheduler in schedulers:
        scheduler.jobs, scheduler.machines = dataset.get_copy()

    pool = Pool(initargs=(RLock(),), initializer=tqdm.set_lock)
    pool_results = list(pool.imap(schedule, schedulers))
    pool.close()
    pool.join()

    pool_results[0].plot_schedule(cumulative=True)
    pool_results[1].plot_schedule(cumulative=True)
    pool_results[2].plot_schedule(cumulative=True)
    # for scheduler in pool_results:
    #     print(scheduler, [(job.id, job.S) for job in sorted(scheduler.jobs, key=lambda x: x.S)])
    # pool_results[2].plot_schedule(cumulative=True)

    weighted_completion_times = [sum([(job.S + job.p) * job.w for job in scheduler.jobs]) / N for scheduler in pool_results]
    makespans = [max([job.S + job.p for job in scheduler.jobs]) for scheduler in pool_results]
    print(weighted_completion_times)

    plt.figure(figsize=(10, 5), dpi=200)
    bars = plt.barh([str(scheduler) for scheduler in schedulers], weighted_completion_times)
    autolabel_h(bars, scientific_notation=True)
    plt.margins(x=0.10)
    plt.title(f'One-Shot Comparison - $N={N}$, $M={M}$, Dataset={dataset.dataset}')
    plt.xlabel(r"Average Weighted Completion Time $(\sum_j w_jC_j)$")
    plt.ylabel("Scheduler")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5), dpi=200)
    bars = plt.barh([str(scheduler) for scheduler in schedulers], makespans)
    # autolabel_h(bars, scientific_notation=True)
    plt.margins(x=0.10)
    plt.title(f'One-Shot Comparison - $N={N}$, $M={M}$, Dataset={dataset.dataset}')
    plt.xlabel(r"Makespan $(\max_j C_j)$")
    plt.ylabel("Scheduler")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
