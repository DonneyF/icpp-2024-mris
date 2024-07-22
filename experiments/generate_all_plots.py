import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
from pathlib import Path
import itertools
import scipy.stats as st
import argparse
import shutil

PATH_TO_RESULTS = Path(__file__).absolute().parent / "results"

def main():
    global PATH_TO_RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(__file__).absolute().parent / "results",
        help="Path to the results directory",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs executed for each run",
    )
    p = parser.parse_args()
    PATH_TO_RESULTS = p.results_dir
    NUM_RUNS = p.num_runs

    MRIS_herustics(
        N=4_096_000,
        DOWNSAMPLE_FACTORS=np.array([1024, 512, 256, 128, 64]),
        NUM_RUNS=NUM_RUNS
    )
    MRIS_knapsack(
        N=4_096_000,
        DOWNSAMPLE_FACTORS=np.array([1024, 512, 256, 128, 64]),
        NUM_RUNS=NUM_RUNS
    )
    scheduler_benchmark_jobs(
        N=4_096_000,
        DOWNSAMPLE_FACTORS=np.array([512, 256, 128, 64, 32, 16]),
        NUM_RUNS=NUM_RUNS
    )
    scheduler_benchmark_machines(
        MACHINES=np.array([5, 10, 20, 40]),
        NUM_RUNS=NUM_RUNS
    )
    scheduler_benchmark_resources(
        RESOURCES=np.arange(4, 20 + 1),
        NUM_RUNS=NUM_RUNS
    )
    adversarial(
        R=3,
        N=2500,
        RUN_NUMBER=1
    )

if shutil.which('latex'):
    plt.style.use(['science'])
    params = {
        "font.family": "serif",
        "text.usetex": True,
        'text.latex.preamble':
            r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
            """,
    }
else:
    plt.style.use(['science', 'no-latex'])
    params = {
        "font.family": "serif",
    }


# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
def get_linestyles():
    return itertools.cycle([
        ('solid', (0, ())),
        # ('loosely dotted',        (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        # ('densely dotted',        (0, (1, 1))),
        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed', (0, (5, 10))),
        # ('dashed',                (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        # ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ])


mpl.rcParams.update(params)

def MRIS_herustics(N, DOWNSAMPLE_FACTORS, NUM_RUNS):
    N_arr = (N / DOWNSAMPLE_FACTORS).astype(int)

    awct_data = {}
    for i, downsample_factor in enumerate(DOWNSAMPLE_FACTORS):
        for run in range(1, NUM_RUNS + 1):
            data = pd.read_parquet(PATH_TO_RESULTS / Path(f'1_MRIS_heuristics_{downsample_factor}_{run}.parquet'))
            schedulers = list(data['scheduler'].unique())
            for scheduler in schedulers:
                df = data[data['scheduler'] == scheduler]
                weighted_completion_time = df['C'].dot(df['w']) / len(df)
                vals = awct_data.get(scheduler, np.zeros(shape=(len(DOWNSAMPLE_FACTORS), NUM_RUNS)).astype(float))
                vals[i][run - 1] = weighted_completion_time
                awct_data[scheduler] = vals

    awct_x = {scheduler: np.mean(data, axis=1) for scheduler, data in awct_data.items()}
    # 95 % Confidence Intervals
    awct_error = {
        scheduler: st.t.interval(0.95, df=data.shape[1] - 1, loc=np.mean(data, axis=1), scale=st.sem(data, axis=1)) for
        scheduler, data in awct_data.items()}

    plt.figure(figsize=(6, 2), dpi=300)

    linestyles = get_linestyles()
    for scheduler, weighted_completion_times in awct_x.items():
        plt.plot(N_arr, weighted_completion_times, label=scheduler, linestyle=next(linestyles)[1])
        plt.fill_between(N_arr, awct_error[scheduler][0], awct_error[scheduler][1], alpha=0.25)
    plt.xscale('log')
    plt.xticks(N_arr, N_arr)
    # plt.yscale('log', base=2)
    plt.ylabel("Average weighted\n completion time")
    plt.xlabel(r"Number of jobs released in 12.5 days")
    plt.legend(prop={'size': 8}, handlelength=3, ncol=1)
    plt.savefig('1_MRIS_heuristics.png', bbox_inches='tight')


def MRIS_knapsack(N, DOWNSAMPLE_FACTORS, NUM_RUNS):
    N_arr = (N / DOWNSAMPLE_FACTORS).astype(int)

    awct_data = {}
    for i, downsample_factor in enumerate(DOWNSAMPLE_FACTORS):
        for run in range(1, NUM_RUNS + 1):
            data = pd.read_parquet(PATH_TO_RESULTS / Path(f'2_MRIS_knapsack_{downsample_factor}_{run}.parquet'))
            schedulers = list(data['scheduler'].unique())
            for scheduler in schedulers:
                df = data[data['scheduler'] == scheduler]
                weighted_completion_time = df['C'].dot(df['w']) / len(df)
                vals = awct_data.get(scheduler, np.zeros(shape=(len(DOWNSAMPLE_FACTORS), NUM_RUNS)).astype(float))
                vals[i][run - 1] = weighted_completion_time
                awct_data[scheduler] = vals

    awct_x = {scheduler: np.mean(data, axis=1) for scheduler, data in awct_data.items()}
    # 95 % Confidence Intervals
    awct_error = {
        scheduler: st.t.interval(0.95, df=data.shape[1] - 1, loc=np.mean(data, axis=1), scale=st.sem(data, axis=1)) for
        scheduler, data in awct_data.items()}

    plt.figure(figsize=(6, 1.5), dpi=300)

    linestyles = get_linestyles()
    for scheduler, weighted_completion_times in awct_x.items():
        plt.plot(N_arr, weighted_completion_times, label=scheduler, linestyle=next(linestyles)[1])
        plt.fill_between(N_arr, awct_error[scheduler][0], awct_error[scheduler][1], alpha=0.25)
    plt.xscale('log')
    plt.xticks(N_arr, N_arr)
    # plt.yscale('log', base=2)
    plt.ylabel("Average weighted\n completion time")
    plt.xlabel(r"Number of jobs released in 12.5 days")
    plt.legend(prop={'size': 8}, handlelength=3, ncol=1)
    plt.savefig('2_MRIS_knapsack.png', bbox_inches='tight')


def scheduler_benchmark_jobs(N, DOWNSAMPLE_FACTORS, NUM_RUNS):
    N_arr = (N / DOWNSAMPLE_FACTORS).astype(int)  # The number of jobs processed

    awct_data = {}
    for i, downsample_factor in enumerate(DOWNSAMPLE_FACTORS):
        for run in range(1, NUM_RUNS + 1):
            data = pd.read_parquet(
                PATH_TO_RESULTS / Path(f'3_scheduler_benchmark_jobs_{downsample_factor}_{run}.parquet'))
            schedulers = list(data['scheduler'].unique())
            for scheduler in schedulers:
                df = data[data['scheduler'] == scheduler]
                weighted_completion_time = df['C'].dot(df['w']) / len(df)
                vals = awct_data.get(scheduler, np.zeros(shape=(len(DOWNSAMPLE_FACTORS), NUM_RUNS)).astype(float))
                vals[i][run - 1] = weighted_completion_time
                awct_data[scheduler] = vals

    awct_x = {scheduler: np.mean(data, axis=1) for scheduler, data in awct_data.items()}
    # 95 % Confidence Intervals
    awct_error = {
        scheduler: st.t.interval(0.95, df=data.shape[1] - 1, loc=np.mean(data, axis=1), scale=st.sem(data, axis=1)) for
        scheduler, data in awct_data.items()}

    plt.figure(figsize=(6, 2), dpi=200)

    linestyles = get_linestyles()
    for scheduler, weighted_completion_times in awct_x.items():
        if scheduler == 'Tetris-instantaneous':
            label = r'\textsc{Tetris}'
        elif scheduler == 'PQ-WSJF':
            label = 'CA-PQ-WSJF'
        elif scheduler == 'OnlinePQ-WSJF':
            label = 'PQ-WSJF'
        else:
            label = scheduler
        plt.plot(N_arr, weighted_completion_times, label=label, linestyle=next(linestyles)[1])
        plt.fill_between(N_arr, awct_error[scheduler][0], awct_error[scheduler][1], alpha=0.25)
    plt.xscale('log')
    plt.xticks(N_arr, N_arr)
    # plt.yscale('log', base=2)
    plt.ylabel("Average weighted\n completion time")
    plt.xlabel(r"Number of jobs released in 12.5 days")
    plt.legend(prop={'size': 8}, handlelength=3, ncol=2)
    plt.savefig('3_scheduler_benchmark_jobs.png', bbox_inches='tight')

    # Queuing Delay
    N = N_arr[-3]
    print(f'N={N}')
    queuing_data = {scheduler: np.zeros(shape=(NUM_RUNS, N)).astype(float) for scheduler in schedulers}
    downsample_factor = 4_096_000 // N
    for run in range(1, NUM_RUNS + 1):
        data = pd.read_parquet(PATH_TO_RESULTS / Path(f'3_scheduler_benchmark_jobs_{downsample_factor}_{run}.parquet'))
        schedulers = list(data['scheduler'].unique())
        for scheduler in schedulers:
            df = data[data['scheduler'] == scheduler]
            queuing_delay = df['S'] - df['r']
            queuing_data[scheduler][run - 1:] = np.sort(queuing_delay)

    queuing_error = {
        scheduler: st.t.interval(0.95, df=data.shape[0] - 1, loc=np.mean(data, axis=0), scale=st.sem(data, axis=0)) for
        scheduler, data in queuing_data.items()}

    plt.figure(figsize=(6, 2), dpi=200)
    plt.xscale('symlog')
    linestyles = get_linestyles()
    for scheduler, queuing_delay in queuing_data.items():
        if scheduler == 'Tetris-instantaneous':
            label = r'\textsc{Tetris}'
        elif scheduler == 'PQ-WSJF':
            label = 'CA-PQ-WSJF'
        elif scheduler == 'OnlinePQ-WSJF':
            label = 'PQ-WSJF'
        else:
            label = scheduler
        cdf_2d = np.arange(1, N + 1) / N
        plt.plot(np.mean(queuing_delay, axis=0), cdf_2d, label=label, linestyle=next(linestyles)[1])
        plt.fill_betweenx(cdf_2d, queuing_error[scheduler][0], queuing_error[scheduler][1], alpha=0.25, rasterized=True)

    plt.legend(prop={'size': 8}, handlelength=3)
    # plt.title(f'CDF of Queuing Delay of Jobs, $M=20, R=4, N={N}$')
    plt.ylabel('CDF')
    plt.xlabel('Queuing delay (Days)')
    plt.savefig('3_scheduler_benchmark_jobs_queuing_delay.png', bbox_inches='tight')

def scheduler_benchmark_machines(MACHINES, NUM_RUNS):
    awct_data = {}
    for i, M in enumerate(MACHINES):
        for run in range(1, NUM_RUNS + 1):
            data = pd.read_parquet(PATH_TO_RESULTS / Path(f'4_scheduler_benchmark_machines_{M}_{run}.parquet'))
            schedulers = list(data['scheduler'].unique())
            for scheduler in schedulers:
                df = data[data['scheduler'] == scheduler]
                weighted_completion_time = df['C'].dot(df['w']) / len(df)
                vals = awct_data.get(scheduler, np.zeros(shape=(len(MACHINES), NUM_RUNS)).astype(float))
                vals[i][run - 1] = weighted_completion_time
                awct_data[scheduler] = vals

    awct_x = {scheduler: np.mean(data, axis=1) for scheduler, data in awct_data.items()}
    # 95 % Confidence Intervals
    awct_error = {
        scheduler: st.t.interval(0.95, df=data.shape[1] - 1, loc=np.mean(data, axis=1), scale=st.sem(data, axis=1)) for
        scheduler, data in awct_data.items()}

    plt.figure(figsize=(6, 2), dpi=200)

    linestyles = get_linestyles()
    for scheduler, weighted_completion_times in awct_x.items():
        if scheduler == 'Tetris-instantaneous':
            label = r'\textsc{Tetris}'
        elif scheduler == 'PQ-WSJF':
            label = 'CA-PQ-WSJF'
        elif scheduler == 'OnlinePQ-WSJF':
            label = 'PQ-WSJF'
        else:
            label = scheduler
        plt.plot(MACHINES, weighted_completion_times, label=label, linestyle=next(linestyles)[1])
        plt.fill_between(MACHINES, awct_error[scheduler][0], awct_error[scheduler][1], alpha=0.25)
    plt.xticks(MACHINES, MACHINES)
    # plt.yscale('log', base=2)
    plt.ylabel("Average weighted\n completion time")
    plt.xlabel(r"Number of machines")
    plt.legend(prop={'size': 8}, handlelength=3, ncol=2)
    plt.savefig('4_scheduler_benchmark_machines.png', bbox_inches='tight')

def scheduler_benchmark_resources(RESOURCES, NUM_RUNS):
    awct_data = {}
    for i, R in enumerate(RESOURCES):
        for run in range(1, NUM_RUNS + 1):
            data = pd.read_parquet(PATH_TO_RESULTS / Path(f'5_scheduler_benchmark_resources_{R}_{run}.parquet'))
            schedulers = list(data['scheduler'].unique())
            for scheduler in schedulers:
                df = data[data['scheduler'] == scheduler]
                weighted_completion_time = df['C'].dot(df['w']) / len(df)
                vals = awct_data.get(scheduler, np.zeros(shape=(len(RESOURCES), NUM_RUNS)).astype(float))
                vals[i][run - 1] = weighted_completion_time
                awct_data[scheduler] = vals

    awct_x = {scheduler: np.mean(data, axis=1) for scheduler, data in awct_data.items()}
    # 95 % Confidence Intervals
    awct_error = {
        scheduler: st.t.interval(0.95, df=data.shape[1] - 1, loc=np.mean(data, axis=1), scale=st.sem(data, axis=1)) for
        scheduler, data in awct_data.items()}

    plt.figure(figsize=(6, 2), dpi=200)

    linestyles = get_linestyles()
    for scheduler, weighted_completion_times in awct_x.items():
        if scheduler == 'Tetris-instantaneous':
            label = r'\textsc{Tetris}'
        elif scheduler == 'PQ-WSJF':
            label = 'CA-PQ-WSJF'
        elif scheduler == 'OnlinePQ-WSJF':
            label = 'PQ-WSJF'
        else:
            label = scheduler
        plt.plot(RESOURCES, weighted_completion_times, label=label, linestyle=next(linestyles)[1])
        plt.fill_between(RESOURCES, awct_error[scheduler][0], awct_error[scheduler][1], alpha=0.25)
    plt.xticks(RESOURCES, RESOURCES)
    # plt.yscale('log', base=2)
    plt.ylabel("Average weighted\n completion time")
    plt.xlabel(r"Number of resource types")
    plt.legend(prop={'size': 8}, handlelength=3, ncol=2)
    plt.savefig('5_scheduler_benchmark_resources.png', bbox_inches='tight')

def adversarial(R, N, RUN_NUMBER):
    df = pd.read_parquet(PATH_TO_RESULTS / Path(f'6_adversarial_{RUN_NUMBER}.parquet'))
    schedulers = df['scheduler'].unique().tolist()

    resource_idx = 0
    demand_levels = 10
    fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True, dpi=300)
    fig.set_figheight(3)
    fig.set_figwidth(6)

    # axs return value depends on the input rows and cols. Convert to always use 2D ndarray
    if type(axs) is not np.ndarray:
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        axs = axs.reshape(1, -1)

    axes = axs.flat

    for s, scheduler in enumerate(schedulers):
        if scheduler == 'Tetris-instantaneous':
            label = r'\textsc{Tetris}'
        elif scheduler == 'PQ-WSVF':
            label = 'CA-PQ-WSVF'
        elif scheduler == 'OnlinePQ-WSVF':
            label = 'PQ-WSVF'
        else:
            label = scheduler

        df_ = df[df['scheduler'] == scheduler]
        start_times = df_['S'].to_numpy()
        completion_times = df_['C'].to_numpy()
        demands = df_['d'].to_numpy()

        x = np.concatenate([np.zeros(1), start_times, completion_times, start_times + 1E-12, completion_times - 1E-12])
        x = np.unique(x)

        x = np.sort(x)
        cumulative_resource = np.zeros_like(x)

        stacked_data = []

        for i in range(N):
            cumulative_resource += np.where((start_times[i] < x) & (completion_times[i] > x), demands[i][resource_idx],
                                            0) / demand_levels
            # cumulative_resource += np.where(((start_times[i]) <= x) & (completion_times[i] > x), demands[i][resource_idx], 0) / demand_levels
            stacked_data.append(np.copy(cumulative_resource))
        stacked_data.reverse()
        for j, data in enumerate(stacked_data):
            axes[s].stackplot(x, data, rasterized=True)
            axes[s].set_ylim(0, 1.1)
            axes[s].set_title(label)

    fig.text(0.5, -0.00, 'Time [arb. units]', ha='center')
    fig.text(-0.01, 0.5, 'Resource usage [arb. units]', va='center', rotation='vertical')
    plt.savefig('6_adversarial.png', bbox_inches='tight')

    awct_data = {}
    for scheduler in schedulers:
        df_ = df[df['scheduler'] == scheduler]
        awct = df_['C'].dot(df_['w']) / len(df)
        awct_data[scheduler] = awct

    plt.figure(figsize=(10, 5), dpi=200)
    bars = plt.barh([str(scheduler) for scheduler in schedulers], awct_data.values(), align='center')
    # autolabel_h(bars, scientific_notation=True)
    plt.margins(x=0.10)
    plt.title(f'One-Shot Comparison')
    plt.xlabel(r"Average Weighted Completion Time")
    plt.ylabel("Scheduler")
    plt.tight_layout()
    plt.savefig('6_adversarial_metrics.png', bbox_inches='tight')

if __name__ == '__main__':
    main()