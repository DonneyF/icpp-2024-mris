from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
import tqdm

@dataclass
class Scheduler:
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)
    id: int = field(default_factory=count(start=1).__next__)

    def __schedule_earliest_feasible_machine__(self, jobs, pbar=None):
        # An offline scheduling algorithm
        # For each job, compute the earliest start time of a machine that could feasibly schedule the job
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        # Three columns are id, assigned machine, S_j, C_j, and resource demands
        X = np.ones(shape=(len(jobs), 4 + R)) * -1

        def earliest_feasible_start_date(job, machine):
            i = machine.id
            makespan = np.max(X[np.where(X[:, 1] == i)][:, 3], initial=0)
            t = 0
            S = None
            # Scan over the time horizon, while maintaining S that keeps track of earliest feasible start
            while t <= makespan + job.p:
                # Find all the jobs alive at time t on machine i, i.e. S_j <= t < C_j
                alive_jobs = X[np.where((X[:, 1] == i) & (X[:, 2] <= t) & (t < X[:, 3]))]
                total_demand = np.sum(alive_jobs[:, 4:], axis=0)
                if (np.less_equal(total_demand + job.d, machine.D)).all():
                    # t is a feasible start. Ensure the job fits for the entirety of its processing time
                    if S is None:
                        S = t

                    if S + job.p <= t:
                        break
                else:
                    # Could not feasibly schedule job over its processing time. Our candidate is not valid
                    S = None

                # Advance time horizon by the earliest completion time of occupying jobs
                t = min(alive_jobs[:, 3]) if alive_jobs.size > 0 else t + job.p

            return S

        # Schedule jobs via Earliest Feasible Mechanism
        for j, job in enumerate(jobs):
            start_times = np.zeros(shape=len(self.machines))
            for i, machine in enumerate(self.machines):
                start_times[i] = earliest_feasible_start_date(job, machine)

            i = np.argmin(start_times)
            job.S = start_times[i]
            job.i = i
            self.machines[i].add_job(job)
            X[j, 0] = job.id
            X[j, 1] = i
            X[j, 2] = job.S
            X[j, 3] = job.S + job.p
            X[j, 4:] = job.d

            if pbar:
                pbar.update(1)

        return self.jobs

    def __offline_to_online__(self, jobs=None):
        jobs = jobs if jobs is not None else self.jobs
        # Converts this offline schedule to an online schedule simply by waiting for all jobs to arrive
        r_max = max([job.r for job in jobs])
        for job in self.jobs:
            job.S += r_max

        return jobs

    def plot_schedule(self, cumulative=True, max_M=None):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        M_ = max_M if max_M is not None else len(self.machines)

        fig, axs = plt.subplots(nrows=M_, ncols=R, tight_layout=True, dpi=300)

        colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta']

        # axs return value depends on the input rows and cols. Convert to always use 2D ndarray
        if type(axs) is not np.ndarray:
            axs = np.array([[axs]])
        elif axs.ndim == 1:
            axs = axs.reshape(1, -1)

        for i in range(M_):
            machine = self.machines[i]
            # Plot using integer time points for simplicity
            start_times = np.array([job.S for job in machine.jobs])
            completion_times = np.array([job.S + job.p for job in machine.jobs])

            x = np.concatenate([np.zeros(1), start_times, completion_times, start_times+1E-12, completion_times-1E-12])
            x = np.unique(x)

            x = np.sort(x)

            for l in range(R):
                ax = axs[i, l] if R > 1 else axs[l][0]
                cumulative_resource = np.zeros_like(x)

                if not cumulative:
                    stacked_data = []

                    for job in machine.jobs:
                        # Update cumulative resource usage
                        cumulative_resource += np.where((job.S < x) & (job.S + job.p > x), job.d[l], 0)
                        stacked_data.append(np.copy(cumulative_resource))

                    stacked_data.reverse()
                    for j, data in enumerate(stacked_data):
                        ax.stackplot(x, data, color=colors[machine.jobs[len(machine.jobs) - j - 1].id % len(colors)])

                else:
                    for job in machine.jobs:
                        # Update cumulative resource usage
                        cumulative_resource += np.where((job.S < x) & (job.S + job.p > x), job.d[l], 0)

                    ax.stackplot(x, cumulative_resource)

                ax.set_title(f"$i={i + 1}$, $l={l + 1}$")
                ax.set_yticks([0, self.machines[0].D[0]])

        plt.title(str(self))
        plt.show()

    def results_as_dataframe(self):
        df_jobs = pd.DataFrame({
            'scheduler': self.__repr__(),
            'id': [job.id for job in self.jobs],
            'i': [job.i for job in self.jobs],
            'p': [job.p for job in self.jobs],
            'w': [job.w for job in self.jobs],
            'd': [job.d for job in self.jobs],
            'r': [job.r for job in self.jobs],
            'S': [job.S for job in self.jobs],
            'C': [job.S + job.p for job in self.jobs]
        })

        df_jobs.sort_values(by='id', inplace=True)

        return df_jobs