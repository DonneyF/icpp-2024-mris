from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.scheduler import Scheduler
from typing import List
import numpy as np
from tqdm import tqdm
import heapq
import bisect
import time

@dataclass
class PriorityQueueScheduler(Scheduler):
    sort: str = None
    online: bool = False  # Wait for all jobs to arrive, then schedule them offline.
    pbar: tqdm = None

    def __repr__(self):
        return f"PQ-{self.sort.upper()}" if self.sort else 'PQ'

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        num_processed_jobs = 0

        unscheduled_jobs = self.jobs.copy()
        unscheduled_job_demands = np.fromiter((job.d for job in self.jobs), dtype=np.dtype((type(self.jobs[0].d), R)), count=len(self.jobs))
        if R == 1:
            unscheduled_job_demands = np.atleast_2d(unscheduled_job_demands).T
        machine_resources = np.zeros(shape=(len(self.machines), R))
        alive_jobs = {i: [] for i in range(len(self.machines))}

        unscheduled_job_heuristics = None
        # Compute the heuristic for each job without sorting by ascending order

        match self.sort:
            case "SJF":
                unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs), np.dtype(type(self.jobs[0].p)), len(self.jobs))
                unscheduled_job_heuristics = unscheduled_job_processing_times
            case "SVF":
                unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs), np.dtype(type(self.jobs[0].p)), len(self.jobs))
                unscheduled_job_heuristics = unscheduled_job_processing_times * np.sum(unscheduled_job_demands, axis=1)
            case "SDF":
                unscheduled_job_heuristics = np.sum(unscheduled_job_demands, axis=1)
            case "WSJF":
                unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs), np.dtype(type(self.jobs[0].p)), len(self.jobs))
                unscheduled_job_weights = np.fromiter((job.w for job in self.jobs), np.dtype(type(self.jobs[0].w)), len(self.jobs))
                unscheduled_job_heuristics = unscheduled_job_processing_times / unscheduled_job_weights
            case "WSVF":
                unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs), np.dtype(type(self.jobs[0].p)), len(self.jobs))
                unscheduled_job_weights = np.fromiter((job.w for job in self.jobs), np.dtype(type(self.jobs[0].w)), len(self.jobs))
                unscheduled_job_heuristics = unscheduled_job_processing_times * np.sum(unscheduled_job_demands, axis=1) / unscheduled_job_weights
            case "WSDF":
                unscheduled_job_weights = np.fromiter((job.w for job in self.jobs), np.dtype(type(self.jobs[0].w)), len(self.jobs))
                unscheduled_job_heuristics = np.sum(unscheduled_job_demands, axis=1) / unscheduled_job_weights
            case _:
                pass

        if self.sort:
            # Sort the jobs and demands
            job_sorted_idx = unscheduled_job_heuristics.argsort()
            unscheduled_jobs = [unscheduled_jobs[i] for i in job_sorted_idx]
            unscheduled_job_demands = unscheduled_job_demands[job_sorted_idx]

        t = np.zeros(len(self.machines))

        if self.pbar is None:
            pbar = tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        else:
            pbar = self.pbar
        while num_processed_jobs != len(self.jobs):
            i = np.argmin(t)
            machine = self.machines[i]

            # For each machine, if a job can fit on the machine at time t[i], go ahead and schedule it

            # Property of List Scheduling: Scheduled jobs have start times at or before t[i].
            # Therefore, we only need to check the current resource usage at time t[i]

            total_demand = machine_resources[i, :]

            # List Scheduling requires us to always scan from the start of the list, even after identifying a job to schedule
            result = np.atleast_2d(np.less_equal(total_demand + unscheduled_job_demands, machine.D)).all(axis=1)

            scheduled_jobs_idx = []
            for j in np.where(result == True)[0]:
                job = unscheduled_jobs[j]
                if (np.less_equal(total_demand + job.d, machine.D)).all():
                    if job.id == 18:
                        pass
                    # Set the first time we can feasible schedule
                    job.S = t[i]
                    job.i = i
                    machine.add_job(job)
                    num_processed_jobs += 1

                    heapq.heappush(alive_jobs[i], (job.S + job.p, job.id, job))

                    scheduled_jobs_idx.append(j)

                    total_demand += job.d  # Update the demand since we've scheduled it to start at this time
                    machine_resources[i, :] = total_demand

            for idx_ in sorted(scheduled_jobs_idx, reverse=True):
                unscheduled_jobs.pop(idx_)

            if scheduled_jobs_idx:
                pbar.update(len(scheduled_jobs_idx))
                unscheduled_job_demands = np.delete(unscheduled_job_demands, scheduled_jobs_idx, axis=0)

            # Advance time to the next time a job finishes
            (min_completion_time, _, job) = heapq.heappop(alive_jobs[i])
            t[i] = min_completion_time
            machine_resources[i, :] -= job.d

            # One or more jobs finished at min_completion_time
            while True:
                if len(alive_jobs[i]) >= 1:
                    (t_, _, job) = alive_jobs[i][0]
                    if np.isclose(min_completion_time, t_):
                        heapq.heappop(alive_jobs[i])
                        machine_resources[i, :] -= job.d
                    else:
                        break
                else:
                    break

        if self.online:
            self.jobs = self.__offline_to_online__(self.jobs)

        return self.jobs


@dataclass
class OnlinePriorityQueueScheduler(Scheduler):
    # Same as offline list scheduling, but job start times must respect release dates
    sort: str = None

    def __repr__(self):
        return f"OnlinePQ-{self.sort.upper()}" if self.sort else 'OnlinePQ'

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        num_processed_jobs = 0

        unscheduled_jobs = self.jobs.copy()
        unscheduled_release_dates = np.fromiter((job.r for job in self.jobs), dtype=np.dtype(type(self.jobs[0].r)), count=len(self.jobs))
        unscheduled_job_demands = np.fromiter((job.d for job in self.jobs), dtype=np.dtype((type(self.jobs[0].d), R)), count=len(self.jobs))
        unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs), dtype=np.dtype(type(self.jobs[0].p)), count=len(self.jobs))
        unscheduled_job_weights = np.fromiter((job.w for job in self.jobs), dtype=np.dtype(type(self.jobs[0].w)), count=len(self.jobs))

        max_release_date = np.max(unscheduled_release_dates)

        machine_resources = np.zeros(shape=(len(self.machines), R))
        alive_jobs = {i: [] for i in range(len(self.machines))}

        unscheduled_job_heuristics = None
        # Compute the heuristic for each job without sorting by ascending order
        match self.sort:
            case "ERF":
                unscheduled_job_heuristics = unscheduled_release_dates
            case "SJF":
                unscheduled_job_heuristics = unscheduled_job_processing_times
            case "SVF":
                unscheduled_job_heuristics = unscheduled_job_processing_times * np.sum(unscheduled_job_demands, axis=1)
            case "WERF":
                unscheduled_job_heuristics = unscheduled_release_dates / unscheduled_job_weights
            case "SDF":
                unscheduled_job_heuristics = np.sum(unscheduled_job_demands, axis=1)
            case "WSJF":
                unscheduled_job_heuristics = unscheduled_job_processing_times / unscheduled_job_weights
            case "WSVF":
                unscheduled_job_heuristics = unscheduled_job_processing_times * np.sum(unscheduled_job_demands, axis=1) / unscheduled_job_weights
            case _:
                unscheduled_job_heuristics = unscheduled_release_dates

        unscheduled_job_heuristics_idx = np.argsort(unscheduled_job_heuristics)

        t = np.ones(len(self.machines)) * min(unscheduled_release_dates)
        pbar = tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        while num_processed_jobs != len(self.jobs):
            i = np.argmin(t)
            machine = self.machines[i]

            idx = bisect.bisect_right(unscheduled_release_dates, t[i])
            arrived_jobs = unscheduled_jobs[:idx]
            unscheduled_release_dates_after_t = unscheduled_release_dates[idx:]

            arrived_job_sorted_idx_idx = np.where(unscheduled_job_heuristics_idx < idx)[0]  # Index of index array
            arrived_job_sorted_idx = unscheduled_job_heuristics_idx[arrived_job_sorted_idx_idx]

            # For each machine, if a job can fit on the machine at time t[i], go ahead and schedule it

            # Property of List Scheduling: Scheduled jobs have start times at or before t[i].
            # Therefore, we only need to check the current resource usage at time t[i]

            total_demand = machine_resources[i, :]

            # List Scheduling requires us to always scan from the start of the list, even after identifying a job to schedule
            arrived_job_demands = unscheduled_job_demands[arrived_job_sorted_idx]
            result = (np.less_equal(total_demand + arrived_job_demands, machine.D)).all(axis=1)

            scheduled_jobs_idx = []
            for j in np.where(result == True)[0]:
                job = arrived_jobs[arrived_job_sorted_idx[j]]
                if (np.less_equal(total_demand + job.d, machine.D)).all():
                    # Set the first time we can feasible schedule
                    job.S = t[i]
                    job.i = i
                    machine.add_job(job)
                    num_processed_jobs += 1

                    heapq.heappush(alive_jobs[i], (job.S + job.p, job.id, job))

                    scheduled_jobs_idx.append(arrived_job_sorted_idx[j])

                    total_demand += job.d  # Update the demand since we've scheduled it to start at this time
                    machine_resources[i, :] = total_demand

            sorted_scheduled_jobs_idx = sorted(scheduled_jobs_idx, reverse=True)
            for idx_ in sorted_scheduled_jobs_idx:
                unscheduled_jobs.pop(idx_)

            if scheduled_jobs_idx:
                pbar.update(len(scheduled_jobs_idx))
                unscheduled_job_demands = np.delete(unscheduled_job_demands, scheduled_jobs_idx, axis=0)
                unscheduled_release_dates = np.delete(unscheduled_release_dates, scheduled_jobs_idx)

                # Update the unscheduled job heuristics to ensure we maintain argsorted order at every time instance
                to_null_idx = np.where(np.isin(unscheduled_job_heuristics_idx, scheduled_jobs_idx))
                unscheduled_job_heuristics_idx[to_null_idx] = -1
                for idx_ in sorted_scheduled_jobs_idx:
                    unscheduled_job_heuristics_idx[unscheduled_job_heuristics_idx > idx_] -= 1
                unscheduled_job_heuristics_idx = np.delete(unscheduled_job_heuristics_idx, to_null_idx)

            # Advance time to the next time a job releases or finishes
            min_release_date = min(unscheduled_release_dates_after_t) if unscheduled_release_dates_after_t.size != 0 else np.inf

            # Find the earliest time when a job could be feasibly scheduled at this current time instance
            if t[i] < max_release_date:
                result = (np.less_equal(total_demand + unscheduled_job_demands, machine.D)).all(axis=1)
                idx_of_first_feasible_job = np.argmax(result) if np.any(result) else -1
                release_time_of_first_feasible_job = unscheduled_release_dates[idx_of_first_feasible_job] if idx_of_first_feasible_job > 0 else -1

            if alive_jobs[i]:
                (min_completion_time, _, _) = alive_jobs[i][0]
            else:
                min_completion_time = np.inf

            if min_completion_time < min_release_date or (t[i] < max_release_date and idx_of_first_feasible_job == -1):
                t[i] = min_completion_time

                # One or more jobs finished at min_completion_time
                while True:
                    if len(alive_jobs[i]) >= 1:
                        (t_, _, job) = alive_jobs[i][0]
                        if min_completion_time == t_:
                            heapq.heappop(alive_jobs[i])
                            machine_resources[i, :] -= job.d
                        else:
                            break
                    else:
                        break
            elif t[i] < max_release_date and min_release_date < release_time_of_first_feasible_job < min_completion_time:
                # Skip all the infeasible jobs
                t[i] = release_time_of_first_feasible_job
            else:
                t[i] = min_release_date

        return self.jobs
