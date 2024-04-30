from dataclasses import dataclass
import numpy as np
from scheduler.scheduler import Scheduler
from tqdm.auto import tqdm
import gurobipy as gp
from gurobipy import GRB
import bisect


@dataclass
class MRIS(Scheduler):
    # General outline
    # Iterate over geometrically increasing intervals
    # Find a subset of jobs with maximum weight
    # Schedule them offline using list scheduling
    # Repeat until completion

    sort: str = None

    def __repr__(self):
        return "MRIS-ERF" if not self.sort else f"MRIS-{self.sort}"

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        unscheduled_jobs = self.jobs.copy()
        unscheduled_release_dates = np.fromiter((job.r for job in self.jobs), dtype=np.dtype(type(self.jobs[0].r)),
                                                count=len(self.jobs))
        unscheduled_job_demands = np.fromiter((job.d for job in self.jobs), dtype=np.dtype((type(self.jobs[0].d), R)),
                                              count=len(self.jobs))
        unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs),
                                                       dtype=np.dtype(type(self.jobs[0].p)), count=len(self.jobs))
        unscheduled_job_volumes = np.sum(unscheduled_job_demands, axis=1) * unscheduled_job_processing_times
        unscheduled_job_weights = np.fromiter((job.w for job in self.jobs), dtype=np.dtype(type(self.jobs[0].w)),
                                              count=len(self.jobs))

        p_min = np.min(unscheduled_job_processing_times)

        # Rescale for our purposes
        unscheduled_job_processing_times /= p_min
        unscheduled_release_dates /= p_min
        unscheduled_job_volumes /= p_min

        for job in unscheduled_jobs:
            job.p /= p_min

        num_processed_jobs = 0

        k = 0
        pbar = tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        while num_processed_jobs != len(self.jobs):
            t = np.power(2, k)

            # Find all the jobs released up to time t
            idx = bisect.bisect_right(unscheduled_release_dates, t)
            potential_jobs_idxs = np.where(t >= unscheduled_job_processing_times[:idx])[0]

            released_volumes = unscheduled_job_volumes[potential_jobs_idxs]
            released_weights = unscheduled_job_weights[potential_jobs_idxs]

            N_k = len(potential_jobs_idxs)

            if N_k == 0:
                k += 1
                continue

            max_volume = max(self.machines[0].D) * R * len(self.machines) * t

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()

            m = gp.Model(env=env)
            x = m.addVars(N_k, name="x", vtype=GRB.BINARY)

            m.setObjective(gp.quicksum(released_weights[i] * x[i] for i in range(N_k)), GRB.MAXIMIZE)

            m.addConstr((gp.quicksum(released_volumes[i] * x[i] for i in range(N_k)) <= max_volume), name="knapsack")

            m.setParam(GRB.Param.LogToConsole, 0)

            m.optimize()

            jobs_to_schedule_idxs = [potential_jobs_idxs[idx] for idx in range(N_k) if x[idx].x == 1]
            # jobs_to_schedule = [unscheduled_jobs[idx] for idx in jobs_to_schedule_idxs]

            # Schedule the jobs using list scheduling starting at current time
            unscheduled_job_heuristics = None
            # Compute the heuristic for each job without sorting by ascending order
            release_dates_to_schedule = unscheduled_release_dates[jobs_to_schedule_idxs]
            processing_times_to_schedule = unscheduled_job_processing_times[jobs_to_schedule_idxs]
            job_demands_to_schedule = unscheduled_job_demands[jobs_to_schedule_idxs]
            job_weights_to_schedule = unscheduled_job_weights[jobs_to_schedule_idxs]

            match self.sort:
                case "ERF":
                    unscheduled_job_heuristics = release_dates_to_schedule
                case "SJF":
                    unscheduled_job_heuristics = processing_times_to_schedule
                case "SVF":
                    unscheduled_job_heuristics = processing_times_to_schedule * np.sum(job_demands_to_schedule, axis=1)
                case "WERF":
                    unscheduled_job_heuristics = unscheduled_release_dates / job_weights_to_schedule
                case "SDF":
                    unscheduled_job_heuristics = np.sum(job_demands_to_schedule, axis=1)
                case "WSJF":
                    unscheduled_job_heuristics = processing_times_to_schedule / job_weights_to_schedule
                case "WSVF":
                    unscheduled_job_heuristics = processing_times_to_schedule * np.sum(job_demands_to_schedule,axis=1) / job_weights_to_schedule
                case "WSDF":
                    unscheduled_job_heuristics = np.sum(job_demands_to_schedule, axis=1) / job_weights_to_schedule
                case _:
                    unscheduled_job_heuristics = unscheduled_release_dates

            unscheduled_job_heuristics_idx = np.argsort(unscheduled_job_heuristics)
            jobs_to_schedule = [unscheduled_jobs[jobs_to_schedule_idxs[idx]] for idx in unscheduled_job_heuristics_idx]

            release_dates_to_schedule = release_dates_to_schedule.tolist()

            num_processed_jobs_k = 0

            # print(num_processed_jobs, len(jobs_to_schedule), len(unscheduled_jobs))
            t_k = np.ones(len(self.machines)) * t
            while num_processed_jobs_k < len(jobs_to_schedule_idxs):
                i = np.argmin(t_k)
                machine = self.machines[i]
                # For each machine, if a job can fit on the machine at time t[i], go ahead and schedule it

                # Property of List Scheduling: Scheduled jobs have start times at or before t[i].
                # Therefore, we only need to check the current resource usage at time t[i]
                # intervals = machine.intersecter.at(t_k[i])
                # total_demand = np.add.reduce([interval.data.d for interval in intervals]) if intervals else np.zeros(R)

                # List Scheduling requires us to always scan from the start of the list, even after identifying a job to schedule

                while True:
                    # Check that this job fits over this time horizon
                    for job in jobs_to_schedule:
                        t_ = t_k[i]
                        S = None
                        while t_ <= t_k[i] + job.p:
                            # Find all the jobs alive at time t_ on machine i, i.e. S_j <= t < C_j
                            intervals = machine.intersecter.at(t_)
                            completion_times = [interval.data.S + interval.data.p for interval in intervals if t_k[i] + job.p > interval.data.S + interval.data.p] if intervals else []
                            total_demand = np.add.reduce([interval.data.d for interval in intervals]) if intervals else np.zeros(R)
                            if (np.less_equal(total_demand + job.d, machine.D)).all():
                                # t is a feasible start. Ensure the job fits for the entirety of its processing time
                                if S is None:
                                    S = t_

                                if S + job.p <= t_:
                                    job.S = t_k[i]
                                    job.i = i
                                    machine.add_job(job)
                                    num_processed_jobs_k += 1
                                    pbar.update(1)
                                    # print(num_processed_jobs_k)
                                    break
                            else:
                                # Could not feasibly schedule job over its processing time. Our candidate is not valid
                                break

                            # Advance time horizon by the earliest completion time of occupying jobs
                            t_ = min(completion_times) if len(completion_times) > 0 else S + job.p
                    else:
                        # Reached the end of the loop without scheduling a job. Safe to advance to next time
                        break

                jobs_to_schedule[:] = [job for job in jobs_to_schedule if job.S is None]

                # Advance time to the next time a job starts, finishes, or releases
                intervals = machine.intersecter.at(t_k[i])
                release_dates_to_schedule[:] = [ele for ele in release_dates_to_schedule if ele > t_k[i]]
                times = [j.begin for j in intervals if j.begin > t_k[i]] + [j.end for j in intervals if j.end > t_k[i]] + release_dates_to_schedule
                t_k[i] = min(times)

            for idx in sorted(jobs_to_schedule_idxs, reverse=True):
                unscheduled_jobs.pop(idx)

            if jobs_to_schedule_idxs:
                unscheduled_job_volumes = np.delete(unscheduled_job_volumes, jobs_to_schedule_idxs)
                unscheduled_job_weights = np.delete(unscheduled_job_weights, jobs_to_schedule_idxs)
                unscheduled_release_dates = np.delete(unscheduled_release_dates, jobs_to_schedule_idxs)
                unscheduled_job_processing_times = np.delete(unscheduled_job_processing_times, jobs_to_schedule_idxs)

            k += 1
            num_processed_jobs += num_processed_jobs_k

        for job in self.jobs:
            job.p *= p_min
            job.S *= p_min

        return self.jobs


@dataclass
class MRISGreedy(Scheduler):
    # MRIS that uses greedy heuristic to solve knapsack

    sort: str = None

    def __repr__(self):
        return "MRIS-Greedy-ERF" if not self.sort else f"MRIS-Greedy-{self.sort}"

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        unscheduled_jobs = self.jobs.copy()
        unscheduled_release_dates = np.fromiter((job.r for job in self.jobs), dtype=np.dtype(type(self.jobs[0].r)),
                                                count=len(self.jobs))
        unscheduled_job_demands = np.fromiter((job.d for job in self.jobs), dtype=np.dtype((type(self.jobs[0].d), R)),
                                              count=len(self.jobs))
        unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs),
                                                       dtype=np.dtype(type(self.jobs[0].p)), count=len(self.jobs))
        unscheduled_job_volumes = np.sum(unscheduled_job_demands, axis=1) * unscheduled_job_processing_times
        unscheduled_job_weights = np.fromiter((job.w for job in self.jobs), dtype=np.dtype(type(self.jobs[0].w)),
                                              count=len(self.jobs))

        p_min = np.min(unscheduled_job_processing_times)

        # Rescale for our purposes
        unscheduled_job_processing_times /= p_min
        unscheduled_release_dates /= p_min
        unscheduled_job_volumes /= p_min

        for job in unscheduled_jobs:
            job.p /= p_min

        num_processed_jobs = 0

        k = 0
        pbar = tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        while num_processed_jobs != len(self.jobs):
            t = np.power(2, k)

            # Find all the jobs released up to time t
            idx = bisect.bisect_right(unscheduled_release_dates, t)
            potential_jobs_idxs = np.where(t >= unscheduled_job_processing_times[:idx])[0]

            released_volumes = unscheduled_job_volumes[potential_jobs_idxs]
            released_weights = unscheduled_job_weights[potential_jobs_idxs]

            N_k = len(potential_jobs_idxs)

            if N_k == 0:
                k += 1
                continue

            max_volume = 2 * max(self.machines[0].D) * R * len(self.machines) * t

            # Sort the items by increasing order of value/weight
            knapsack_ratios = released_weights / released_volumes
            sorted_items_idx = np.argsort(knapsack_ratios)[::-1] # non-increasing order
            cumsum = np.cumsum(released_volumes[sorted_items_idx])
            idx = np.argmax(cumsum > max_volume) if cumsum.max() >= max_volume else len(cumsum) - 1

            if idx == len(sorted_items_idx):
                # Can schedule all of them
                jobs_to_schedule_idxs = sorted_items_idx
            elif released_weights[sorted_items_idx[:idx]].sum() < released_weights[sorted_items_idx[idx]]:
                jobs_to_schedule_idxs = np.array([idx])
            else:
                jobs_to_schedule_idxs = sorted_items_idx[:idx]

            # Schedule the jobs using list scheduling starting at current time
            unscheduled_job_heuristics = None
            # Compute the heuristic for each job without sorting by ascending order
            release_dates_to_schedule = unscheduled_release_dates[jobs_to_schedule_idxs]
            processing_times_to_schedule = unscheduled_job_processing_times[jobs_to_schedule_idxs]
            job_demands_to_schedule = unscheduled_job_demands[jobs_to_schedule_idxs]
            job_weights_to_schedule = unscheduled_job_weights[jobs_to_schedule_idxs]

            match self.sort:
                case "ERF":
                    unscheduled_job_heuristics = release_dates_to_schedule
                case "SJF":
                    unscheduled_job_heuristics = processing_times_to_schedule
                case "SVF":
                    unscheduled_job_heuristics = processing_times_to_schedule * np.sum(job_demands_to_schedule, axis=1)
                case "WERF":
                    unscheduled_job_heuristics = unscheduled_release_dates / job_weights_to_schedule
                case "SDF":
                    unscheduled_job_heuristics = np.sum(job_demands_to_schedule, axis=1)
                case "WSJF":
                    unscheduled_job_heuristics = processing_times_to_schedule / job_weights_to_schedule
                case "WSVF":
                    unscheduled_job_heuristics = processing_times_to_schedule * np.sum(job_demands_to_schedule,axis=1) / job_weights_to_schedule
                case "WSDF":
                    unscheduled_job_heuristics = np.sum(job_demands_to_schedule, axis=1) / job_weights_to_schedule
                case _:
                    unscheduled_job_heuristics = unscheduled_release_dates

            unscheduled_job_heuristics_idx = np.argsort(unscheduled_job_heuristics)
            jobs_to_schedule = [unscheduled_jobs[jobs_to_schedule_idxs[idx]] for idx in unscheduled_job_heuristics_idx]

            release_dates_to_schedule = release_dates_to_schedule.tolist()

            num_processed_jobs_k = 0

            # print(num_processed_jobs, len(jobs_to_schedule), len(unscheduled_jobs))
            t_k = np.ones(len(self.machines)) * t
            while num_processed_jobs_k < len(jobs_to_schedule_idxs):
                i = np.argmin(t_k)
                machine = self.machines[i]
                # For each machine, if a job can fit on the machine at time t[i], go ahead and schedule it

                # Property of List Scheduling: Scheduled jobs have start times at or before t[i].
                # Therefore, we only need to check the current resource usage at time t[i]
                # intervals = machine.intersecter.at(t_k[i])
                # total_demand = np.add.reduce([interval.data.d for interval in intervals]) if intervals else np.zeros(R)

                # List Scheduling requires us to always scan from the start of the list, even after identifying a job to schedule

                while True:
                    # Check that this job fits over this time horizon
                    for job in jobs_to_schedule:
                        t_ = t_k[i]
                        S = None
                        while t_ <= t_k[i] + job.p:
                            # Find all the jobs alive at time t_ on machine i, i.e. S_j <= t < C_j
                            intervals = machine.intersecter.at(t_)
                            completion_times = [interval.data.S + interval.data.p for interval in intervals if t_k[i] + job.p > interval.data.S + interval.data.p] if intervals else []
                            total_demand = np.add.reduce([interval.data.d for interval in intervals]) if intervals else np.zeros(R)
                            if (np.less_equal(total_demand + job.d, machine.D)).all():
                                # t is a feasible start. Ensure the job fits for the entirety of its processing time
                                if S is None:
                                    S = t_

                                if S + job.p <= t_:
                                    job.S = t_k[i]
                                    job.i = i
                                    machine.add_job(job)
                                    num_processed_jobs_k += 1
                                    pbar.update(1)
                                    # print(num_processed_jobs_k)
                                    break
                            else:
                                # Could not feasibly schedule job over its processing time. Our candidate is not valid
                                break

                            # Advance time horizon by the earliest completion time of occupying jobs
                            t_ = min(completion_times) if len(completion_times) > 0 else S + job.p
                    else:
                        # Reached the end of the loop without scheduling a job. Safe to advance to next time
                        break

                jobs_to_schedule[:] = [job for job in jobs_to_schedule if job.S is None]

                # Advance time to the next time a job starts, finishes, or releases
                intervals = machine.intersecter.at(t_k[i])
                release_dates_to_schedule[:] = [ele for ele in release_dates_to_schedule if ele > t_k[i]]
                times = [j.begin for j in intervals if j.begin > t_k[i]] + [j.end for j in intervals if j.end > t_k[i]] + release_dates_to_schedule
                t_k[i] = min(times)

            for idx in sorted(jobs_to_schedule_idxs, reverse=True):
                unscheduled_jobs.pop(idx)

            if len(jobs_to_schedule_idxs) > 0:
                unscheduled_job_volumes = np.delete(unscheduled_job_volumes, jobs_to_schedule_idxs)
                unscheduled_job_weights = np.delete(unscheduled_job_weights, jobs_to_schedule_idxs)
                unscheduled_release_dates = np.delete(unscheduled_release_dates, jobs_to_schedule_idxs)
                unscheduled_job_processing_times = np.delete(unscheduled_job_processing_times, jobs_to_schedule_idxs)

            k += 1
            num_processed_jobs += num_processed_jobs_k

        for job in self.jobs:
            job.p *= p_min
            job.S *= p_min

        return self.jobs