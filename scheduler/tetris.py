from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.scheduler import Scheduler
from typing import List
import numpy as np
from tqdm import tqdm
import bisect
import heapq

@dataclass
class TetrisScheduler(Scheduler):
    svf_weight: float = None  # A number in (0,infty) of the amount of weight to give to the SVF heuristic
    # score_averaging: {'instantaneous', 'total'}
    # instantaneous: Given the batch of current jobs, compute the weight using average of the scores of the batch
    # total: Given the batch of current jobs, compute the weight using average of all past scheduled jobs
    score_averaging: str = 'instantaneous'

    def __repr__(self):
        base = "Tetris" if self.svf_weight is None else f'Tetris({self.svf_weight})'
        return f"{base}-{self.score_averaging}"

    def process(self):
        total_alignment_score = 0
        total_remaining_work_score = 0
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        num_processed_jobs = 0

        unscheduled_jobs = self.jobs.copy()
        unscheduled_release_dates = np.fromiter((job.r for job in self.jobs), dtype=np.dtype(type(self.jobs[0].r)), count=len(self.jobs))
        unscheduled_job_demands = np.fromiter((job.d for job in self.jobs), dtype=np.dtype((type(self.jobs[0].d), R)), count=len(self.jobs))
        unscheduled_job_processing_times = np.fromiter((job.p for job in self.jobs), dtype=np.dtype(type(self.jobs[0].p)), count=len(self.jobs))

        machine_resources = np.zeros(shape=(len(self.machines), R))
        alive_jobs = {i: [] for i in range(len(self.machines))}

        t = np.ones(len(self.machines)) * min(unscheduled_release_dates)

        pbar = tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        while num_processed_jobs != len(self.jobs):
            i = np.argmin(t)
            machine = self.machines[i]

            idx = bisect.bisect_right(unscheduled_release_dates, t[i])
            arrived_jobs = unscheduled_jobs[:idx]
            arrived_job_processing_times = unscheduled_job_processing_times[:idx]
            unscheduled_release_dates_after_t = unscheduled_release_dates[idx:]
            # Try to schedule the jobs so that its start time is t

            # For each job, compute an alignment score
            total_demand = machine_resources[i, :]
            remaining_resources = machine.D - total_demand

            # Filter out the jobs that cannot feasibly be scheduled
            arrived_job_demands = unscheduled_job_demands[:idx]
            result = (np.less_equal(total_demand + arrived_job_demands, machine.D)).all(axis=1)
            feasible_job_idxs = np.where(result == True)[0]

            scheduled_jobs_idxs = []
            while feasible_job_idxs.size > 0:
                alignment = np.ones(len(arrived_jobs)) * - 1E9
                remaining_work = np.ones(len(arrived_jobs)) * -1E9

                alignment[feasible_job_idxs] = arrived_job_demands[feasible_job_idxs] @ remaining_resources
                remaining_work[feasible_job_idxs] = 1/(arrived_job_processing_times[feasible_job_idxs] * np.sum(arrived_job_demands[feasible_job_idxs], axis=1))

                if self.svf_weight is not None:
                    weight = self.svf_weight
                else:
                    # Use a weight to combine metrics, use average as recommended by paper
                    if self.score_averaging == 'instantaneous':
                        weight = np.mean(alignment[alignment >= 0]) / np.mean(remaining_work[remaining_work >= 0])
                    elif self.score_averaging == 'total':
                        weight = total_alignment_score / total_remaining_work_score if total_remaining_work_score != 0 else 1
                    else:
                        raise ValueError('score_averaging must be either "instantaneous" or "total"')

                scores = alignment + weight * remaining_work

                # Schedule the highest scoring job
                best_job_idx = np.argmax(scores)
                job = arrived_jobs[best_job_idx]
                total_alignment_score += alignment[best_job_idx]
                total_remaining_work_score += remaining_work[best_job_idx]

                job.S = t[i]
                job.i = i
                machine.add_job(job)
                num_processed_jobs += 1
                pbar.update(1)

                scheduled_jobs_idxs.append(best_job_idx)

                heapq.heappush(alive_jobs[i], (job.S + job.p, job.id, job))

                total_demand += job.d  # Update the demand since we've scheduled it to start at this time
                remaining_resources = machine.D - total_demand
                machine_resources[i, :] = total_demand

                # Recompute feasible jobs
                result = (np.less_equal(total_demand + arrived_job_demands, machine.D)).all(axis=1)
                feasible_job_idxs = np.where(result == True)[0]
                if feasible_job_idxs.size > 0:
                    feasible_job_idxs = np.setdiff1d(feasible_job_idxs, scheduled_jobs_idxs)

            for idx_ in sorted(scheduled_jobs_idxs, reverse=True):
                unscheduled_jobs.pop(idx_)

            if scheduled_jobs_idxs:
                unscheduled_job_demands = np.delete(unscheduled_job_demands, scheduled_jobs_idxs, axis=0)
                unscheduled_release_dates = np.delete(unscheduled_release_dates, scheduled_jobs_idxs)
                unscheduled_job_processing_times = np.delete(unscheduled_job_processing_times, scheduled_jobs_idxs)

            # Advance time to the next time a job starts, finishes, or releases
            min_release_date = min(unscheduled_release_dates_after_t) if unscheduled_release_dates_after_t.size != 0 else np.inf
            if alive_jobs[i]:
                (min_completion_time, _, _) = alive_jobs[i][0]
            else:
                min_completion_time = np.inf

            if min_completion_time < min_release_date:
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
            else:
                t[i] = min_release_date

        return self.jobs
