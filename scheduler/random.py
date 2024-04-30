from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.scheduler import Scheduler
from typing import List
import numpy as np
import random

@dataclass
class RandomRandomScheduler(Scheduler):

    def __repr__(self):
        return f"RAND-RAND"

    def process(self):
        random.shuffle(self.jobs)

        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        for job in self.jobs:
            # Randomly select a machine
            machine_idx = np.random.randint(0, len(self.machines))
            machine = self.machines[machine_idx]

            candidate_start = None
            t = 0
            # Retrieve current makespan
            makespan = 0 if not machine.jobs else max([j.S + j.p for j in machine.jobs])
            while t <= makespan + job.p:
                occupying_jobs = [j for j in machine.jobs if j.S <= t < j.S + j.p]
                occupying_jobs = [j for j in occupying_jobs if not j.S > t + job.p]

                total_demand = np.add.reduce([j.d for j in occupying_jobs]) if occupying_jobs else np.zeros(R)
                if (np.less_equal(total_demand + job.d, machine.D)).all():
                    # Set the first time we can feasible schedule
                    if candidate_start is None:
                        candidate_start = t

                    # Ensure that we can fit the job for the entire job horizon
                    if candidate_start + job.p <= t:
                        job.S = candidate_start
                        job.i = machine.id
                        machine.jobs.append(job)
                        break
                else:
                    # Could not feasibly schedule job over its processing time
                    if candidate_start is not None and candidate_start + job.p > t:
                        candidate_start = None

                # Advance time horizon by the earliest completion time of occupying jobs
                completion_times = [j.S + j.p for j in occupying_jobs]
                t = min(completion_times) if completion_times else t + job.p

        return self.jobs

