from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.scheduler import Scheduler
from typing import List
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor as Pool
import os
from tqdm.auto import tqdm


def __smallest_volume_1d__(jobs):
    volumes = [job.d * job.p for job in jobs]
    return [x for _, x in sorted(zip(volumes, jobs), key=lambda x: x[0])]


def __smallest_volume_multid_max__(jobs):
    volumes = [np.max(job.d) * job.p for job in jobs]
    return [x for _, x in sorted(zip(volumes, jobs), key=lambda x: x[0])]


def __smallest_volume_multid_prod__(jobs):
    volumes = [np.prod(job.d) * job.p for job in jobs]
    return [x for _, x in sorted(zip(volumes, jobs), key=lambda x: x[0])]


def __smallest_volume_multid_sum__(jobs):
    volumes = [np.sum(job.d) * job.p for job in jobs]
    return [x for _, x in sorted(zip(volumes, jobs), key=lambda x: x[0])]


@dataclass
class SVFEFScheduler(Scheduler):
    # Smallest Volume First Earliest Feasible
    multid_priority: str = None
    online: bool = False

    def __repr__(self):
        return f"SVF-EF-{self.multid_priority.upper()}" if self.multid_priority else "SVF-EF"

    def process(self):
        jobs = None
        match self.multid_priority:
            case "max":
                jobs = __smallest_volume_multid_max__(self.jobs)
            case "prod":
                jobs = __smallest_volume_multid_prod__(self.jobs)
            case "sum":
                jobs = __smallest_volume_multid_sum__(self.jobs)
            case None:
                jobs = __smallest_volume_1d__(self.jobs)

        jobs = self.__schedule_earliest_feasible_machine__(jobs)

        if self.online:
            jobs = self.__offline_to_online__(jobs)

        return jobs

@dataclass
class SVFRRScheduler:
    # Smallest Volume First Round Robin Scheduler
    multid_priority: str = None
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __repr__(self):
        return f"SVF-RR-{self.multid_priority.upper()}"

    def process(self):
        jobs = None
        match self.multid_priority:
            case "max":
                jobs = __smallest_volume_multid_max__(self.jobs)
            case "prod":
                jobs = __smallest_volume_multid_prod__(self.jobs)
            case "sum":
                jobs = __smallest_volume_multid_sum__(self.jobs)
            case None:
                jobs = __smallest_volume_1d__(self.jobs)

        return self.__schedule__(jobs)

    def __schedule__(self, jobs):
        try:
            R = jobs[0].d.size
        except AttributeError:
            R = 1

        machine_idx = 0

        for job in jobs:
            machine = self.machines[machine_idx]
            candidate_start = None
            t = 0
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
                        machine_idx = (machine_idx + 1) % len(self.machines)
                        break
                else:
                    # Could not feasibly schedule job over its processing time
                    if candidate_start is not None and candidate_start + job.p > t:
                        candidate_start = None

                # Advance time horizon by the earliest completion time of occupying jobs
                completion_times = [j.S + j.p for j in occupying_jobs]
                t = min(completion_times) if completion_times else t + job.p

        return self.jobs
