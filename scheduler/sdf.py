from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.scheduler import Scheduler
from typing import List
import numpy as np


def __smallest_demand_1d__(jobs):
    demand = [job.d for job in jobs]
    return [x for _, x in sorted(zip(demand, jobs), key=lambda x: x[0])]


def __smallest_demand_multid_max__(jobs):
    demand = [np.max(job.d) for job in jobs]
    return [x for _, x in sorted(zip(demand, jobs), key=lambda x: x[0])]


def __smallest_demand_multid_prod__(jobs):
    demand = [np.prod(job.d) for job in jobs]
    return [x for _, x in sorted(zip(demand, jobs), key=lambda x: x[0])]


def __smallest_demand_multid_sum__(jobs):
    demand = [np.sum(job.d) for job in jobs]
    return [x for _, x in sorted(zip(demand, jobs), key=lambda x: x[0])]


@dataclass
class SDFEFScheduler(Scheduler):
    # Smallest Demand First Earliest Feasible
    multid_priority: str = None

    def __repr__(self):
        return f"SDF-EF-{self.multid_priority.upper()}"

    def process(self):
        jobs = None
        match self.multid_priority:
            case "max":
                jobs = __smallest_demand_multid_max__(self.jobs)
            case "prod":
                jobs = __smallest_demand_multid_prod__(self.jobs)
            case "sum":
                jobs = __smallest_demand_multid_sum__(self.jobs)
            case None:
                jobs = __smallest_demand_1d__(self.jobs)

        return self.__schedule__(jobs)

    def __schedule__(self, jobs):
        # Schedule jobs via Earlist Feasible Mechanism

        for job in jobs:
            start_time, machine_idx, machine = super().earliest_feasible_machine(job)

            job.S = start_time
            job.i = machine_idx
            machine.jobs.append(job)

        return self.jobs


@dataclass
class SDFRRScheduler:
    # Smallest Demand First Round Robin Scheduler
    multid_priority: str = None
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __repr__(self):
        return f"SDF-RR-{self.multid_priority.upper()}"

    def process(self):
        jobs = None
        match self.multid_priority:
            case "max":
                jobs = __smallest_demand_multid_max__(self.jobs)
            case "prod":
                jobs = __smallest_demand_multid_prod__(self.jobs)
            case "sum":
                jobs = __smallest_demand_multid_sum__(self.jobs)
            case None:
                jobs = __smallest_demand_1d__(self.jobs)

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
