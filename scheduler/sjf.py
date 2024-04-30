from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from typing import List
import numpy as np

@dataclass
class SJFEFScheduler:
    # Smallest Job First Earliest Feasible

    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __repr__(self):
        return f"SJF-EF"

    def process(self):
        processing_times = [job.p for job in self.jobs]
        jobs = [x for _, x in sorted(zip(processing_times, self.jobs), key=lambda x: x[0])]
        # Schedule jobs via Earlist Feasible Mechanism
        try:
            R = jobs[0].d.size
        except AttributeError:
            R = 1

        for job in jobs:
            # Get the earliest feasible machine
            start_times = []
            for i, machine in enumerate(self.machines):
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
                            start_times.append(candidate_start)
                            break
                    else:
                        # Could not feasibly schedule job over its processing time
                        if candidate_start is not None and candidate_start + job.p > t:
                            candidate_start = None

                    # Advance time horizon by the earliest completion time of occupying jobs
                    completion_times = [j.S + j.p for j in occupying_jobs]
                    t = min(completion_times) if completion_times else t + job.p

            earliest_start_time = min(start_times)
            earliest_feasible_machine_idx = start_times.index(earliest_start_time)
            earliest_feasible_machine = self.machines[earliest_feasible_machine_idx]

            job.S = earliest_start_time
            job.i = earliest_feasible_machine_idx
            earliest_feasible_machine.jobs.append(job)

        return self.jobs


@dataclass
class SJFScheduler:
    # Shortest Job First Scheduler
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __repr__(self):
        return f"SJF"

    def process(self):
        processing_times = [job.p for job in self.jobs]
        jobs = [x for _, x in sorted(zip(processing_times, self.jobs), key=lambda x: x[0])]
        # Schedule jobs on the earliest machine. We assume machine can only schedule one job at a time

        completion_times = np.zeros(len(self.machines))
        for job in jobs:
            earliest_machine_idx = np.argmin(completion_times)
            earliest_start_time = np.min(completion_times)
            earliest_machine = self.machines[earliest_machine_idx]

            # Schedule
            job.S = earliest_start_time
            job.id = earliest_machine_idx
            earliest_machine.jobs.append(job)

            completion_times[earliest_machine_idx] += job.p

        return jobs

@dataclass
class SJFRRScheduler:
    # Smallest Volume First Round Robin Scheduler
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __repr__(self):
        return f"SJF-RR"

    def process(self):
        processing_times = [job.p for job in self.jobs]
        jobs = [x for _, x in sorted(zip(processing_times, self.jobs), key=lambda x: x[0])]

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
