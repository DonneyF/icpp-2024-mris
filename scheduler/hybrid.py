from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.svf import SVFEFScheduler
from scheduler.sjf import SJFScheduler
from typing import List
import numpy as np

@dataclass
class HybridScheduler:
    # Combination of SVF and SJF on distinct machines
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __repr__(self):
        return f"HYBRID"

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        # Partition the jobs into R+1 distinct sets
        job_sets = {l:[] for l in range(0, R+1)}

        for job in self.jobs:
            if np.max(job.d) <= 0.5:
                job_sets[0].append(job)
            else:
                job_sets[np.argmax(job.d)+1].append(job)

        beta = (-(R+1) + np.sqrt(np.power(R, 2) + 2*R + 9)) / 2

        M = len(self.machines)
        Ml = int(max({1, np.floor((1-beta)*M/R)}))
        M0 = M - R*Ml

        # Schedule jobs J0 using SVF using M0 machines

        if len(job_sets[0]) != 0:
            svf_scheduler = SVFEFScheduler(multid_priority="max")
            svf_scheduler.jobs = job_sets[0]
            svf_scheduler.machines = self.machines[0:M0]
            svf_scheduler.process()

        # Schedule jobs jl using SJF using Ml machines
        for l in range(1, R+1):
            if len(job_sets[l]) == 0:
                continue
            sjf_scheduler = SJFScheduler()
            sjf_scheduler.jobs = job_sets[l]
            sjf_scheduler.machines = self.machines[M0 + (l-1)*Ml:M0 + l*Ml]
            sjf_scheduler.process()

        return self.jobs
