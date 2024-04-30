from dataclasses import dataclass, field
import numpy as np
from typing import List
from scheduler.job import Job
from intervaltree import Interval, IntervalTree
from itertools import count

@dataclass
class Machine:
    D: np.array  # Capacity
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    intersecter: IntervalTree = field(default_factory=IntervalTree)

    id: int = field(default_factory=count().__next__)

    def add_job(self, job):
        self.jobs.append(job)
        self.intersecter.addi(job.S, job.S + job.p, job)
