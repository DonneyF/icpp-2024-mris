from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.scheduler import Scheduler
from typing import List
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor as Pool
import os
from tqdm import tqdm

@dataclass
class EFScheduler(Scheduler):
    # Earliest Feasible Scheduler
    multid: str = None
    online: bool = False
    sort: str = None
    pbar: tqdm = None

    def __repr__(self):
        return f"{self.sort}-EF-{self.multid.upper()}" if self.multid else "SVF-EF"

    def process(self):
        if self.pbar is None:
            pbar = tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        else:
            pbar = self.pbar

        jobs = self.jobs

        match self.multid:
            case "max":
                multid_function = np.max
            case "prod":
                multid_function = np.prod
            case "sum":
                multid_function = np.sum
            case _:
                multid_function = np.identity

        match self.sort:
            case "SJF":
                heuristic = [job.p for job in self.jobs]
            case "WSJF":
                heuristic = [job.p / job.w for job in self.jobs]
            case "SVF":
                heuristic = [multid_function(job.d) * job.p for job in self.jobs]
            case "WSVF":
                heuristic = [multid_function(job.d) * job.p / job.w for job in self.jobs]
            case "SDF":
                heuristic = [multid_function(job.d) for job in self.jobs]
            case "WSDF":
                heuristic = [multid_function(job.d) / job.w for job in self.jobs]
            case _:
                heuristic = None

        if heuristic is not None:
            jobs = [x for _, x in sorted(zip(heuristic, jobs), key=lambda x: x[0])]

        jobs = self.__schedule_earliest_feasible_machine__(jobs, pbar=pbar)

        if self.online:
            jobs = self.__offline_to_online__(jobs)

        return jobs
