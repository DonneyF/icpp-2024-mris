from dataclasses import dataclass, field
from scheduler.job import Job
from scheduler.machine import Machine
from typing import List
import numpy as np
from scheduler.svf import SVFEFScheduler
import gurobipy as gp
from gurobipy import GRB

@dataclass
class GurobiScheduler:
    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)
    env = None
    heuristic_start = False

    def __init__(self, heuristic_start=False, env=None):
        self.heuristic_start = heuristic_start
        self.env = env

    def __repr__(self):
        if self.heuristic_start:
            return f"Gurobi-Heuristic"
        else:
            return f"Gurobi"

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        N = len(self.jobs)
        M = len(self.machines)
        # T = int(sum([job.p for job in self.jobs]) // M * 1.2)  # Relaxed upper bound on makespan
        T = int(sum([job.p for job in self.jobs]) // M * 2) + max([job.r for job in self.jobs])  # Relaxed upper bound on makespan

        if self.env is None:
            self.env = gp.Env()

        m = gp.Model("mip1", self.env)

        x = m.addVars(N, M, T, name="x", vtype=GRB.BINARY)

        C = m.addVars(N, name="C", vtype=GRB.INTEGER, lb=0)

        # Objective
        m.setObjective(C.sum(), GRB.MINIMIZE)

        if self.heuristic_start:
            svf = SVFEFScheduler(multid_priority="sum")
            svf.jobs = self.jobs
            svf.machines = self.machines
            svf_jobs = svf.process()

            for j in range(N):
                for i in range(M):
                    for t in range(T):
                        if svf_jobs[j].i == i:
                            if svf_jobs[j].S == t:
                                x[j, i, t].setAttr(GRB.Attr.Start, 1)
                            else:
                                x[j, i, t].setAttr(GRB.Attr.Start, 0)
                        else:
                            x[j, i, t].setAttr(GRB.Attr.Start, 0)

                C[j].setAttr(GRB.Attr.Start, svf_jobs[j].S + svf_jobs[j].p)

        # Completion time constraint
        for j in range(N):
            m.addConstr(
                gp.quicksum(t * x[j, i, t] for i in range(M) for t in range(T)) + self.jobs[j].p <= C[j],
                f"Completion_{j}"
            )

        # Release time constraint
        for j in range(N):
            m.addConstr(
                gp.quicksum(t * x[j, i, t] for i in range(M) for t in range(T)) >= self.jobs[j].r,
               f"Release_{j}"
            )

        # Disallow migration across machines
        for j in range(N):
            m.addConstr(x.sum(j, '*', '*') == 1, f"Migration_{j}")

        # Resource capacity constraint
        for i in range(M):
            for l in range(R):
                for t in range(T):
                    expr = gp.quicksum(self.jobs[j].d[l] * x[j, i, v] for j in range(N) for v in range(max({0, t - self.jobs[j].p}), t))
                    m.addConstr(expr <= self.machines[i].D[l], f"Resource_{i}_{l}_{t}")

        m.optimize()

        for j in range(N):
            for i in range(M):
                x_ji = [x[j, i, t].x for t in range(T)]
                if np.sum(x_ji) > 0:
                    self.jobs[j].S = int(np.argmax(x_ji))

                    self.machines[i].jobs.append(self.jobs[j])
                    self.jobs[j].i = i

        return self.jobs
