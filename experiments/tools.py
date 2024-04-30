from scheduler.job import Job
from scheduler.machine import Machine
from typing import List
from dataclasses import dataclass, field
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
import sqlite3
from tqdm import tqdm
import lz4.frame

def reset_state(jobs: List[Job], machines: List[Machine]):
    for job in jobs:
        job.S = None

    for machine in machines:
        machine.jobs = []

# Function to add labels above each bar
def autolabel(bars):
    for bar in bars:
        height = round(bar.get_height(), 1)
        plt.text(bar.get_x() + bar.get_width() / 2., height, str(height), ha='center', va='bottom')


def autolabel_h(bars, scientific_notation=False):
    for bar in bars:
        width = round(bar.get_width(), 1)
        if scientific_notation:
            plt.text(width, f"{bar.get_y() + bar.get_height() / 2.:.2e}", str(width), ha='left', va='bottom')
        else:
            plt.text(width, bar.get_y() + bar.get_height() / 2., str(width), ha='left', va='bottom')

@dataclass
class DataGenerator:
    N: int = 0
    R: int = 0
    M: int = 0
    integer: bool = True
    p_max: int = 0
    r_max: int = 0
    w_max: int = 2
    num_demand_levels = 10
    dataset: str = 'synthetic'
    float_resource_augmentation = 1E-8 # Additional absolute resources given to the machines due to floating point

    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __post_init__(self):
        match self.dataset:
            case 'azure_packing_2020':
                self.load_azure_packing_2020()
            case 'synthetic':
                self.generate_jobs()

        self.generate_machines()

        self.jobs.sort(key=lambda job: job.r)

    def generate_jobs(self):
        # Generate a list of jobs first
        for i in range(self.N):
            if self.integer:
                d = np.random.randint(1, self.num_demand_levels, size=self.R)
                self.jobs.append(Job(
                    p=np.random.randint(low=1, high=self.p_max),
                    r=np.random.randint(low=1, high=self.r_max),
                    w=np.random.randint(low=1, high=self.w_max),
                    d=d)
                )

            else:
                self.jobs.append(Job(
                    p=np.random.ranf() * self.p_max,
                    r=np.random.ranf() * self.r_max,
                    w=np.random.ranf() * self.w_max,
                    d=np.random.rand(self.R))
                )

    def generate_machines(self):
        if self.integer:
            for i in range(self.M):
                self.machines.append(Machine(D=np.ones(self.R).astype(int) * self.num_demand_levels))
        else:
            for i in range(self.M):
                self.machines.append(Machine(D=np.ones(self.R).astype(float) * self.num_demand_levels + self.float_resource_augmentation))

    def get_copy(self):
        # Create a deep copy of machines and jobs and return the list
        return copy.deepcopy(self.jobs), copy.deepcopy(self.machines)

    def add_new_jobs(self, N):
        if self.N == N:
            return
        for i in range(self.N, N):
            if self.integer:
                d = np.random.randint(1, self.num_demand_levels, size=self.R)
                self.jobs.append(Job(
                    p=np.random.randint(low=1, high=self.p_max),
                    r=np.random.randint(low=1, high=self.r_max),
                    w=np.random.randint(low=1, high=self.w_max),
                    d=d)
                )
            else:
                self.jobs.append(Job(
                    p=np.random.ranf() * self.p_max,
                    r=np.random.ranf() * self.r_max,
                    w=np.random.ranf() * self.w_max,
                    d=np.random.rand(self.R))
                )

        self.N = N

    def add_new_machines(self, M):
        if self.M == M:
            return
        for i in range(self.M, M):
            self.machines.append(Machine(D=np.ones(self.R).astype(int) * self.num_demand_levels))

        self.M = M

    def add_new_resources(self, R):
        if R == self.R:
            return

        for machine in self.machines:
            machine.D = np.ones(R) * self.num_demand_levels

        for job in self.jobs:
            if self.integer:
                new_d_vals = np.random.randint(0, self.num_demand_levels, size=(R-self.R))
            else:
                new_d_vals = np.random.rand(R-self.R)

            job.d = np.concatenate([job.d, new_d_vals])

        self.R = R

    def load_azure_packing_2020(self):
        # Stats:
        # 4.6M jobs
        # 4619 distinct VM types
        # 35 distinct machineIds
        # 500000 is approx 1 day's worth of jobs
        # Storing the entire job list requires around 2 GB of RAM
        self.integer = False

        resources = 4 if not self.R else self.R

        def path_per_resource(resource):
            return Path(__file__).parent / Path(f'data/azure_packing_2020_{resource}.pickle.lz4')

        if not path_per_resource(resources).exists():
            dat = sqlite3.connect('data/packing_trace_zone_a_v1.sqlite')
            query = dat.execute("SELECT * FROM vm")
            cols = [column[0] for column in query.description]
            df_vm = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

            query = dat.execute("SELECT * FROM vmType")
            cols = [column[0] for column in query.description]
            df_vmtype = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

            dat.close()

            df_vm.dropna(inplace=True)
            df_vm = df_vm[(df_vm['starttime'] >= 0)]
            df_vm.drop(columns=['vmId', 'tenantId'])
            df_vm['priority'] += 1  # Transform priorities to >= 1
            df_vm['processingtime'] = df_vm['endtime'] - df_vm['starttime']

            # Normalize so that p_min >= 1
            # df_vm['starttime'] = df_vm['starttime'] / p_min
            # df_vm['endtime'] = df_vm['endtime'] / p_min
            # df_vm['processingtime'] = df_vm['processingtime'] / p_min

            # Merge the HDD and SSD to single column, whenever SSD usage is below 1E-5
            df_vmtype['storage'] = df_vmtype['hdd'].where(df_vmtype['ssd'] < 1E-5, df_vmtype['ssd'])
            df_vmtype.drop(columns=['machineId', 'hdd', 'ssd', 'id'], inplace=True)

            unique_vmTypes = df_vmtype['vmTypeId'].unique()
            vmType_resources = {}

            for vmType in unique_vmTypes:
                vmType_resources[vmType] = df_vmtype[df_vmtype['vmTypeId'] == vmType].to_numpy()[:, 1:]

            df_vm.sort_values('starttime', ascending=True, inplace=True)

            df_vm.reset_index(drop=True, inplace=True)

            D = np.zeros(shape=(len(df_vm), self.R))

            jobs = []
            for j, row in tqdm(df_vm.iterrows(), total=len(df_vm), desc='4'):
                index = np.random.choice(vmType_resources[row['vmTypeId']].shape[0], 1, replace=False)
                # Randomly sample a demand from one of the machine types
                d = vmType_resources[row['vmTypeId']][index, :][0] * DataGenerator.num_demand_levels
                D[j, :4] = d
                job = Job(p=row['processingtime'], r=row['starttime'], w=row['priority'], d=d)
                jobs.append(job)

            with lz4.frame.open(str(path_per_resource(4)), 'wb') as f:
                pickle.dump(jobs, f)

            # Add new resources if required, by sampling on CPU, as it is relatively evenly distributed
            if self.R:
                resources_to_add = self.R - 4
                # Normal sampling
                # mean, std = np.mean(D[:, 0]), np.std(D[:, 0])
                # samples = np.random.normal(mean, std, size=(len(df_vm), resources_to_add))

                # Random uniform sampling
                samples = np.random.choice(D[:, 0], size=(len(df_vm), resources_to_add))

                D[:, 4:] = samples

                for i in range(0, resources_to_add):
                    for j, job in tqdm(enumerate(jobs), total=len(jobs), desc=str(4+i+1)):
                        job.d = D[j, :4+i+1]

                    with lz4.frame.open(str(path_per_resource(4+i+1)), 'wb') as f:
                        pickle.dump(jobs, f)

        else:
            with lz4.frame.open(str(path_per_resource(resources)), 'rb') as f:
                jobs = pickle.load(f)

        self.jobs = jobs[:self.N]
        self.R = resources

