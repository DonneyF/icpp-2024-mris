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

def plot_job_reduction_cdf(pool_results, reference_scheduler_id, N):
    # For each scheduler, compute the percentage reduction of each job
    job_completion_times = np.zeros(shape=(N, len(pool_results)))  # Indexed by job ID
    for i in range(len(pool_results)):
        for job in pool_results[i].jobs:
            job_completion_times[job.id, i] = job.S + job.p

    reference_job_completion_times = job_completion_times[:, reference_scheduler_id]

    job_completion_times_reduction = (job_completion_times - reference_job_completion_times[:, None]) / job_completion_times * 100

    job_completion_times_reduction = np.sort(job_completion_times_reduction, axis=0)

    # print(np.cumsum(job_completion_times_reduction, axis=1)[:, 0])
    plt.figure(figsize=(10, 5), dpi=200)
    plt.xscale('symlog')
    for i in [ele for ele in range(len(pool_results)) if ele != reference_scheduler_id]:
        x = job_completion_times_reduction[:, i]
        cdf_2d = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, cdf_2d, label=f'{pool_results[i]}')

    plt.legend()
    plt.ylabel('CDF')
    plt.xlabel('Reduction (%) in Job Completion Time')
    plt.show()


def plot_job_queuing_delay_cdf(pool_results, N, M, R):
    data_queuing = {}
    for scheduler in pool_results: #type: str, pd.DataFrame
        queuing_delay = np.array([(job.S - job.r) * job.w for job in scheduler.jobs])
        data_queuing[str(scheduler)] = np.sort(queuing_delay)

    plt.figure(figsize=(10, 5), dpi=200)
    plt.xscale('symlog')
    for scheduler, queuing_delay in data_queuing.items():
        cdf_2d = np.arange(1, N + 1) / N
        plt.plot(queuing_delay, cdf_2d, label=str(scheduler))

    plt.legend()
    plt.title(f'CDF of Queuing Delay of Jobs, $N={N}, M={M}, R={R}$')
    plt.ylabel('CDF')
    plt.xlabel('Queuing Delay ($S_j - r_j$)')
    plt.show()

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


def autolabel_h(bars):
    for bar in bars:
        width = round(bar.get_width(), 1)
        plt.text(width, bar.get_y() + bar.get_height() / 2., str(width), ha='center', va='bottom')

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
    data_location: Path = Path(__file__).parent / Path('data')

    jobs: List[Job] = field(default_factory=list)  # Assigned jobs
    machines: List[Machine] = field(default_factory=list)

    def __post_init__(self):
        match self.dataset:
            case 'azure_packing_2020':
                self.load_azure_packing_2020()
            case 'azure_2019_v2':
                self.load_azure_2019_v2_jobs()
                self.R = 2
            case 'alibaba_2018':
                self.load_alibaba_2018_jobs()
                self.R = 2
            case 'google_2019':
                self.load_google_2019_jobs()
                self.R = 2
            case 'google_2019_high':
                self.load_google_2019_jobs_high()
                self.R = 2
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

    def load_google_2019_jobs_high(self):
        return self.load_google_2019_jobs(min_start=82000, filename='google_2019_jobs_a_high.pickle')

    def load_google_2019_jobs(self, min_start=0, filename='google_2019_jobs_a.pickle'):
        processed_jobs_pickle_path = Path(__file__).parent / Path('data') / Path(filename)
        if not processed_jobs_pickle_path.exists():
            df = pd.read_pickle(Path(__file__).parent / Path('data/df_google_trace_jobs_by_collections.pickle'))
            df['processing_time'] = df['processing_time'].div(1E6).astype(int)  # As second
            df['submit_time'] = df['submit_time'].div(1E6).astype(int)  # As seconds
            df = df.sort_values('submit_time')
            df = df[df['processing_time'] != 0]   # Remove short jobs
            df = df[df['priority'] != 0]   # Remove jobs with 0 weight
            df = df[df['cpus'] != 0]
            df = df[df['memory'] != 0]

            df = df[df['submit_time'] >= min_start]
            df['submit_time'] -= min_start

            df = df[:1000000] # Take only first 1M jobs

            # Some jobs submitted before time 600 are long running jobs that started before start of trace measurement
            df = df[df['submit_time'] > 600]
            df['submit_time'] = df['submit_time'] - 600

            jobs = []
            for index, row in df.iterrows():
                d = np.array([row['cpus'], row['memory']]) * DataGenerator.num_demand_levels
                job = Job(p=row['processing_time'], r=row['submit_time'], w=row['priority'], d=d)
                jobs.append(job)

            with open(str(processed_jobs_pickle_path), 'wb') as f:
                pickle.dump(jobs, f)
        else:
            with open(str(processed_jobs_pickle_path), 'rb') as f:
                jobs = pickle.load(f)

        self.jobs = jobs[:self.N]

    def load_azure_2019_v2_jobs(self):
        # 2695548 jobs
        # 2551087 jobs with p > 0
        self.integer = False

        MAX_CPU = 64  # Number of cores
        MAX_MEM = 256  # Memory in GB

        resources = 2 if not self.R else self.R

        def path_per_resource(resource):
            return Path(__file__).parent / Path(f'data/azure_2019_v2_{resource}.pickle.lz4')

        if not path_per_resource(resources).exists():
            headers = ['vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 'vmdeleted', 'maxcpu', 'avgcpu',
                       'p95maxcpu', 'vmcategory', 'vmcorecountbucket', 'vmmemorybucket']
            df = pd.read_csv('data/azure_2019_v2/vmtable.csv.gz', header=None, index_col=False, names=headers, delimiter=',')

            # Transform vmcorecount '>24' bucket to 30 and '>64' to 70
            max_value_vmcorecountbucket = 30
            max_value_vmmemorybucket = 70
            df = df.replace({'vmcorecountbucket': '>24'}, max_value_vmcorecountbucket)
            df = df.replace({'vmmemorybucket': '>64'}, max_value_vmmemorybucket)

            df = df.astype({
                "vmcorecountbucket": float,
                "vmmemorybucket": float,
                'vmdeleted': float,
                'vmcreated': float
            })

            df['vmcorecountbucket'] /= MAX_CPU
            df['vmmemorybucket'] /= MAX_MEM

            category_to_weight = {
                'Delay-insensitive': 1,
                'Interactive': 10,
                'Unknown': 5
            }

            # Transform categories to weights
            df['weight'] = df['vmcategory'].map(category_to_weight)

            df['lifetime'] = df['vmdeleted'] - df['vmcreated']

            df = df[df['lifetime'] > 0]

            df = df.sort_values('vmcreated')

            jobs = []
            for j, row in tqdm(df.iterrows(), total=len(df), desc='2'):
                d = np.array([row['vmcorecountbucket'], row['vmmemorybucket']]) * DataGenerator.num_demand_levels
                job = Job(p=row['lifetime'], r=row['vmcreated'], w=row['weight'], d=d)
                jobs.append(job)

            with lz4.frame.open(str(path_per_resource(resources)), 'wb') as f:
                pickle.dump(jobs, f)

        else:
            with lz4.frame.open(str(path_per_resource(resources)), 'rb') as f:
                jobs = pickle.load(f)

        self.jobs = jobs[:self.N]

    def load_alibaba_2018_jobs(self):
        # 4M jobs
        self.integer = False

        resources = 2 if not self.R else self.R
        CPU_MAX = 96
        MEM_MAX = 100

        def path_per_resource(resource):
            return self.data_location / Path(f'alibaba_2018_{resource}.pickle.lz4')

        if not path_per_resource(resources).exists():
            df = pd.read_parquet('data/df_alibaba_trace_2018_4M.parquet')

            df = df[(df['start_time'] > 10000) & (df['end_time'] - df['start_time'] > 0)]

            df = df[~df['plan_mem'].isnull()]
            df = df[~df['plan_cpu'].isnull()]

            p_min = df['start_time'].min()
            df['start_time'] -= p_min
            df['end_time'] -= p_min
            df['processing_time'] = df['end_time'] - df['start_time']

            df['plan_cpu'] /= df['plan_cpu'].max()
            df['plan_mem'] /= df['plan_mem'].max()

            df = df[:4_000_000]

            jobs = []
            for j, row in tqdm(df.iterrows(), total=len(df), desc='2'):
                d = np.array([row['plan_cpu'], row['plan_mem']]) * DataGenerator.num_demand_levels
                job = Job(p=row['processing_time'], r=row['start_time'], w=1, d=d)
                jobs.append(job)

            with lz4.frame.open(str(path_per_resource(resources)), 'wb') as f:
                pickle.dump(jobs, f)

        else:
            with lz4.frame.open(str(path_per_resource(resources)), 'rb') as f:
                jobs = pickle.load(f)

        self.jobs = jobs[:self.N]

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
            return self.data_location / Path(f'azure_packing_2020_{resource}.pickle.lz4')

        if not path_per_resource(resources).exists():
            dat = sqlite3.connect(self.data_location / Path('packing_trace_zone_a_v1.sqlite'))
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

            D = np.zeros(shape=(len(df_vm), resources))

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

        self.jobs = jobs
        self.R = resources

