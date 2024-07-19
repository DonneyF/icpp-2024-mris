# Machine Scheduler

A job scheduling simulator. We provide a bash script for structured replication, individual scripts for each experiment (simulation), as well as a Docker container image.

## Installation & Dependencies

This project is created using Python 3.12. A MIP solver is required, and by default this project select [GLPK](https://www.gnu.org/software/glpk/), which exists in the form of a `cvxpy` dependency. `gurobipy` is also accepted, but modifications to the `MRIS` class is required.

To install the dependencies run

```
python3 -m pip install -r requirements.txt
```

## Simulator Architecture

All the classes used by this project use [dataclasses](https://docs.python.org/3/library/dataclasses.html). To familiarize with the definition of each of the classes, see the `Scheduler`, `Job`, and `Machine` class definitions.

The `scheduler` module implements the schedulers and contains a list of machines, a list of jobs, and an id.

The `experiments` module defines the system environment for scheduler to run. This includes instantiating `Machine` and `Job` objects. The experiments files save the results as dataframes for interpretation later.

## System Requirements

Experiments use `multiprocessing` to run schedulers in parallel. Each scheduler to simulate requires a single core/thread. Per scheduler, it requires 2-8GB of RAM to read the entire dataset into memory and to process, depending on the number of jobs and resources. The scheduler outputs (dataframes in parquet format) are in the range of megabytes.

## Obtaining Data

The primary datasource is the [Azure Packing 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md) dataset. The Azure Packing dataset contains 5 distinct resource types, which is later reduced to 4 by combining the SSD/HDD values. As we also investigate the effect of the number of resource types, we also synthetically generate additional resource types. 
- See `experiments/tools.py` for specifics on how the dataset is imported and augmented.

The datasets should be stored in `experiments/data/`. We provide a script to download and automatically generate and compress these datasets (takes perhaps 30 minutes).

```bash
./download_data.sh [max number of resource types, default 20]
```

## Running Experiments

To replicate the results in the paper, we provide a `run.sh` that contains all the parameters used for each experiment (aside from number of runs). The scripts has two positional parameters.

```bash
./run.sh [EXPERIMENT_NUMBER] [NUMBER_OF_RUNS]
```

`EXPERIMENT_NUMBER` ranges from 1 to 6 and is described below. By default `NUMBER_OF_RUNS=1`.

There are six experiments. We also specify the number of schedulers these experiments process in parallel.

We can run experiments by directly calling certain Python scripts with arguments. We list the arguments here as well as provide example scripts for executing each experiment.

- `DOWNSAMPLE_FACTOR` (integer): Experiments load 4.096M of jobs from the Azure Packing Dataset. Due to runtime constraints it is not feasible to schedule all of these, so we evenly downsample these jobs by selecting every `DOWNSAMPLE_FACTOR` number of jobs to process. e.g. if `DOWNSAMPLE_FACTOR=64`, we process `4096000/64=64000` jobs in the experiment.
- `DOWNSAMPLE_OFFSET` (integer): This selects the offset starting from `0` of when to start including every `DOWNSAMPLE_FACTOR` job.
- `MACHINES`: The number of machines used in the simulation
- `RESOURCES`: The number of distinct resource types to simulate
- `RUN_NUMBER`: For multi-sample runs, we specify the run number to distinguish output files
- `JOBS`: The number of input jobs used for the simulation (only used in `6_adversarial`)

### Experiment Descriptions

Unless explicitly specified, `MACHINES=20` and `RESOURCES=4` are specified within the Python script for each experiment.

#### `1_MRIS_heuristics`

This explores different sorting heuristics for the `MRIS` scheduler (7 schedulers)

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD%/*}/scheduler"
python3 experiments/1_MRIS_heuristics.py --downsample_factor [DOWNSAMPLE_FACTOR] --downsample_offset [DOWNSAMPLE_OFFSET] --run [RUN_NUMBER]
```

#### `2_MRIS_knapsack`

This explores options for the knapsack solution of the `MRIS` scheduler (2 schedulers)

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD%/*}/scheduler"
python3 experiments/2_MRIS_knapsack.py --downsample_factor [DOWNSAMPLE_FACTOR] --downsample_offset [DOWNSAMPLE_OFFSET] --run [RUN_NUMBER]
```

#### `3_scheduler_benchmark_jobs`

We compare `MRIS` and a few other schedulers with a chosen sorting heuristic, sweeping the number of jobs (5 schedulers)

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD%/*}/scheduler"
python3 experiments/3_scheduler_benchmark_jobs.py --downsample_factor [DOWNSAMPLE_FACTOR] --downsample_offset [DOWNSAMPLE_OFFSET] --run [RUN_NUMBER]
```

#### `4_scheduler_benchmark_machines`

We compare `MRIS` and a few other schedulers with a chosen sorting heuristic, sweeping the number of machines (5 schedulers)

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD%/*}/scheduler"
python3 experiments/4_scheduler_benchmark_machines.py --downsample_factor [DOWNSAMPLE_FACTOR] --downsample_offset [DOWNSAMPLE_OFFSET] --run [RUN_NUMBER] -m [MACHINES]
```

#### `5_scheduler_benchmark_resources`

We compare `MRIS` and a few other schedulers with a chosen sorting heuristic, sweeping the number of resources (5 schedulers)

```bash
python3 experiments/5_scheduler_benchmark_resources.py --downsample_factor [DOWNSAMPLE_FACTOR] --downsample_offset [DOWNSAMPLE_OFFSET] --run [RUN_NUMBER] -r [RESOURCES]
```

#### `6_adversarial`

We provide a specific adversarial input to help illustrate an example where `MRIS` is dominant.

```bash
python3 experiments/6_scheduler_benchmark_resources.py -n [JOBS] -m [MACHINES] --run [RUN_NUMBER] -r [RESOURCES]
```

## Plotting Results

Associated with each experiment file is a Jupyter Notebook that interprets the output of the experiment dataframes.


## Docker

### Obtaining the image

One can build using the provided `Dockerfile`

```bash
docker build -t machine-scheduler .
```

or pull an image built using GitHub Actions (`linux/amd64` and `linux/arm64` images are provided):

```bash
docker pull gchr.io/donneyf/icpp-2024-mris:main
```

### Running experiments

There are two directories that need to be mounted from the host to the container, one directory contains datasets, and the other for experimental results.

#### Step 1. Obtaining Data

```bash
docker run \
  -v ./data:/app/experiments/data \
  -v ./results:/app/experiments/results \
  ghcr.io/donneyf/icpp-2024-mris:main ./download_data.sh
```

#### Step 2. Running experiments

We pass in two positional arguments to `run.sh` to execute experiments after obtaining data.

- `EXPERIMENT_NUMBER`: An integer [1-5] that maps to one of the experiments, as mentioned above
- `NUMBER_OF_RUNS`: Number of times to run each experiment, with a randomized `DOWNSAMPLE_OFFSET` for each run. Default `1`.

```bash
docker run \
  -v ./data:/app/experiments/data \
  -v ./results:/app/experiments/results \
  ghcr.io/donneyf/icpp-2024-mris:main ./run.sh [EXPERIMENT_NUMBER] [NUMBER_OF_RUNS]
```

### Viewing Results

We use the same Jupyter notebooks as above. The dependencies for plotting only are given below:

```bash
python3 -m pip install numpy<2 matplotlib pandas pyarrow scienceplots lz4 notebook
```

## Variations in Results

There some possibilities that there could be run to run variations.

For instance, given that sorting is used, there is a possibility that two jobs yield the same heuristic (i.e. same processing time for SJF), and thus their order in scheduling can be different. However we have not observed this in our experiments.

The `DOWNSAMPLE_OFFSET` factor is randomized across runs so that each run selects a different subset of jobs to schedule. However, as the downsampling occurs by evenly selecting every fixed number of jobs so as to evenly represent jobs released in the 12.5-day time window, we generally expect results to not vary wildly between one run to the next. However when the number of jobs being scheduled is small, this effect may be more pronounced.

The machine resources capacities are represented by floating point values and some jobs may be permitted to be scheduled when simulated on platform/library and not the other.