# ICPP 2024 MRIS

A job scheduling simulator

## Installation & Dependencies

This project is created using Python 3.12. A major requirement is the [Gurobi](https://www.gurobi.com/) solver, which takes on the form of `gurobipy` in this project.

To install the dependencies run

```
python3 -m pip install -r requirements.txt
```

## Simulator Architecture

All the classes used by this project use [dataclasses](https://docs.python.org/3/library/dataclasses.html). To familiarize with the definition of each of the classes, see the `Scheduler`, `Job`, and `Machine` class definitions.

The `scheduler` module implements the schedulers and contains a list of machines, a list of jobs, and an id.

The `experiments` module defines the system environment for scheduler to run. This includes instantiating `Machine` and `Job` objects. The experiments files save the results as dataframes for interpretation later.

## Obtaining Data

The primary datasource is the [Azure Packing 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md) dataset. Also provided is a sythentic data generator.

See `experiments/tools.py` for specifics on how the dataset is imported and augmented.

The datasets should be stored in `experiments/data/`

Due to randomness and sake of time, we also provide pre-augmented datasets, which can be downloaded using the following command:

```bash
DATA_PATH=experiments/data/

# Original Azure dataset
wget -P $DATA_PATH https://icpp-2024-mris.s3.us-west-001.backblazeb2.com/packing_trace_zone_a_v1.sqlite

# Pre-augmented datasets
for l in {4..20}; do
  wget -P $DATA_PATH "https://icpp-2024-mris.s3.us-west-001.backblazeb2.com/azure_packing_2020_$l.pickle.lz4"
done
```

## Launching an Experiment

To run an experiment, say `experiment1.py`, we perform the following:

```
PROJECT_PATH=./machine-scheduler/
export PYTHONPATH="${PYTHONPATH}:$PROJECT_PATH"
python3 experiment1.py
```

### System Requirements

Experiments use `multiprocessing` to run schedulers in parallel. Each scheduler to simulate typically requires a single core. Per scheduler, it requires 4-8GB of RAM to read the entire Azure dataset into memory and to process.

## Plotting Results

Associated with each experiment file is a Jupyter Notebook that interprets the output of the experiment dataframes.
