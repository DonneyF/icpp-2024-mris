#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD%/*}/scheduler"

MRIS_heuristics() {
  DOWNSAMPLE_FACTORS=(1024 512 256 128 64)
  for DOWNSAMPLE_FACTOR in "${DOWNSAMPLE_FACTORS[@]}"
  do
  OFFSETS=($(python3 -S -c "import random; print(' '.join(map(str, random.sample(range(0, $DOWNSAMPLE_FACTOR), $NUM_RUNS))))"))
     for ((i=1; i<=NUM_RUNS; i++))
     do
       DOWNSAMPLE_OFFSET=${OFFSETS[$((i-1))]}
       echo "Run $i of $NUM_RUNS for downsample factor $DOWNSAMPLE_FACTOR. Using offset $DOWNSAMPLE_OFFSET."
       # python3 experiments/1_MRIS_heuristics.py --downsample_factor $DOWNSAMPLE_FACTOR --downsample_offset $DOWNSAMPLE_OFFSET --run $i
     done
  done
}

MRIS_knapsack() {
  DOWNSAMPLE_FACTORS=(1024 512 256 128 64)
  for DOWNSAMPLE_FACTOR in "${DOWNSAMPLE_FACTORS[@]}"
  do
    OFFSETS=($(python3 -S -c "import random; print(' '.join(map(str, random.sample(range(0, $DOWNSAMPLE_FACTOR), $NUM_RUNS))))"))
     for ((i=1; i<=NUM_RUNS; i++))
     do
       DOWNSAMPLE_OFFSET=${OFFSETS[$((i-1))]}
       echo "Run $i of $NUM_RUNS for downsample factor $DOWNSAMPLE_FACTOR. Using offset $DOWNSAMPLE_OFFSET."
       python3 experiments/2_MRIS_knapsack.py --downsample_factor $DOWNSAMPLE_FACTOR --downsample_offset $DOWNSAMPLE_OFFSET --run $i
     done
  done
}

scheduler_benchmark_jobs() {
  DOWNSAMPLE_FACTORS=(512 256 128 64 32 16)
  for DOWNSAMPLE_FACTOR in "${DOWNSAMPLE_FACTORS[@]}"
  do
    OFFSETS=($(python3 -S -c "import random; print(' '.join(map(str, random.sample(range(0, $DOWNSAMPLE_FACTOR), $NUM_RUNS))))"))
     for ((i=1; i<=NUM_RUNS; i++))
     do
       DOWNSAMPLE_OFFSET=${OFFSETS[$((i-1))]}
       echo "Run $i of $NUM_RUNS for downsample factor $DOWNSAMPLE_FACTOR. Using offset $DOWNSAMPLE_OFFSET."
       python3 experiments/3_scheduler_benchmark_jobs.py --downsample_factor $DOWNSAMPLE_FACTOR --downsample_offset $DOWNSAMPLE_OFFSET --run $i
     done
  done
}

scheduler_benchmark_machines() {
  MACHINES=(5 10 20 40)
  DOWNSAMPLE_FACTOR=64
  for MACHINE in "${MACHINES[@]}"
  do
    OFFSETS=($(python3 -S -c "import random; print(' '.join(map(str, random.sample(range(0, $DOWNSAMPLE_FACTOR), $NUM_RUNS))))"))
     for ((i=1; i<=NUM_RUNS; i++))
     do
       DOWNSAMPLE_OFFSET=${OFFSETS[$((i-1))]}
       echo "Run $i of $NUM_RUNS for $MACHINE machines. Using offset $DOWNSAMPLE_OFFSET."
       python3 experiments/4_scheduler_benchmark_machines.py --downsample_factor $DOWNSAMPLE_FACTOR --downsample_offset $DOWNSAMPLE_OFFSET --run $i -m $MACHINE
     done
  done
}

scheduler_benchmark_resources() {
  RESOURCES=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
  DOWNSAMPLE_FACTOR=128
  for RESOURCE in "${RESOURCES[@]}"
  do
    OFFSETS=($(python3 -S -c "import random; print(' '.join(map(str, random.sample(range(0, $DOWNSAMPLE_FACTOR), $NUM_RUNS))))"))
     for ((i=1; i<=NUM_RUNS; i++))
     do
       DOWNSAMPLE_OFFSET=${OFFSETS[$((i-1))]}
       echo "Run $i of $NUM_RUNS for $RESOURCE resources. Using offset $DOWNSAMPLE_OFFSET."
       python3 experiments/5_scheduler_benchmark_resources.py --downsample_factor $DOWNSAMPLE_FACTOR --downsample_offset $DOWNSAMPLE_OFFSET --run $i -r $RESOURCE
     done
  done
}

adversarial() {
    python3 experiments/6_adversarial.py -n 2500 -m 1 --run 1 -r 3
}

EXPERIMENT="$1"
NUM_RUNS="${2:-1}"
DATA_LOCATION='./experiments/data'

if [ -z "$(ls -A $DATA_LOCATION)" ]; then
  echo "First run download_data.sh to obtain datasets"
  exit
fi

case "$EXPERIMENT" in
  1)
    echo "Executing MRIS_heuristics"
    MRIS_heuristics
    ;;
  2)
    echo "Executing MRIS_knapsack"
    MRIS_knapsack
    ;;
  3)
    echo "Executing scheduler_benchmark_jobs"
    scheduler_benchmark_jobs
    ;;
  4)
    echo "Executing scheduler_benchmark_machines"
    scheduler_benchmark_machines
    ;;
  5)
    echo "Executing scheduler_benchmark_resources"
    scheduler_benchmark_resources
    ;;
  6)
    echo "Executing adversarial"
    adversarial
    ;;
  *)
    echo "Experiment number needs to range from 1 to 6"
    ;;
esac