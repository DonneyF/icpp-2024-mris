# Initial simulations and reproduction workflow

## Repeat the actual small test

Activate `.venv` and run from the repository root. The output directory must not exist; the log parent may exist.

```sh
mkdir -p results/logs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -p python scripts/smoke_test.py \
  --jobs 128 --machines 2 --resources 4 --seed 2026 \
  --output-dir results/my-128-run \
  > results/logs/my-128-run.stdout.txt 2> results/logs/my-128-run.stderr.txt
```

Check the command's exit status with `echo $?` immediately afterward and read `results/my-128-run/summary.json`. PASS requires solver/import checks, complete assignments, release-time and resource feasibility, exact same-seed repeatability, and Parquet round trips. A failed run writes FAIL (or RUNNING if interrupted); it must never be reported as success. Choose another output folder to retry. Ctrl+C stops a run.

The runner calls unchanged `MRIS(sort='WSJF')` and `OnlinePriorityQueue(sort='WSJF')` serially, twice each, on freshly constructed identical synthetic inputs. Workloads are bounded to 1–1000 jobs, 1–40 machines and 2–20 resources; those limits are guardrails, not a guarantee that every parameter combination is cheap or validated.

Outputs: `inputs.json`, `summary.json`, `MRIS.parquet`, `OnlinePriorityQueue.parquet`. Parquet uses the upstream `results_as_dataframe` schema. The summary stores full schedules, AWCT, per-scheduler durations, total measured CPU/wall time, peak RSS, seed, dimensions, source commit, runner hash, input checksum and thread settings. Shell time includes process startup; Python timing starts before scientific imports. Do not compare timings from different scopes.

The reviewed 128-job evidence is under `evidence/initial-simulation/`, including inputs, schedules in summary.json, console/timing logs and generated-file checksums. The corresponding Parquet files remain local under ignored `results/initial-128/` and can be regenerated. These results do not reproduce Azure traces or any paper figure. Arbitrary synthetic time units are not Azure days.

## Azure trace runs

The bounded serial Azure pilot is implemented and passed: `scripts/azure_pilot.py` (one scheduler at a time, trace held as arrays, seeded sampling, RSS watchdog). Reduced-scale Figure 3/4/6 sweeps are driven by `scripts/run_reduced_figures.sh` and plotted by `scripts/plot_reduced_figures.py`. Commands, measured runtime/RAM, results, deviations from upstream and limitations are in [AZURE_VALIDATION.md](AZURE_VALIDATION.md); the figure/parameter mapping is in [FIGURE_REPRODUCTION.md](FIGURE_REPRODUCTION.md).

Published-scale comparisons require 10 sampled offsets per point and mean/95% confidence intervals. Two offsets give a very wide t-interval (t = 12.7 at df = 1), so reduced runs show trends, not significance. Full Azure figure sweeps at published size remain NOT TESTED.
