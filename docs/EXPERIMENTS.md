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

## Next: bounded serial Azure pilot

1. Use the checksummed SQLite source described in MAC_SETUP.md.
2. Add a separately reviewed bounded loader/serial runner without editing the scheduler algorithms. Match upstream filtering, priority transformation, storage merging, machine capacities and time units. Record intentional deviations from full-data sampling. Neither simply increasing downsampling nor replacing the pool alone removes full-load memory cost.
3. Begin with 100–1000 trace jobs and one scheduler at a time. Freeze selected job IDs, preprocessing seed, downsample offsets and resource augmentation. Check validity before increasing size.
4. Compare all five paper schedulers with identical deep-copied inputs. AWCT is `sum(w*C)/N`; use the upstream label mapping in FIGURE_REPRODUCTION.md.
5. Save raw outputs and measure memory/time. Scale only after reviewing memory pressure. Use university HPC if full preprocessing exceeds laptop capacity.
6. Generate only clearly labeled reduced-scale preliminary curves until published parameters and repeated trials can be run.

Published-scale comparisons require 10 sampled offsets per point and mean/95% confidence intervals. One run does not support a confidence band. A seed and exact selection manifest are required for repeatability, though different solver versions may resolve tied optima differently. Full Azure preprocessing, published sweeps and their runtimes remain NOT TESTED.
