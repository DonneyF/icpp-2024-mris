# Three-figure reproduction plan (Figures 3, 4, 6)

Source: [ICPP 2024 paper](https://www.comm.utoronto.ca/~liang/publications/ICPP2024_FL.pdf), section 7 (pages 7–9). Figures 3–6 are on page 9. Captions and the method were re-read from the PDF on 2026-10-29. All three figures compare the same five schedulers on the Azure packing trace. The metric is AWCT = Σ w·C / N, plotted as the mean over 10 sampled job sets with a shaded 95% confidence interval (section 7.1).

The figure-to-experiment assignments in earlier documentation were correct. The only correction is Figure 6's job count: use **factor 64 (64,000 jobs)** as primary. See [FIGURE6_JOB_COUNT.md](FIGURE6_JOB_COUNT.md) for the evidence.

## Verified mapping

| | Figure 3 | Figure 4 | Figure 6 |
|---|---|---|---|
| Paper caption | Effect of job arrival rate on AWCT, with M = 20 machines | Effect of the number of machines on AWCT, with N = 64000 jobs | Effect of the number of resource types on AWCT, with M = 20 machines and N = 64000 jobs |
| x-axis | Jobs released in 12.5 days: 8,000 / 16,000 / 32,000 / 64,000 / 128,000 / 256,000 | Machines: 5 / 10 / 20 / 40 | Resource types: 4, 5, …, 20 |
| Fixed parameters | M = 20, R = 4 | N = 64,000 (factor 64), R = 4 | N = 64,000 (factor 64), M = 20 |
| Downsample factors | 512, 256, 128, 64, 32, 16 | 64 | 64 (upstream `run.sh` uses 128; see FIGURE6_JOB_COUNT.md) |
| Upstream script | `experiments/3_scheduler_benchmark_jobs.py` | `experiments/4_scheduler_benchmark_machines.py -m M` | `experiments/5_scheduler_benchmark_resources.py -r R` |
| Upstream sweep | `./run.sh 3 10` | `./run.sh 4 10` | `./run.sh 5 10` runs factor 128. Not caption-matched. |
| Input dataset | Azure packing trace zone A v1 (`packing_trace_zone_a_v1.sqlite`, MD5 `fd7a6e12…`). First 4.096M jobs by start time (last release ≈ 12.39 days), every *f*-th job after offset Δ. Same for all three figures. | same | Same, plus synthetic resource columns 5..R. Each column is drawn uniformly from all jobs' CPU demands (section 7.5.3). |
| Runs per point | 10 offsets Δ drawn without replacement from [0, f) | same | same |
| Schedulers (code → paper label) | `MRIS(sort='WSJF')` → MRIS-WSJF; `PriorityQueue(sort='WSJF', collect=True)` → CA-PQ-WSJF; `OnlinePriorityQueue(sort='WSJF')` → PQ-WSJF; `TetrisScheduler()` → Tetris; `BFEXECScheduler()` → BF-EXEC | same | same |
| Raw output | `3_scheduler_benchmark_jobs_{f}_{run}.parquet` | `4_scheduler_benchmark_machines_{M}_{run}.parquet` | `5_scheduler_benchmark_resources_{R}_{run}.parquet` (the factor is not in the name) |
| Plot function | `scheduler_benchmark_jobs(4096000, np.array([512,256,128,64,32,16]), 10)` | `scheduler_benchmark_machines(np.array([5,10,20,40]), 10)` | `scheduler_benchmark_resources(np.arange(4,21), 10)` |
| PNG | `3_scheduler_benchmark_jobs.png` (+ `_queuing_delay.png` = Figure 5) | `4_scheduler_benchmark_machines.png` | `5_scheduler_benchmark_resources.png` |
| Published trend | Tetris, PQ and BF-EXEC beat MRIS below ~45k jobs. MRIS is best above that and ≈½ of the others at 256k. CA-PQ is worst until ~128k. | MRIS is best at 5–20 machines (≈½ of Tetris at M = 5). The online baselines overtake MRIS at 40. CA-PQ is worst from 10 machines up. | MRIS rises ≈17% from R = 4 to 20. Tetris rises ≈80%. Ranking: MRIS < PQ ≈ BF-EXEC ≈ Tetris < CA-PQ. |
| Simulations at paper scale | 6 × 10 × 5 = 300 scheduler runs, up to 256,000 jobs each | 4 × 10 × 5 = 200, 64,000 jobs each | 17 × 10 × 5 = 850, 64,000 jobs each |
| Measured cost on 8 GB M2 | Reduced run: 1,000–8,000 jobs, 2 offsets, 1,001 s total, peak RSS 1.26 GB | Reduced run: 4,000 jobs, M = 2–40, 2 offsets, 816 s, peak RSS 1.17 GB | Reduced runs: 4,000 jobs, M = 20, R = 4–20: 1,001 s, 1.05 GB; M = 2 load variant: 516 s, 1.51 GB. Building the 64k and 32k inputs (no scheduling) peaked at 1.18 GB. |
| Exact reproduction on 8 GB MacBook Air? | **No.** MRIS alone took 182–230 s per 8,000-job run (3 runs) and grew roughly as N^2.1–2.6 between 2,000 and 8,000 jobs. Extrapolated, not measured: ~4–12 h per 64k-job MRIS run, far more at 256k. | **No.** Same 64k-job cost. 40 MRIS runs at an extrapolated ~4–12 h each. | **No.** 170 MRIS runs at 64k jobs. Needs HPC or many machines in parallel. |

Upstream's own runner (`run.sh` + `multiprocessing.Pool`) loads the full 4.66M-job trace into every worker. The upstream README estimates 2–8 GB per scheduler. Do not use it on an 8 GB laptop. `scripts/azure_pilot.py` is the memory-bounded serial alternative and calls the same scheduler classes. Its documented differences are listed in AZURE_VALIDATION.md.

## Commands

All commands run from the repository root with `.venv` active. Each output directory must be new. Keep factor-64 and factor-128 Figure 6 results in separate directories.

Reduced-scale validation (what was run; ~35–45 min serial):

```sh
scripts/run_reduced_figures.sh results/reduced-<date> fig3 fig4 fig6 fig6-config
MPLBACKEND=Agg python scripts/plot_reduced_figures.py results/reduced-<date>
```

A single paper-configuration point, one offset per invocation. This is slow on a laptop, so run one at a time in the background. `--max-jobs` must be raised deliberately.

```sh
# Figure 3, 16,000-job point
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/azure_pilot.py --experiment jobs --factors 256 --runs 1 --output-dir results/fig3-f256
# Figure 4, M = 5 at the published 64,000 jobs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/azure_pilot.py --experiment machines --factors 64 --machines 5 --runs 1 --max-jobs 64000 --output-dir results/fig4-f64-m5
# Figure 6, primary (caption) and upstream-script variants
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/azure_pilot.py --experiment resources --factors 64 --resources 4 --runs 1 --max-jobs 64000 --output-dir results/fig6-f64-r4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/azure_pilot.py --experiment resources --factors 128 --resources 4 --runs 1 --max-jobs 64000 --output-dir results/fig6-f128-r4
```

For full-scale runs on HPC, the original per-point commands are unchanged. Pass `--downsample_factor 64` explicitly for Figure 6:

```sh
PYTHONPATH="$PWD" python experiments/3_scheduler_benchmark_jobs.py --downsample_factor 512 --downsample_offset 0 --run 1
PYTHONPATH="$PWD" python experiments/4_scheduler_benchmark_machines.py --downsample_factor 64 --downsample_offset 0 --run 1 -m 5
PYTHONPATH="$PWD" python experiments/5_scheduler_benchmark_resources.py --downsample_factor 64 --downsample_offset 0 --run 1 -r 4
```

To plot a complete set of 10 runs, set `plots.PATH_TO_RESULTS` to the results directory and call the function from the table above. Or run `scripts/plot_reduced_figures.py`, which reads its point list from each `summary.json`. Do not run the all-plots entry point against partial results.

## Status labels

| Label | Meaning | Figures |
|---|---|---|
| Full paper reproduction | Published sizes, all points, 10 offsets | None yet |
| Reduced-scale experimental validation | Real Azure trace and the unchanged five schedulers, but fewer jobs or offsets | 3 (matches the paper at 8,000 jobs), 4 (light-load side only), 6 (trend appears only in the M = 2 load variant). See AZURE_VALIDATION.md. |
| Functionality test | Code path runs and outputs are feasible; no trend claim | Azure pilot at 1,000 jobs; Figure 6 64k/32k input build (dry run) |
| Not yet tested | — | Any 64,000+ job scheduling run; Figure 6 factor-64 vs factor-128 comparison at full size; Docker; Intel |

Never label reduced-scale results as paper reproductions.

## Notes kept from the earlier plan

- Source labels: `PQ-WSJF` in the code is the paper's CA-PQ-WSJF. `OnlinePQ-WSJF` is the paper's PQ-WSJF. `Tetris-instantaneous` is Tetris.
- MRIS defaults to a binary knapsack solved by GLPK_MI. The paper describes CADP. Keep the released code and report the distinction.
- Resource augmentation and machine-type sampling are stochastic. Upstream does not seed them, so only statistical agreement is possible, not exact equality.
- Ownership: Member 2 owns Figure 3 (`experiments/jobs`), Member 3 Figure 4 (`experiments/machines`), Member 4 Figure 6 (`experiments/resources`). Member 1 owns shared environment, pilot and memory evidence. See TEAM_CONTRIBUTIONS.md.
