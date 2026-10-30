# Azure pilot and reduced-scale Figure 3/4/6 validation

Executed 2026-10-29 on the 8 GB M2 MacBook Air (macOS 26.3, Python 3.12.14, `.venv` from `requirements-mac.lock.txt`), using the checksummed Azure trace (`packing_trace_zone_a_v1.sqlite`, MD5 `fd7a6e12…`). Base commit `5de7768`. Scheduler, packing and upstream experiment code are unchanged (`git diff a35216c -- scheduler packing experiments/*.py experiments/*.ipynb run.sh` is empty).

**What this is:** a functionality test and reduced-scale experimental validation on the real trace with the original schedulers. **What this is not:** a paper reproduction. Every point uses fewer jobs than the paper, or 2 offsets instead of 10, or both.

## Runner

`scripts/azure_pilot.py` builds jobs exactly as `experiments/tools.py::load_azure_packing_2020` does:

- drop rows with NULLs (only `endtime` has them) and rows with `starttime < 0`;
- weight = priority + 1;
- p = end − start (days);
- merge HDD/SSD into one storage column;
- give each VM a uniformly sampled machine-type row, with demands × 10;
- sort with the same `sort_values('starttime')` call;
- take every *f*-th job from offset Δ, limited to 4,096,000/*f* jobs;
- use machines of capacity 10 + 1e-8 per resource.

It then runs `MRIS(sort='WSJF')`, `PriorityQueue(sort='WSJF', collect=True)`, `OnlinePriorityQueue(sort='WSJF')`, `TetrisScheduler()` and `BFEXECScheduler()` on deep copies, one after another. Parquet output uses the upstream schema and file names.

Documented differences from upstream:

| Upstream | Bounded runner | Effect |
|---|---|---|
| Builds 4.66M `Job` objects; the comment estimates ~2 GB, and the README says 2–8 GB per worker | Holds the trace as NumPy arrays; builds only the selected jobs | Peak RSS ≈1.0–1.3 GB in every run here |
| `multiprocessing.Pool(cpu_count)`, each worker with a dataset copy | Serial, one scheduler at a time | No concurrent copies |
| Unseeded `np.random.choice` per job for machine type and augmentation, cached once | `np.random.default_rng(seed)` for machine type. Augmentation uses `default_rng([seed, 1, trace_index])`, so a job's extra demands do not depend on which offset selected it. Columns are nested across R, like the upstream caches. | Same distributions. Not bit-identical to the authors' (unrecoverable) random state. |
| `run.sh` offsets: unseeded `random.sample(range(f), runs)` | `random.Random(seed).sample(range(f), runs)` per point, recorded in `summary.json` | Repeatable |
| `SELECT * FROM vm` (no ORDER BY) | `... ORDER BY vmId` | Verified identical: the query plan is a primary-key scan, and the first 200,000 ids match |

Every run also checks:

- each job is scheduled exactly once, on a valid machine;
- no job starts before its release;
- per-machine capacity holds at every event (sweep line).

Statuses: `PASS` means all checks pass. `PASS_WITH_WARNINGS` means the only failure is the known collect-all PQ overlap described below. `FAIL` covers anything else, including the RSS watchdog stopping the run (`--max-rss-mb`, default 2,500).

## Task 2: Azure pilot

Seed 2026, M = 20, R = 4, all five schedulers. Evidence is in `evidence/azure-pilot/`.

| Run | Jobs (factor, offset) | Whole process | Peak RSS | Status |
|---|---|---|---|---|
| 1 | 1,000 (*f* = 4096, Δ = 975) | 18.7 s real, 14.9 s user | 1,221,984,256 B (1.14 GiB); peak footprint 1.47 GiB | PASS |
| 2 | 8,000 (*f* = 512, Δ = 121), the paper's smallest Figure 3 point | 408.5 s real, 395.2 s user | 1,255,047,168 B (1.17 GiB) | Recorded `FAIL` by the first validator. Re-validated as PASS_WITH_WARNINGS (see Errors). |

The trace has 5,559,800 VM rows, of which 4,664,759 remain after filtering. The 4,096,000th job is released at 12.386 days, matching the paper's "nearly 12.5 days". Loading the trace takes 8–10 s.

AWCT in days, from the pilot's single offset. Paper values are read by eye from Figure 3 at 8,000 jobs.

| Scheduler (paper label) | 1,000 jobs | 8,000 jobs | 8,000-job time | Paper Fig. 3 at 8,000 |
|---|---|---|---|---|
| MRIS-WSJF | 10.401 | 10.086 | 229.9 s | ≈10 |
| CA-PQ-WSJF (`PQ-WSJF` in code) | 14.820 | 15.590 | 24.4 s | ≈15.5 |
| PQ-WSJF (`OnlinePQ-WSJF`) | 6.966 | 7.169 | 89.4 s | ≈7 |
| Tetris | 6.966 | 7.203 | 38.7 s | ≈7 |
| BF-EXEC | 6.966 | 7.208 | 14.1 s | ≈7 |

**Result: success.** The original MRIS and four baselines ran on the real trace within about 1.2 GB. At the one published configuration that fits a laptop, the AWCT values and the ranking agree with the paper.

## Task 4: reduced-scale figure tests

Command: `scripts/run_reduced_figures.sh results/reduced-2026-10-29 fig3 fig4 fig6 fig6-config` (and `fig6-loaded` afterwards), then `scripts/plot_reduced_figures.py results/reduced-2026-10-29`. Seed 2026 and 2 offsets per point throughout. Offsets, per-run AWCT, per-scheduler times and checks are in `evidence/reduced-figures/*/summary.json`. Stamped PNGs and the AWCT CSVs are in `evidence/reduced-figures/figures/`.

| Sweep | Points | Scheduler runs | Wall (whole process) | Peak RSS | Status |
|---|---|---|---|---|---|
| Figure 3 | 1,000 / 2,000 / 4,000 / 8,000 jobs; M = 20; R = 4 | 40 | 1,001.2 s | 1.18 GiB (1,263,550,464 B) | PASS_WITH_WARNINGS |
| Figure 4 | 4,000 jobs; M = 2 / 5 / 10 / 20 / 40; R = 4 | 50 | 816.4 s | 1.09 GiB | PASS_WITH_WARNINGS |
| Figure 6 | 4,000 jobs; M = 20; R = 4 / 8 / 12 / 16 / 20 | 50 | 1,001.1 s | 0.97 GiB | PASS_WITH_WARNINGS |
| Figure 6 load variant (not a paper configuration) | 4,000 jobs; **M = 2**; R = 4 / 12 / 20 | 30 | 515.8 s | 1.40 GiB (1,508,212,736 B) | PASS_WITH_WARNINGS |
| Figure 6 inputs, dry run | 64,000 and 32,000 jobs; R = 4 / 20; no scheduling | 0 | 15.0 s | 1.10 GiB | PASS |

System memory stayed between 36% and 66% free (`memory_pressure`) before and after each sweep. The watchdog never fired. In every sweep, the warnings were the CA-PQ overlap only: 6 of 8 CA-PQ runs in Figure 3, 9 of 10 in Figure 4, 9 of 10 in Figure 6 and 6 of 6 in the load variant. Every one passed under the `np.isclose` tolerance.

Mean AWCT (days) over 2 offsets. The full min/max table is `awct_summary.csv`.

**Figure 3 (jobs, M = 20, R = 4)**

| Jobs | MRIS-WSJF | CA-PQ-WSJF | PQ-WSJF | Tetris | BF-EXEC |
|---|---|---|---|---|---|
| 1,000 | 10.386 | 14.672 | 6.929 | 6.929 | 6.929 |
| 2,000 | 9.706 | 14.356 | 6.729 | 6.729 | 6.729 |
| 4,000 | 9.910 | 14.834 | 6.879 | 6.879 | 6.879 |
| 8,000 | 10.094 | 15.462 | 6.939 | 6.955 | 6.944 |

**Figure 4 (machines, 4,000 jobs, R = 4)**

| M | Jobs per machine | MRIS-WSJF | CA-PQ-WSJF | PQ-WSJF | Tetris | BF-EXEC | MRIS / PQ |
|---|---|---|---|---|---|---|---|
| 2 | 2,000 | 15.146 | 23.904 | 13.626 | 13.948 | 13.515 | 1.11 |
| 5 | 800 | 11.315 | 17.011 | 8.311 | 8.564 | 8.399 | 1.36 |
| 10 | 400 | 10.034 | 15.451 | 7.035 | 7.047 | 7.033 | 1.43 |
| 20 | 200 | 10.180 | 14.769 | 6.817 | 6.817 | 6.817 | 1.49 |
| 40 | 100 | 10.085 | 14.517 | 6.813 | 6.813 | 6.813 | 1.48 |

**Figure 6 (resource types, 4,000 jobs, M = 20)**

| R | MRIS-WSJF | CA-PQ-WSJF | PQ-WSJF | Tetris | BF-EXEC |
|---|---|---|---|---|---|
| 4 | 10.242 | 14.845 | 6.863 | 6.863 | 6.863 |
| 8 | 10.310 | 15.059 | 6.861 | 6.861 | 6.861 |
| 12 | 9.967 | 15.314 | 6.881 | 6.881 | 6.880 |
| 16 | 10.242 | 15.329 | 6.819 | 6.819 | 6.820 |
| 20 | 10.190 | 15.412 | 6.824 | 6.825 | 6.829 |

**Figure 6 load variant (4,000 jobs, M = 2, about 2,000 jobs per machine; not a paper configuration)**

The paper's Figure 6 runs at 3,200 jobs per machine. With 20 machines, 4,000 jobs are only 200 per machine. Using 2 machines restores most of the paper's load while keeping each run around 1 minute. The R = 4 point uses the same seed and offsets (243, 654) as the Figure 4 M = 2 point. It was produced by a separate process and gave identical AWCT values, which confirms repeatability across processes.

| R | MRIS-WSJF | CA-PQ-WSJF | PQ-WSJF | Tetris | BF-EXEC |
|---|---|---|---|---|---|
| 4 | 15.146 | 23.904 | 13.626 | 13.948 | 13.515 |
| 12 | 16.069 | 34.453 | 25.571 | 26.285 | 25.951 |
| 20 | 17.539 | 43.728 | 32.847 | 34.850 | 32.608 |
| Change 4 → 20 | **+15.8%** | +82.9% | +141.1% | **+149.9%** | +141.3% |

### Comparison with the published figures

- **Figure 3.** The 8,000-job point is the paper's own configuration with fewer offsets. It matches the paper to within about 0.3 AWCT for every scheduler, and the ranking matches: online baselines < MRIS < CA-PQ. Points below 8,000 jobs are outside the paper's range and stay flat. The paper's crossover at about 45,000 jobs, where MRIS becomes best, and its large advantage at 128k–256k **were not tested**: they need at least 16,000–256,000 jobs.
- **Figure 4.** As in the paper, AWCT falls as machines are added. CA-PQ is worst at every M here; in the paper it is worst from M = 10 up. The paper's main claim is that MRIS wins at 5–20 machines with 64,000 jobs, which is 3,200–12,800 jobs per machine. That is **not reproduced**: 4,000 jobs give at most 2,000 jobs per machine, the load of the paper's lightest point (M = 40, 1,600 per machine). There the paper also shows the online baselines beating MRIS (≈12 vs ≈14, a ratio of 0.86). Our M = 2 point gives 13.6 vs 15.1 (0.90). The MRIS/PQ ratio falls steadily as load rises (1.49 → 1.11), the direction the paper's crossover requires.
- **Figure 6.** At 4,000 jobs on 20 machines, extra resource types change almost nothing: MRIS stays at ≈10.2, the online baselines at ≈6.85, and CA-PQ rises 3.8%. The paper reports Tetris +80% and MRIS +17% at 3,200 jobs per machine. With 200 jobs per machine, almost every job starts on release, so packing quality does not matter. The **published trend is not reproduced at this load**. In the M = 2 load variant (about 0.6× the paper's per-machine load), **the qualitative trend appears**: MRIS rises 15.8% from R = 4 to 20 (paper: 17%), while Tetris rises 150% (paper: 80%), and PQ and BF-EXEC rise 141%. MRIS goes from second-best at R = 4 to best by a wide margin at R = 12 and 20, matching the paper's ranking from R = 4 upward. The baselines grow more than in the paper. That variant uses 2 machines and 4,000 jobs, so the magnitudes are not comparable.

### Status summary

| Item | Status |
|---|---|
| Figures 3, 4, 6 at published size and 10 offsets | **Not yet tested.** Not feasible on this laptop (see the runtime note). |
| Figure 3 at 8,000 jobs, 3 offsets in total (pilot plus 2) | Reduced-scale validation of a published point. Matches the paper. |
| Figures 3, 4, 6 at 1,000–8,000 jobs, 2 offsets; Figure 6 M = 2 load variant | Reduced-scale experimental validation (the load variant is not a paper configuration) |
| Azure pilot at 1,000 jobs; Figure 6 64k/32k input build | Functionality tests |
| Figure 6 factor-64 vs factor-128 scheduling | **Not yet tested.** Inputs are prepared and verified. |

### Runtime outlook

MRIS dominates the runtime: 0.5–0.6 s at 1,000 jobs, 3.9–6.3 s at 2,000, 43–51 s at 4,000 and 182–230 s at 8,000. That growth is about N^2.1–2.6. Extrapolating, which was not measured, gives roughly 4–12 h per 64,000-job MRIS run. A complete Figure 6 is 170 such runs; Figure 4 is 40. The baselines are faster (12–89 s at 8,000 jobs) but also superlinear. A published-scale sweep needs HPC or many machines working in parallel, each one running a single point with `--max-jobs 64000`.

Evidence notes:

- The pilot manifests record runner SHA-256 `6614c993…`, an earlier revision with the same loader and scheduling code but without the tolerance check and `--dry-run`.
- All reduced-scale manifests record the committed runner, `5cbac9bb…`.
- The local checkout path was removed from `argv` in the committed manifests.
- Parquet files are not committed (`*.parquet` is ignored). Their SHA-256 hashes are, and the files can be regenerated with the same seed and commands.

## Errors and fixes

1. **No PDF tools.** `pdftotext` and `pdftoppm` are not installed. The paper was read with macOS PDFKit through a throwaway Swift script in a scratch directory. Nothing was installed or committed.
2. **The first 8,000-job validator reported FAIL.** The cause is upstream `scheduler/pq.py:123`, used by the collect-all PQ (CA-PQ) baseline. After a job completes, every other job whose completion time is `np.isclose` (rtol 1e-5, atol 1e-8) is also freed. At about 12 offline days that is up to ~1.2e-4 days early. The observed overlap was 2.2e-7 days, with machine load reaching 10.98 against a capacity of 10. Across the sweeps, the largest instantaneous excess was 3.23 of 10 units at M = 20 and 6.54 of 10 in the M = 2 load variant. All other schedulers were feasible. Fix applied: the validator now re-checks with that tolerance and reports `PASS_WITH_WARNINGS`. The saved schedules were re-validated without re-simulation (`evidence/azure-pilot/8000-jobs/revalidation.txt`). **Upstream code was not changed.** A possible upstream fix is exact equality or an absolute-only tolerance in that comparison. It would change a published baseline's numbers slightly, so it needs team approval and would have to be reported.
3. **Upstream full loader and Pool are unsuitable for 8 GB.** These were replaced for laptop runs by the bounded runner described above.

## Remaining limitations

- No published-size point at 16,000 jobs or more has been scheduled. The paper's MRIS-wins regime is untested.
- Only 2 offsets per point, so the 95% t-intervals (df = 1) are very wide. Treat the curves as trends only.
- Machine-type sampling and augmentation use our seed. Only statistical agreement with the paper is possible.
- Paper values are read visually from the PDF (±1 unit).
- MRIS uses the released GLPK_MI knapsack, not the paper's CADP description.
- The CA-PQ results include the upstream `np.isclose` overlap. It probably biases CA-PQ AWCT slightly downward; this has not been quantified.
- `run.sh 5` still runs factor 128. Use factor 64 for Figure 6 (see FIGURE6_JOB_COUNT.md).
