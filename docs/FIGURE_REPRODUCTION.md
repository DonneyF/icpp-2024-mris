# Three-figure reproduction plan

Source: [ICPP 2024 paper](https://www.comm.utoronto.ca/~liang/publications/ICPP2024_FL.pdf), page 9. The mappings were rechecked against the full paper during PR review. Captions below are paraphrased. All three compare MRIS-WSJF, CA-PQ-WSJF, PQ-WSJF, Tetris, and BF-EXEC using AWCT. No reproduction figures have been generated.

| Figure | Measurement | Script and parameters | Raw file under experiments/results | Plot function / PNG |
|---|---|---|---|---|
| 3 | Arrival-rate effect, 20 machines | `3_scheduler_benchmark_jobs.py`; factors 512,256,128,64,32,16; 4 resources | `3_scheduler_benchmark_jobs_{factor}_{run}.parquet` | `scheduler_benchmark_jobs(4096000, np.array([512,256,128,64,32,16]), 10)` / `3_scheduler_benchmark_jobs.png` |
| 4 | Machine-count effect, 64,000 jobs | `4_scheduler_benchmark_machines.py`; factor 64; machines 5,10,20,40; 4 resources | `4_scheduler_benchmark_machines_{M}_{run}.parquet` | `scheduler_benchmark_machines(np.array([5,10,20,40]), 10)` / `4_scheduler_benchmark_machines.png` |
| 6 | Resource-count effect, 20 machines and caption's 64,000 jobs | `5_scheduler_benchmark_resources.py`; resources 4–20; factor 64 for caption, versus upstream run.sh's 128 | `5_scheduler_benchmark_resources_{R}_{run}.parquet` | `scheduler_benchmark_resources(np.arange(4,21), 10)` / `5_scheduler_benchmark_resources.png` |

All require the Azure SQLite trace and the corresponding per-resource cache produced by `experiments/tools.py`. Resource augmentation is stochastic. Full runtime and peak memory have not been measured. Exact numerical equality is uncertain without the authors' random input state; statistical reproduction is the target.

These are exact upstream command examples for one point, NOT the recommended initial 8 GB execution path (they spawn worker pools):

```sh
PYTHONPATH="$PWD" python experiments/3_scheduler_benchmark_jobs.py --downsample_factor 512 --downsample_offset 0 --run 1
PYTHONPATH="$PWD" python experiments/4_scheduler_benchmark_machines.py --downsample_factor 64 --downsample_offset 0 --run 1 -m 5
PYTHONPATH="$PWD" python experiments/5_scheduler_benchmark_resources.py --downsample_factor 64 --downsample_offset 0 --run 1 -r 4
```

Expand each sweep with 10 recorded offsets per point only after the serial pilot. Separate factor-64 and factor-128 resource outputs into separate result directories, since filenames do not encode the factor.

Plot selectively after all required raw files exist: import `experiments.generate_all_plots as plots` and `numpy as np`, set `plots.PATH_TO_RESULTS` to the appropriate results directory, and call the corresponding function in the table. PNGs are saved in the current working directory. Matching `.ipynb` notebooks also exist. Do not run the all-plots entrypoint against partial results.

Cheapest preliminary order: reuse a small four-resource dataset for a jobs sweep, then machines, then resource augmentation. These reduced-scale curves cannot be labeled as full paper reproductions. Figure 5 is a possible fallback because it reuses the job benchmark schedules; its CDF is a distinct comparison, subject to instructor acceptance.

## Ownership and limits

Member 2 owns Figure 3 on `experiments/jobs`; Member 3 owns Figure 4 on `experiments/machines`; Member 4 owns Figure 6 on `experiments/resources`. Member 1 reviews shared environment and memory evidence. These assignments now correspond to verified published figures, not assumptions based only on script names.

Figure 3 job counts are 8,000, 16,000, 32,000, 64,000, 128,000 and 256,000. Figure 4 uses four machine counts, with 64,000 jobs. Figure 6 uses 4–20 resources with 20 machines. Keep factor 64 (caption-matched 64,000 jobs) separate from factor 128 (upstream shell-script 32,000 jobs); neither resolves the authors' intended experiment conclusively. Do not label an upstream-default 32,000-job curve as matching the caption.

Source labels: `PQ-WSJF` is the collect-all baseline (paper CA-PQ-WSJF), while `OnlinePQ-WSJF` is the paper PQ-WSJF. Preserve this mapping in analysis. The default implementation solves a binary knapsack through GLPK_MI, whereas the paper describes CADP; preserve released code and report that distinction instead of claiming to have independently validated the paper's theoretical runtime.

All three are candidates for eventual reproduction on the Mac after a memory-bounded runner is validated. The native small synthetic test passed on an 8 GB M2 MacBook Air; none of the full Azure figure sweeps has been run. Exact full-sweep RAM/runtime is unknown. Resource augmentation creates multiple caches and is likely the most demanding setup, but that ranking is not measured.

For a complete set of 10 runs, generate selected plots from the repo root:

```sh
mkdir -p results/figures
MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib" PYTHONPATH="$PWD" python - <<'PYTHON'
from pathlib import Path
import os
import numpy as np
from experiments import generate_all_plots as plots
plots.PATH_TO_RESULTS = Path('experiments/results').resolve()
os.chdir('results/figures')
plots.scheduler_benchmark_jobs(4096000, np.array([512,256,128,64,32,16]), 10)
plots.scheduler_benchmark_machines(np.array([5,10,20,40]), 10)
plots.scheduler_benchmark_resources(np.arange(4,21), 10)
PYTHON
```

This plotting command is prepared but NOT EXECUTED: its required raw results do not yet exist. Use a separate results directory for each incompatible parameter set; never mix preliminary runs with full-scale runs. Expected PNG names are listed above. No placeholder curves have been created.
