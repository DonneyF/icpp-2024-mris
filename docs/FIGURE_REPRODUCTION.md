# Three-figure reproduction plan

Source: [ICPP 2024 paper](https://www.comm.utoronto.ca/~liang/publications/ICPP2024_FL.pdf), page 9. Captions below are paraphrased. All three compare MRIS-WSJF, CA-PQ-WSJF, PQ-WSJF, Tetris, and BF-EXEC using AWCT. No reproduction figures have been generated.

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
