# Experiment execution plan

## Completed setup scope

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py` from the checkout with the virtual environment active. It executes unchanged MRIS-WSJF and OnlinePriorityQueue-WSJF serially, twice each, on fresh identical seeded inputs. It checks assignment completeness, release-time feasibility, capacity at event boundaries, and exact repeatability. Outputs are JSON plus console/timing evidence. This is not an Azure experiment.

## Next gate: serial Azure pilot

1. Download and checksum the source database as described in MAC_SETUP.md.
2. Add a separately reviewed serial runner and bounded trace-loading path. Preserve all upstream scheduling methods; document any difference from full-data preprocessing. Do not simply raise downsample factor and call the existing multiprocessing scripts on an 8 GB laptop.
3. Start with 100–1,000 trace jobs, one scheduler at a time. Save exact selected job IDs, seed, preprocessing configuration, commit, package versions, command, and raw schedule outputs.
4. Time each command using `/usr/bin/time -l`, with BLAS thread counts set to 1. Capture whole-process wall/user/system times and peak RSS. Stop on sustained memory pressure; scale only after reviewing the measurement.
5. Compare all five paper baselines on identical deep-copied inputs. Check feasibility and metric definition before plotting. AWCT is `sum(w * C) / N`, not divided by sum of weights.
6. Repeat using the same input manifest; compare normalized schedule tables and metrics, not binary parquet bytes. Then generate three clearly labeled preliminary figure sweeps. Full-data execution and plots are deliberately deferred from the initial setup.

For published-scale runs, preserve 10 sampled offsets per point, selected without replacement, and produce mean/95% intervals. Dataset seed, offsets, checksums, and hardware belong in the experiment manifest. Use a university machine if full-data loading cannot fit on the laptop. Raw results stay outside Git; retain small manifests and summary metrics.
