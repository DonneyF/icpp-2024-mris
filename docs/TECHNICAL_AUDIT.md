# Initial technical audit

Audited 2026-10-29 at upstream commit `a35216c9f2511d905b71e31dffb445d6d58220e9`.

## Implementation and environment

- Python simulator; upstream specifies Python 3.12 and Docker pins 3.12.4. Pattern matching requires at least 3.10 syntactically, but older runtimes are not validated. This session uses isolated Python 3.12.14, native arm64, macOS 26.3. Default system Python 3.15.2 was not used.
- `scheduler/mris.py`: MRIS, MRISOffline, MRISGreedy. Default MRIS uses CVXPY GLPK_MI; Gurobi is imported unconditionally. `scheduler/pq.py`: offline/collect-all and online priority queues. `scheduler/tetris.py` and `scheduler/bf_exec.py`: comparison schedulers. Job/machine/state definitions are in the corresponding files under `scheduler/`.
- Direct dependencies are listed in `requirements-mac.txt`; exact installed versions are recorded in `requirements-mac.lock.txt`. Upstream's `tdqm` entry does not match its `tqdm` imports. The team requirements explicitly use tqdm. Latest installed PyArrow 26.0.0 failed at import with NumPy 1.26.4; PyArrow 17.0.0 passed imports and Parquet round-trip and is pinned in team requirements. Notebook UI is omitted from the runtime installation and can be installed separately.
- GLPK is the free solver path. CVXOPT provides GLPK support to CVXPY. Gurobi remains installed for the unchanged import but no Gurobi solve/license activation is required by our GLPK tests. Its bundled solver license is size limited; do not silently switch full experiments to it. [CVXPY install](https://www.cvxpy.org/install/) · [Gurobi licensing](https://www.gurobi.com/product/pricing-and-licensing).
- Docker is optional, not needed for the native test. The upstream publishing workflow targets linux/amd64 and linux/arm64; this is configuration, not evidence of successful builds here.

## Data and resources

`experiments/tools.py` loads Azure Packing 2020 from `experiments/data/packing_trace_zone_a_v1.sqlite`, generating `azure_packing_2020_R.pickle.lz4` caches. The cited Zenodo v1 record lists a 181.1 MB SQLite file. The downloader uses a Backblaze mirror and wget. Follow-up verification downloaded the 181,116,928-byte SQLite source, matched its published MD5 and passed SQLite quick_check. Full preprocessing is still not executed; see evidence/dataset-verification.json.

Critical: the loader reads full database tables, constructs all jobs, and only then experiment scripts downsample. Increasing the downsample factor does not remove the full-load RAM cost. Workers receive copied datasets and the default pool size is CPU count. Upstream estimates 2–8 GB per scheduler, so default parallel runs are unsuitable as an initial test on an 8 GB laptop. Exact peak RAM and runtime of full experiments are unknown. The installed virtual environment measured about 521 MB; total dataset caches/result storage is unmeasured. Reserve several GB as a planning allowance, not a measured minimum.

## Reproduction risks

1. `run.sh` experiment 1 has its Python invocation commented out: it can print progress without computing results.
2. `run.sh` experiment 5 uses factor 128 (32,000 jobs), unlike Figure 6's caption (64,000 jobs). Track the discrepancy explicitly before selecting factor 64 for a caption-matched run.
3. Random preprocessing uses NumPy without a fixed seed; run.sh uses unseeded random offsets. Fix and record both preprocessing seed and offset manifest, and share checksummed caches before comparing machines.
4. `generate_all_plots.py` expects all experiments' inputs by default and uses confidence intervals. Invoke only the selected plot functions; a single run cannot establish a 95% confidence interval.
5. No LICENSE file found in GitHub checkout. Preserve attribution/history, prefer a GitHub fork, and clarify reuse terms before repackaging/relicensing. Zenodo v1 is not assumed identical to current GitHub main.
6. Static inspection suggests the WERF branch may use an unfiltered release vector; untested variants are not validated by WSJF smoke tests. No algorithm changes were made.

## Assessment

Native ARM64 dependency installation and pip consistency checks succeeded. See TEST_RESULTS.md for actual algorithm results. Intel Macs, Docker, full trace loading, all variants, and published-figure reproduction remain NOT TESTED. First advance to a bounded serial trace pilot with memory measurement; consider university HPC if full data loading exceeds laptop capacity.

Follow-up: hardware verified as an 8 GB M2 MacBook Air (Mac14,2), 8 cores. Native 128-job serial simulation and Linux hosted smoke checks passed. Full trace/figure memory and runtime are still unknown. The released default uses a GLPK integer knapsack solve rather than an explicit CADP routine; reproducing the code does not independently verify the paper's theoretical runtime.
