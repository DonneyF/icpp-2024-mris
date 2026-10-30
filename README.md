# MRIS operating systems reproduction

Four-student reproduction of Donney Fan and Ben Liang, *Online Non-preemptive Multi-Resource Scheduling for Weighted Completion Time on Multiple Machines*, ICPP 2024. [Paper](https://www.comm.utoronto.ca/~liang/publications/ICPP2024_FL.pdf) · [Original code](https://github.com/DonneyF/icpp-2024-mris) · [Artifact v1](https://zenodo.org/records/12028115).

[PR #1](https://github.com/Vishwabelaganti/mris-os-research-reproduction/pull/1) was reviewed, tested and merged. The research code under `scheduler/`, `packing/`, and `experiments/` remains unchanged from upstream commit `a35216c9f2511d905b71e31dffb445d6d58220e9`. The original README is preserved in [UPSTREAM_README.md](docs/UPSTREAM_README.md), with trailing whitespace removed. No upstream LICENSE file was found; this project does not assert a new license for the authors' code.

## Start here

1. Follow [Mac setup](docs/MAC_SETUP.md) using Python 3.12 and the tested lockfile.
2. Run the small serial validation below; no dataset or Docker is required.
3. Read [test results](docs/TEST_RESULTS.md), [experiment instructions](docs/EXPERIMENTS.md), and [figure plan](docs/FIGURE_REPRODUCTION.md).
4. Follow the [GitHub workflow](docs/GITHUB_WORKFLOW.md) and [four-member responsibilities](docs/TEAM_CONTRIBUTIONS.md).

```sh
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py --output-dir results/my-first-test
```

Use a new output directory for each run. Success writes `summary.json` with `status: PASS`, `inputs.json`, and one Parquet schedule for each scheduler. The runner refuses to overwrite existing output folders. It leaves tracked historical evidence untouched.

Native installation and small simulations have been tested on an **8 GB M2 MacBook Air**, macOS 26.3. A 128-job synthetic validation passed; this is not an Azure or paper-figure reproduction. The Azure database has been downloaded and checksummed, but full preprocessing and figure sweeps remain pending. No report or presentation is being drafted yet.

Team branches: `setup/mac-support`, `experiments/jobs`, `experiments/machines`, `experiments/resources`; use pull requests into `main`. Collaborator invitations are handled by the owner in GitHub. See the [issue backlog](https://github.com/Vishwabelaganti/mris-os-research-reproduction/issues).

`scripts/` contains team validation; `docs/` contains instructions; `evidence/` contains small reviewed records. `results/`, `experiments/data/`, environments and caches are ignored by Git. See [technical audit](docs/TECHNICAL_AUDIT.md) and [troubleshooting](docs/TROUBLESHOOTING.md) for limitations.
