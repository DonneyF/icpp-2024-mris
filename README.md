# MRIS operating systems reproduction

Initial audit and native Mac setup, 2026-10-29. No paper figures, slides, or final report have been generated.

Original research: Donney Fan and Ben Liang, *Online Non-preemptive Multi-Resource Scheduling for Weighted Completion Time on Multiple Machines*, ICPP 2024. [Paper](https://www.comm.utoronto.ca/~liang/publications/ICPP2024_FL.pdf) · [Original code](https://github.com/DonneyF/icpp-2024-mris) · [Artifact v1](https://zenodo.org/records/12028115).

Baseline commit: `a35216c9f2511d905b71e31dffb445d6d58220e9`. The scheduler and experiment implementation files are unchanged. The upstream README is preserved in `docs/UPSTREAM_README.md`. No LICENSE file was found in this checkout; no new license is asserted for the authors' code.

Start with [technical audit](docs/TECHNICAL_AUDIT.md), [Mac setup](docs/MAC_SETUP.md), and [test evidence](docs/TEST_RESULTS.md). Then see [experiments](docs/EXPERIMENTS.md), [figure plan](docs/FIGURE_REPRODUCTION.md), [GitHub workflow](docs/GITHUB_WORKFLOW.md), and [team contributions](docs/TEAM_CONTRIBUTIONS.md).

```sh
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py
```

This runs only 16 synthetic jobs and needs no downloaded trace. It is a functionality check, not research reproduction. Use Python 3.12 and install the dependencies first as described in the setup guide.

Local structure: `scheduler/`, `packing/`, and `experiments/` are upstream; `scripts/` holds team validation; `docs/` holds plans; `evidence/` holds small test records. Datasets, environments, caches, and generated experiment results are ignored by Git. Team fork: https://github.com/Vishwabelaganti/mris-os-research-reproduction. The owner will invite collaborators through GitHub.
