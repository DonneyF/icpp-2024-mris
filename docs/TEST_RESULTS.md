# Current verification and initial simulation

## PR review and merge

[PR #1](https://github.com/Vishwabelaganti/mris-os-research-reproduction/pull/1) was merged as `b9f469e5ae1e660bcc59bd98e2554d91058d870f` after reviewing all changed files, checking whitespace, rerunning imports/solver/simulation/Parquet checks, and confirming no changes to upstream algorithms, experiment scripts, original requirements or Dockerfile. The only review fix was trailing whitespace in the archived README. Reviewed head: `6c7aac4e1895fdb67292d6aac635f24d64e7749f`; both push and PR smoke checks passed on that head. Evidence from the local review is in `evidence/review/`.

## Hardware and 128-job run

Verified host: **8 GB M2 MacBook Air**, 8 cores, ARM64, macOS 26.3, Python 3.12.14. Non-sensitive hardware evidence is in `evidence/hardware.json`. The previous sandbox could not read hardware; a subsequent permitted read established it. Intel and Docker remain NOT TESTED.

Exact command used:

```sh
mkdir -p evidence/initial-simulation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -p .venv/bin/python scripts/smoke_test.py --jobs 128 --machines 2 --resources 4 --seed 2026 --output-dir results/initial-128 > evidence/initial-simulation/console.txt 2> evidence/initial-simulation/timing.txt
```

Expected: solver/import success; all jobs assigned once, no early starts or over-capacity intervals; repeated schedules agree; Parquet schedules write/read successfully. Actual: **PASS** for all checks.

| Scheduler | Jobs | Machines | Resources | AWCT (synthetic units) | First scheduling call seconds |
|---|---|---|---|---|---|
| MRIS-WSJF | 128 | 2 | 4 | 106.9765625 | 0.065810 |
| Online PQ-WSJF | 128 | 2 | 4 | 91.4375 | 0.016698 |

Whole process: **3.09 s real, 2.13 s user, 0.45 s system** (about 83.5% of one core averaged over the process). Peak RSS: **222625792 bytes (212.3 MiB)**. Internal runner timing is separately recorded in summary.json. Both algorithms execute twice, serially; no multiprocessing pool or Gurobi solve is used.

The online baseline's lower AWCT on this small synthetic input is a measured result, not a contradiction or validation of the paper's trace-driven comparisons. No full figures have been reproduced.

Raw `MRIS.parquet` and `OnlinePriorityQueue.parquet` were produced locally in `results/initial-128/`, with Parquet round-trip checks. Portable review evidence under `evidence/initial-simulation/` contains all inputs, complete schedules, metrics, stdout/stderr and file hashes. The source commit in the manifest is the pre-edit base; the runner SHA-256 identifies the exact tested script subsequently committed in the follow-up PR.

## Dataset availability

**PASS**: downloaded 181,116,928 bytes from the specified Zenodo v1 link, matched published MD5 `fd7a6e124920eaeab25178f7363a5df3`, and passed read-only SQLite quick_check and required table/column checks. See `evidence/dataset-verification.json`. Database is ignored by Git. Full trace preprocessing, all five baseline sweeps and published-scale runtime/RAM: **NOT TESTED**.

## Historical first setup run

The section below records the original 16-job run and its measurements. Its old script command/output paths and hardware-access/CI status describe that execution only; use the current EXPERIMENTS.md commands for new runs.


Executed 2026-10-29 on native ARM64 macOS 26.3, Python 3.12.14, upstream `a35216c9f2511d905b71e31dffb445d6d58220e9`. No scheduler/experiment code was modified.

## Commands

Initial installation: `.venv/bin/python -m pip install 'numpy<2' matplotlib gurobipy pandas tqdm intervaltree pyarrow scienceplots lz4 'cvxpy[GLPK]' scipy`.

Initial import run failed: `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`. Evidence: `evidence/initial-import-failure.txt`. Dependency metadata checks had passed, so actual import testing was necessary.

Correction: `.venv/bin/python -m pip install 'pyarrow==17.0.0'`. This is pinned in the team specification. Original requirements remain unchanged.

Final validation command from repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -p .venv/bin/python scripts/smoke_test.py > evidence/smoke-console.txt 2> evidence/smoke-timing.txt
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q scheduler experiments scripts
```

| Test | Expected | Actual | Status |
|---|---|---|---|
| A: runtime dependencies and solver | Imports succeed; GLPK_MI solves binary toy problem to objective 2 | Imports and objective assertion passed; Parquet read/write round trip also passed | PASS |
| B: MRIS functionality | Schedule all 16 jobs on 2 machines / 4 resources without violating release/capacity constraints | All assertions passed; AWCT 24.3125 | PASS |
| C: paper baseline | OnlinePriorityQueue-WSJF schedules identical jobs feasibly | All assertions passed; AWCT 19.75 | PASS |
| D: repeatability | Fresh seed-2026 input twice per scheduler yields identical starts, assignments, completions, AWCT | Exact equality assertions passed | PASS |
| E: resource measurement | Capture wall/CPU/peak memory for small run | Whole process 6.44 s real, 2.11 s user, 0.50 s system; about 40.5% average one-core CPU; peak RSS 206569472 bytes (197.0 MiB) | PASS, tiny workload only |
| Azure traces / full figures | Full data and published sweeps | Not executed | NOT TESTED |
| Intel / Docker / hosted CI | Verify execution on other targets | Not executed | NOT TESTED |

CPU percentage is `(user + system) / real * 100`, averaged across imports and validation, not scheduler-only utilization. Peak RSS comes from macOS `getrusage`, not an 8 GB-machine guarantee. `/usr/bin/time -l` was restricted by the sandbox's kern.clockrate access; portable `time -p` plus Python resource measurement succeeded. Physical installed RAM could not be queried in the sandbox.

Evidence: `evidence/smoke-results.json` contains complete schedules and per-scheduler timings; `evidence/smoke-console.txt` and `evidence/smoke-timing.txt` capture the final run; `evidence/pip-check.txt`, `evidence/install.txt`, `evidence/pyarrow-install.txt` record installation; `requirements-mac.lock.txt` records exact package versions. Parquet round-trip files are temporary and removed after successful validation. Repeat runs overwrite smoke evidence.

These values validate only the tiny synthetic input. They do not establish superiority of either scheduler or reproduce a paper figure. Notebook UI and non-default MRIS variants have not been validated.

## Fresh environment verification

A new temporary Python 3.12 environment installed `requirements-mac.lock.txt` successfully. `pip freeze` exactly matched all 33 pinned entries; `pip check` and the 16-job GLPK/MRIS/PQ/Parquet checks passed. Whole-process timing for this run was 48.68 s real, 2.62 s user, 0.73 s system; this includes first-use imports in the fresh environment and is not comparable to scheduler-only timing. Exact logs, freeze and summary are in `evidence/clean-install/`.

Executed from the repo root: `.venv/bin/python -m venv /tmp/mris-lock-check`, `/tmp/mris-lock-check/bin/python -m pip install --disable-pip-version-check -r requirements-mac.lock.txt`, then `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -p /tmp/mris-lock-check/bin/python scripts/smoke_test.py --output-dir results/clean-install-16`. The temporary Python path is test infrastructure; students use their `.venv` as shown above.
