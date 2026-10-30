# Initial test results

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
