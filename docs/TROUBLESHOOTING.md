# Troubleshooting

- Wrong Python / NumPy build failure: use Python 3.12 in a fresh local venv; NumPy is constrained below 2 by upstream. Do not upgrade the research stack blindly.
- `No module named tqdm`: use team requirements; upstream lists `tdqm`.
- `No module named gurobipy` despite GLPK: upstream imports Gurobi at module load. Keep the package installed, while leaving MRIS solver at GLPK_MI.
- GLPK_MI missing: inspect `python -c "import cvxpy; print(cvxpy.installed_solvers())"`. Install `cvxpy[GLPK]`/CVXOPT in the same interpreter. Native wheel installation worked here. A source-build fallback may require GLPK and build libraries; it has not been validated here.
- Import error for scheduler: run team script from checkout, or use `PYTHONPATH="$PWD"` for upstream experiments.
- Missing results directory: `mkdir -p experiments/results` before upstream runs.
- Empty experiment 1 results: its run.sh invocation is commented out; invoke the Python script deliberately only after resource review.
- Memory pressure: stop the full parallel run. Downsampling happens after loading all jobs. Use `scripts/azure_pilot.py` (serial, holds the trace as arrays, `--max-rss-mb` watchdog, measured peak about 1.2–1.3 GB).
- `summary.json` says `PASS_WITH_WARNINGS`: only the known collect-all PQ (`PQ-WSJF`, paper CA-PQ) `np.isclose` capacity overlap was found; every other check passed. `FAIL` with `RSS ... exceeded limit` means the watchdog stopped the run; close other apps or reduce the job count rather than raising the limit blindly.
- `A point exceeds --max-jobs`: deliberate guardrail. Raise `--max-jobs` only for a single-point run you intend to leave running for hours.
- Plot missing parquet files: select only the plot function with complete inputs; all-plots assumes every experiment exists.
- Repeatability mismatch: regenerate fresh job/machine objects, retain preprocessing seed and selected offsets, and compare the same input checksum.
- A pip cache permission warning in the sandbox does not mean installation failed. Do not use sudo. The environment was installed successfully and pip check passed.

- PyArrow asks for NumPy 2: use the tested `pyarrow==17.0.0` pin; do not violate upstream NumPy<2 to satisfy the newer wheel.

- `FileExistsError` for a run directory: choose a new `--output-dir`. Existing runs are deliberately preserved.
- No `summary.json` or status other than PASS: inspect stderr and the exit code. Never reuse an older PASS as evidence for a failed retry.
- Git push permission denied: accept the owner invitation, verify write access and the signed-in Git account; do not change branch protection to work around missing access.
- Notebook/LaTeX UI is optional and untested here. The upstream plotting script has a no-LaTeX fallback; do not install a full TeX distribution just for the smoke test.
