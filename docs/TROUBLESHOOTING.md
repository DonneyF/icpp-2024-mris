# Troubleshooting

- Wrong Python / NumPy build failure: use Python 3.12 in a fresh local venv; NumPy is constrained below 2 by upstream. Do not upgrade the research stack blindly.
- `No module named tqdm`: use team requirements; upstream lists `tdqm`.
- `No module named gurobipy` despite GLPK: upstream imports Gurobi at module load. Keep the package installed, while leaving MRIS solver at GLPK_MI.
- GLPK_MI missing: inspect `python -c "import cvxpy; print(cvxpy.installed_solvers())"`. Install `cvxpy[GLPK]`/CVXOPT in the same interpreter. Native wheel installation worked here. A source-build fallback may require GLPK and build libraries; it has not been validated here.
- Import error for scheduler: run team script from checkout, or use `PYTHONPATH="$PWD"` for upstream experiments.
- Missing results directory: `mkdir -p experiments/results` before upstream runs.
- Empty experiment 1 results: its run.sh invocation is commented out; invoke the Python script deliberately only after resource review.
- Memory pressure: stop the full parallel run. Downsampling happens after loading all jobs. A serial bounded pilot is the next implementation task.
- Plot missing parquet files: select only the plot function with complete inputs; all-plots assumes every experiment exists.
- Repeatability mismatch: regenerate fresh job/machine objects, retain preprocessing seed and selected offsets, and compare the same input checksum.
- A pip cache permission warning in the sandbox does not mean installation failed. Do not use sudo. The environment was installed successfully and pip check passed.

- PyArrow asks for NumPy 2: use the tested `pyarrow==17.0.0` pin; do not violate upstream NumPy<2 to satisfy the newer wheel.
