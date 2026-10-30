# MacBook Air setup

Tested hardware: Apple M2 MacBook Air (`Mac14,2`), 8 GB physical RAM, 8 CPU cores, ARM64, macOS 26.3. Tests use Python 3.12.14 in a virtual environment. Intel, Docker, full Azure preprocessing and published-scale experiments have not been tested. Native smoke tests need no Docker, Rosetta, GPU, paid solver or system-wide pip installation.

## 1. Prerequisites

```sh
git --version
python3.12 --version
```

If Git is missing, install Apple command line tools with `xcode-select --install`. If Python 3.12 is missing and Homebrew is already installed:

```sh
brew install python@3.12
```

Otherwise install Python 3.12 using the macOS installer from python.org. The machine used for these tests already had a bundled Python 3.12 runtime; prerequisite installation through Homebrew/python.org has not itself been executed here. Do not use this machine's default Python 3.15 with the NumPy 1.x stack. `python3.12` should report 3.12.x; exact tested patch is 3.12.14.

## 2. Clone and install

```sh
git clone https://github.com/Vishwabelaganti/mris-os-research-reproduction.git
cd mris-os-research-reproduction
git switch main
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.lock.txt
python -m pip check
python -c "import platform, cvxpy; print(platform.machine()); print(cvxpy.installed_solvers())"
```

For an existing clone, use `git status` and save your work before switching to main and running `git pull --ff-only`. Do not clone again into the existing directory. The same instructions are candidates for Intel Macs but are not verified there. The version lock is not a cross-platform guarantee or hash-locked dependency specification.

## 3. Run a small test

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py --output-dir results/my-first-test
```

Success means exit code 0, `status: PASS` in `results/my-first-test/summary.json`, an input manifest, and two Parquet files. Use a new directory name for another run. The default is 16 synthetic jobs on 2 machines with 4 resources. See EXPERIMENTS.md for the measured 128-job command and logs.

GLPK_MI must be listed as an installed solver. CVXOPT supplies its support for CVXPY. Gurobi is installed because upstream imports it; our test uses GLPK and requires no Gurobi license activation. `pyarrow==17.0.0` is pinned because the tested newer wheel failed with `numpy<2`. Notebook UI is optional and not installed in this lock; the upstream plot script can run without it.

## 4. Obtain the data for later experiments

The following download was executed successfully and the checksum verified. Download size is 181,116,928 bytes. Skip the download if a verified file already exists.

```sh
mkdir -p experiments/data experiments/results
curl -fL --retry 3 'https://zenodo.org/records/12028115/files/packing_trace_zone_a_v1.sqlite?download=1' -o experiments/data/packing_trace_zone_a_v1.sqlite
md5 -q experiments/data/packing_trace_zone_a_v1.sqlite
```

Expected MD5: `fd7a6e124920eaeab25178f7363a5df3`. Stop on a mismatch. SHA-256 and read-only SQLite verification are in `evidence/dataset-verification.json`. Download verification does not establish that full preprocessing fits in 8 GB. Do not run `download_data.sh` or the parallel experiment sweep as the first laptop test: upstream constructs millions of jobs before downsampling. The next task is a bounded serial Azure pilot.

The tested environment was approximately 521 MB before generated outputs; the database is approximately 181 MB. Full resource caches and results need additional, unmeasured storage. No measured full-experiment minimum RAM or runtime is available. See TROUBLESHOOTING.md for common failures.

## Fresh environment verification

A new temporary Python 3.12 environment installed `requirements-mac.lock.txt` successfully. `pip freeze` exactly matched all 33 pinned entries; `pip check` and the 16-job GLPK/MRIS/PQ/Parquet checks passed. Whole-process timing for this run was 48.68 s real, 2.62 s user, 0.73 s system; this includes first-use imports in the fresh environment and is not comparable to scheduler-only timing. Exact logs, freeze and summary are in `evidence/clean-install/`.

Executed from the repo root: `.venv/bin/python -m venv /tmp/mris-lock-check`, `/tmp/mris-lock-check/bin/python -m pip install --disable-pip-version-check -r requirements-mac.lock.txt`, then `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -p /tmp/mris-lock-check/bin/python scripts/smoke_test.py --output-dir results/clean-install-16`. The temporary Python path is test infrastructure; students use their `.venv` as shown above.
