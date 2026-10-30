# Mac setup

Validated environment: native ARM64 macOS 26.3 with Python 3.12.14. Intel execution is not tested. Use the same commands on Intel if compatible wheels are available. Docker, Rosetta, paid solvers, and system-wide pip installs are unnecessary for the small smoke test.

## Prerequisites and checkout

Check `git --version` and `python3.12 --version`. If Git is missing, install Apple's command line tools with `xcode-select --install`. If Python 3.12 is missing and Homebrew is already installed, use `brew install python@3.12`. This Homebrew route has not been executed here; the local environment uses an existing Python 3.12 runtime. Do not use the system Python 3.15 for the pinned NumPy 1.x stack.

Team fork: `Vishwabelaganti/mris-os-research-reproduction`. The setup is initially on `setup/mac-support`; check out that branch until its pull request is merged:

```sh
git clone https://github.com/Vishwabelaganti/mris-os-research-reproduction.git
cd mris-os-research-reproduction
git switch setup/mac-support
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.lock.txt
python -m pip check
python -c "import platform, cvxpy; print(platform.machine()); print(cvxpy.installed_solvers())"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py
```

For the local checkout created during this session:

```sh
cd '/Users/vishwakaran/Downloads/Classes/Fall 2026/Operating Systems/OSproject/mris-os-research-reproduction'
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py
```

Success means the script exits 0 and writes `status: PASS` into `evidence/smoke-results.json`. GLPK_MI must appear among installed solvers. The lock is an exact version snapshot, not a hash-locked or cross-platform guarantee. Notebook work is optional: install `notebook` separately and record that expanded environment.

## Dataset stage (not yet executed)

No dataset is necessary for the smoke test. To obtain the original trace for a later pilot:

```sh
mkdir -p experiments/data experiments/results
curl -fL --retry 3 'https://zenodo.org/records/12028115/files/packing_trace_zone_a_v1.sqlite?download=1' -o experiments/data/packing_trace_zone_a_v1.sqlite
md5 -q experiments/data/packing_trace_zone_a_v1.sqlite
```

Expected MD5 from the specified Zenodo v1 record: `fd7a6e124920eaeab25178f7363a5df3`. Stop if it differs. Preprocessing requires full-data RAM; do not launch the default downloader on an 8 GB Mac before measuring a serial pilot. A later bounded-data loader must be documented as reduced scale, and must preserve upstream sampling/normalization semantics. See EXPERIMENTS.md.

## Troubleshooting

See TROUBLESHOOTING.md. Never fix permissions by running pip with sudo.
