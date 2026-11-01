"""Bounded serial validation of unchanged upstream schedulers, not paper reproduction."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

PROCESS_START = time.perf_counter()
CPU_START = time.process_time()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import cvxpy as cp
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.mris import MRIS
from scheduler.pq import OnlinePriorityQueue


def make_inputs(args):
    rng = np.random.default_rng(args.seed)
    return [dict(id=i, p=float(rng.integers(1, 5)), r=float(i // 4),
                 w=float(rng.integers(1, 5)), d=rng.uniform(0.1, 0.8, args.resources).tolist())
            for i in range(args.jobs)]


def run(cls, inputs, args):
    jobs = [Job(**{**row, 'd': np.array(row['d'])}) for row in inputs]
    machines = [Machine(id=i, D=np.ones(args.resources)) for i in range(args.machines)]
    scheduler = cls(jobs=jobs, machines=machines, sort='WSJF', id=1)
    start = time.perf_counter()
    scheduler.process()
    elapsed = time.perf_counter() - start
    if not all(j.S is not None and np.isfinite(j.S) and j.i in range(args.machines)
               and j.S >= j.r - 1e-8 for j in jobs):
        raise AssertionError('Missing/invalid assignment or job scheduled before release')
    if sorted(j.id for m in machines for j in m.jobs) != list(range(args.jobs)):
        raise AssertionError('Missing or duplicated jobs')
    for m in machines:
        for t in sorted({v for j in m.jobs for v in (j.S, j.S + j.p)}):
            demand = sum((j.d for j in m.jobs if j.S <= t < j.S + j.p), np.zeros(args.resources))
            if np.any(demand > m.D + 1e-8):
                raise AssertionError('Resource capacity exceeded')
    for job, original in zip(jobs, inputs):
        if not np.isclose(job.p, original['p']) or not np.array_equal(job.d, original['d']):
            raise AssertionError('Job processing time/demand changed')
    frame = scheduler.results_as_dataframe().reset_index(drop=True)
    summary = {'awct': float(frame['C'].dot(frame['w']) / len(frame)),
               'schedule': [[j.id, int(j.i), float(j.S), float(j.S + j.p)] for j in jobs],
               'seconds': elapsed}
    return summary, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--machines', type=int, default=2)
    parser.add_argument('--resources', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'smoke')
    args = parser.parse_args()
    if not (1 <= args.jobs <= 1000 and 1 <= args.machines <= 40 and 2 <= args.resources <= 20 and args.seed >= 0):
        parser.error('Use 1–1000 jobs, 1–40 machines, 2–20 resources and a nonnegative seed')
    # Refuse to overwrite evidence or leave a stale PASS from an earlier run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / 'summary.json'
    manifest.write_text(json.dumps({'status': 'RUNNING'}) + '\n')
    try:
        for name in ['numpy', 'matplotlib', 'gurobipy', 'pandas', 'tqdm', 'intervaltree',
                     'pyarrow', 'scienceplots', 'lz4.frame', 'cvxpy', 'scipy',
                     'experiments.tools', 'scheduler.tetris', 'scheduler.bf_exec']:
            importlib.import_module(name)
        if 'GLPK_MI' not in cp.installed_solvers():
            raise AssertionError(f'GLPK_MI missing: {cp.installed_solvers()}')
        x = cp.Variable(2, boolean=True)
        problem = cp.Problem(cp.Maximize(np.array([1., 2.]) @ x), [cp.sum(x) <= 1])
        problem.solve(solver='GLPK_MI')
        if problem.status != 'optimal' or abs(problem.value - 2) > 1e-8:
            raise AssertionError('GLPK integer solve failed')
        inputs = make_inputs(args)
        serialized = json.dumps(inputs, sort_keys=True, indent=2) + '\n'
        (args.output_dir / 'inputs.json').write_text(serialized)
        results = {}
        for cls in [MRIS, OnlinePriorityQueue]:
            first, frame = run(cls, inputs, args)
            second, repeated = run(cls, inputs, args)
            if first['schedule'] != second['schedule'] or first['awct'] != second['awct']:
                raise AssertionError(f'{cls.__name__} repeatability failed')
            pd.testing.assert_frame_equal(frame, repeated)
            target = args.output_dir / f'{cls.__name__}.parquet'
            frame.to_parquet(target, index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(target))
            # JSON schedules are reviewable; parquet preserves the upstream schema.
            results[cls.__name__] = first
        wall = time.perf_counter() - PROCESS_START
        cpu = time.process_time() - CPU_START
        usage = resource.getrusage(resource.RUSAGE_SELF)
        output = {'status': 'PASS', 'scope': 'synthetic functionality validation; not paper reproduction',
                  'seed': args.seed, 'jobs': args.jobs, 'machines': args.machines, 'resources': args.resources,
                  'python': sys.version, 'platform': platform.platform(),
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'inputs_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                  'solvers': cp.installed_solvers(), 'results': results,
                  'wall_seconds': wall, 'cpu_seconds': cpu, 'average_cpu_percent': 100 * cpu / wall,
                  'peak_rss_bytes': usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024,
                  'thread_environment': {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']}}
        manifest.write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps(output, indent=2))
    except Exception as error:
        manifest.write_text(json.dumps({'status': 'FAIL', 'error': str(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
