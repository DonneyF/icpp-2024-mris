"""Small serial validation of unchanged upstream schedulers; no Azure data needed."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import importlib
import json
import platform
import resource
import tempfile
import pandas as pd
import time
import numpy as np
import cvxpy as cp
from scheduler.job import Job
from scheduler.machine import Machine
from scheduler.mris import MRIS
from scheduler.pq import OnlinePriorityQueue


def run(cls):
    rng = np.random.default_rng(2026)
    jobs = [Job(id=i, p=float(rng.integers(1, 5)), r=float(i // 4),
                w=float(rng.integers(1, 5)), d=rng.uniform(0.1, 0.8, 4)) for i in range(16)]
    machines = [Machine(id=i, D=np.ones(4)) for i in range(2)]
    scheduler = cls(jobs=jobs, machines=machines, sort='WSJF', id=1)
    start = time.perf_counter()
    scheduler.process()
    elapsed = time.perf_counter() - start
    assert all(j.S is not None and j.i in (0, 1) and j.S >= j.r - 1e-8 for j in jobs)
    assert sorted(j.id for m in machines for j in m.jobs) == list(range(16))
    for m in machines:
        for t in sorted({v for j in m.jobs for v in (j.S, j.S + j.p)}):
            demand = sum((j.d for j in m.jobs if j.S <= t < j.S + j.p), np.zeros(4))
            assert np.all(demand <= m.D + 1e-8)
    return {'awct': sum(j.w * (j.S + j.p) for j in jobs) / len(jobs),
            'schedule': [[j.id, int(j.i), float(j.S), float(j.S + j.p)] for j in jobs],
            'seconds': elapsed}


def main():
    start = time.perf_counter()
    cpu_start = time.process_time()
    for name in ['numpy', 'matplotlib', 'gurobipy', 'pandas', 'tqdm', 'intervaltree',
                 'pyarrow', 'scienceplots', 'lz4.frame', 'cvxpy', 'scipy',
                 'experiments.tools', 'scheduler.tetris', 'scheduler.bf_exec']:
        importlib.import_module(name)
    assert 'GLPK_MI' in cp.installed_solvers(), cp.installed_solvers()
    x = cp.Variable(2, boolean=True)
    problem = cp.Problem(cp.Maximize(np.array([1., 2.]) @ x), [cp.sum(x) <= 1])
    problem.solve(solver='GLPK_MI')
    assert problem.status == 'optimal' and abs(problem.value - 2) < 1e-8
    with tempfile.TemporaryDirectory() as tmp:
        frame = pd.DataFrame({'C': [1., 2.], 'w': [2., 1.]})
        path = Path(tmp) / 'roundtrip.parquet'
        frame.to_parquet(path)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
    results = {}
    for cls in [MRIS, OnlinePriorityQueue]:
        first, second = run(cls), run(cls)
        assert first['schedule'] == second['schedule'] and first['awct'] == second['awct']
        results[cls.__name__] = first
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    output = {'status': 'PASS', 'scope': '16 synthetic jobs, 2 machines, 4 resources; not paper reproduction',
              'seed': 2026, 'python': sys.version, 'platform': platform.platform(),
              'solvers': cp.installed_solvers(), 'results': results,
              'wall_seconds_after_imports': elapsed,
              'cpu_seconds_after_imports': time.process_time() - cpu_start,
              'peak_rss_bytes': usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024}
    target = ROOT / 'evidence' / 'smoke-results.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
