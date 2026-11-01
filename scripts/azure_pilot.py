"""Bounded, serial Azure-trace runner for the unchanged upstream MRIS schedulers.

Mirrors experiments/tools.py::load_azure_packing_2020 and the 3_/4_/5_ benchmark
scripts, with three documented differences that keep an 8 GB Mac safe:

* Only the selected (downsampled) jobs become Job objects; the full trace is held
  as NumPy arrays instead of ~4.66M Python objects.
* Machine-type sampling and resource augmentation use an explicit seed (upstream
  uses the unseeded global NumPy generator once and caches the result).
* Schedulers run one after another in this process instead of a Pool.

Outputs use the upstream Parquet schema and file names, so
experiments/generate_all_plots.py can read them. This is NOT a paper reproduction
unless the published parameters (10 runs per point, published sizes) are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
from scheduler.job import Job
from scheduler.mris import MRIS
from scheduler.tetris import TetrisScheduler
from scheduler.pq import PriorityQueue, OnlinePriorityQueue
from scheduler.bf_exec import BFEXECScheduler
from tools import DataGenerator

N_TOTAL = 4_096_000  # Paper section 7.1: first 4.096M jobs
DEMAND_LEVELS = DataGenerator.num_demand_levels
DB = ROOT / 'experiments' / 'data' / 'packing_trace_zone_a_v1.sqlite'
SCHEDULERS = {
    'MRIS-WSJF': lambda: MRIS(sort='WSJF'),
    'PQ-WSJF': lambda: PriorityQueue(sort='WSJF', collect=True),  # paper CA-PQ-WSJF
    'OnlinePQ-WSJF': lambda: OnlinePriorityQueue(sort='WSJF'),  # paper PQ-WSJF
    'Tetris': lambda: TetrisScheduler(),
    'BF-EXEC': lambda: BFEXECScheduler(),
}
FILE_PATTERNS = {
    'jobs': '3_scheduler_benchmark_jobs_{factor}_{run}.parquet',
    'machines': '4_scheduler_benchmark_machines_{M}_{run}.parquet',
    'resources': '5_scheduler_benchmark_resources_{R}_{run}.parquet',
}


def current_rss():
    out = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) * 1024


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


class Watchdog(threading.Thread):
    """Abort the whole process if resident memory exceeds the limit."""

    def __init__(self, limit, manifest, state):
        super().__init__(daemon=True)
        self.limit, self.manifest, self.state, self.max_seen = limit, manifest, state, 0

    def run(self):
        while True:
            rss = current_rss()
            self.max_seen = max(self.max_seen, rss)
            if rss > self.limit:
                self.state.update(status='FAIL', error=f'RSS {rss} bytes exceeded limit {self.limit}')
                self.manifest.write_text(json.dumps(self.state, indent=2) + '\n')
                print(self.state['error'], file=sys.stderr, flush=True)
                os._exit(3)
            time.sleep(0.5)


def load_trace(seed):
    """Upstream preprocessing as arrays: filter, sort, sample a machine type per VM type row."""
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    df_vm = pd.read_sql_query('SELECT vmTypeId, priority, starttime, endtime FROM vm ORDER BY vmId', con)
    df_vmtype = pd.read_sql_query('SELECT * FROM vmType', con)
    con.close()
    raw_rows = len(df_vm)
    # Upstream: dropna() (only endtime has NULLs), starttime >= 0, priority + 1, p = end - start
    df_vm = df_vm.dropna()
    df_vm = df_vm[df_vm['starttime'] >= 0].copy()
    df_vm['priority'] += 1
    df_vm['processingtime'] = df_vm['endtime'] - df_vm['starttime']
    df_vm.sort_values('starttime', ascending=True, inplace=True)  # same call/kind as upstream
    df_vm.reset_index(drop=True, inplace=True)

    df_vmtype['storage'] = df_vmtype['hdd'].where(df_vmtype['ssd'] < 1E-5, df_vmtype['ssd'])
    df_vmtype = df_vmtype.drop(columns=['machineId', 'hdd', 'ssd', 'id'])  # -> vmTypeId, core, memory, nic, storage
    df_vmtype = df_vmtype.sort_values('vmTypeId', kind='stable')
    type_ids = df_vmtype['vmTypeId'].to_numpy()
    table = df_vmtype.to_numpy()[:, 1:].astype(float)
    uniq, first, counts = np.unique(type_ids, return_index=True, return_counts=True)

    vm_types = df_vm['vmTypeId'].to_numpy()
    pos = np.searchsorted(uniq, vm_types)
    if not np.array_equal(uniq[pos], vm_types):
        raise ValueError('VM with unknown vmTypeId')
    rng = np.random.default_rng(seed)
    # Upstream: np.random.choice(rows_for_type) per job -> uniform machine type per job
    row = first[pos] + np.floor(rng.random(len(vm_types)) * counts[pos]).astype(np.int64)
    trace = {
        'r': df_vm['starttime'].to_numpy(dtype=float),
        'p': df_vm['processingtime'].to_numpy(dtype=float),
        'w': df_vm['priority'].to_numpy(dtype=float),
        'D': table[row] * DEMAND_LEVELS,
    }
    stats = {'raw_vm_rows': raw_rows, 'filtered_jobs': len(df_vm), 'vm_types': int(len(uniq)),
             'vmType_rows': int(len(type_ids)), 'release_max_days': float(trace['r'][N_TOTAL - 1])}
    return trace, stats


def build_dataset(trace, factor, offset, M, R, seed):
    """Equivalent of DataGenerator(M, R, 'azure_packing_2020') + upstream downsampling slice."""
    idx = np.arange(offset, len(trace['r']), factor)[:N_TOTAL // factor]
    D = trace['D'][idx]
    if R > 4:
        # Upstream: new resource columns are uniform draws from all jobs' CPU demand,
        # fixed per job and nested across R. Seed per trace index so the value does not
        # depend on which offset selected the job.
        cpu = trace['D'][:, 0]
        extra = np.stack([cpu[np.random.default_rng([seed, 1, int(j)]).integers(0, len(cpu), size=R - 4)]
                          for j in idx])
        D = np.concatenate([D, extra], axis=1)
    dataset = DataGenerator(M=M, R=R, integer=False, dataset='bounded_azure_packing_2020')
    dataset.jobs = [Job(p=float(trace['p'][j]), r=float(trace['r'][j]), w=float(trace['w'][j]), d=D[k].copy(), id=k)
                    for k, j in enumerate(idx)]
    return dataset, idx


def validate(frame, dataset, M):
    """Check assignment, release and capacity feasibility of one upstream results frame."""
    n = len(dataset.jobs)
    ids = np.sort(frame['id'].to_numpy())
    r = np.array([j.r for j in dataset.jobs])
    S, C, i = frame['S'].to_numpy(float), frame['C'].to_numpy(float), frame['i'].to_numpy()
    jid = frame['id'].to_numpy()
    checks = {
        'all_jobs_once': bool(np.array_equal(ids, np.arange(n))),
        'valid_machine': bool(np.all((i >= 0) & (i < M))),
        'no_start_before_release': bool(np.all(S >= r[jid] - 1e-9 * np.maximum(1, r[jid]))),
    }
    cap = dataset.machines[0].D
    d = np.stack(frame['d'].to_numpy())

    def worst_excess(S, C):
        worst = -np.inf
        for m in range(M):
            sel = i == m
            if not sel.any():
                continue
            t = np.concatenate([S[sel], C[sel]])
            kind = np.concatenate([np.ones(sel.sum()), np.zeros(sel.sum())])  # ends (0) before starts (1)
            delta = np.concatenate([d[sel], -d[sel]])
            order = np.lexsort((kind, np.round(t, 9)))
            load = np.cumsum(delta[order], axis=0)
            worst = max(worst, float(np.max(load - cap)))
        return worst

    strict = worst_excess(S, C)
    checks['capacity_respected'] = bool(strict <= 1e-6)
    checks['max_capacity_excess'] = strict
    if not checks['capacity_respected']:
        # scheduler/pq.py releases jobs whose completion is np.isclose (rtol 1e-5, atol 1e-8) to the
        # current one. Re-check with offline completions moved earlier by that tolerance; collect-all
        # schedules are shifted online by r_max afterwards, so apply it to (C - r_max).
        shift = r.max() if frame['scheduler'].iloc[0] == 'PQ-WSJF' else 0.0
        tolerant = worst_excess(S, C - (1e-8 + 1e-5 * np.abs(C - shift)))
        checks['capacity_respected_within_pq_isclose_tolerance'] = bool(tolerant <= 1e-6)
        checks['max_capacity_excess_within_tolerance'] = tolerant
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=FILE_PATTERNS, required=True)
    parser.add_argument('--factors', type=int, nargs='+', default=[512], help='Downsample factors (jobs = 4096000/f)')
    parser.add_argument('--machines', type=int, nargs='+', default=[20])
    parser.add_argument('--resources', type=int, nargs='+', default=[4])
    parser.add_argument('--runs', type=int, default=1, help='Sampled offsets per point (paper: 10)')
    parser.add_argument('--seed', type=int, default=2026, help='Seeds machine-type sampling, augmentation, offsets')
    parser.add_argument('--schedulers', nargs='+', choices=SCHEDULERS, default=list(SCHEDULERS))
    parser.add_argument('--max-jobs', type=int, default=16000, help='Refuse larger points (guardrail)')
    parser.add_argument('--max-rss-mb', type=int, default=2500, help='Abort if resident memory exceeds this')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Build and record inputs only; run no scheduler')
    args = parser.parse_args()

    points = [(f, M, R) for f in args.factors for M in args.machines for R in args.resources]
    if any(N_TOTAL // f > args.max_jobs for f, _, _ in points):
        parser.error(f'A point exceeds --max-jobs {args.max_jobs}; raise it deliberately')
    if not all(4 <= R <= 20 for _, _, R in points) or not all(1 <= M <= 40 for _, M, _ in points):
        parser.error('Use 4-20 resources and 1-40 machines')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / 'summary.json'
    state = {
        'status': 'RUNNING', 'scope': 'Bounded serial run on the Azure trace; not a full paper reproduction',
        'argv': sys.argv, 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'machine': platform.machine(),
        'threads': {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
        'results': [],
    }
    manifest.write_text(json.dumps(state, indent=2) + '\n')
    watchdog = Watchdog(args.max_rss_mb * 2 ** 20, manifest, state)
    watchdog.start()

    t0 = time.perf_counter()
    trace, stats = load_trace(args.seed)
    state['trace'] = {**stats, 'load_seconds': time.perf_counter() - t0, 'rss_after_load': current_rss()}
    print(json.dumps(state['trace']), flush=True)
    offset_rng = random.Random(args.seed)
    try:
        for factor, M, R in points:
            # Upstream run.sh: offsets = random.sample(range(0, f), NUM_RUNS) per point
            offsets = offset_rng.sample(range(0, factor), args.runs)
            for run, offset in enumerate(offsets, start=1):
                dataset, idx = build_dataset(trace, factor, offset, M, R, args.seed)
                frames, entry = [], {'factor': factor, 'jobs': len(idx), 'M': M, 'R': R, 'run': run,
                                     'offset': offset, 'release_span_days': [float(trace['r'][idx[0]]),
                                                                             float(trace['r'][idx[-1]])],
                                     'schedulers': {}}
                if args.dry_run:
                    entry.update(demand_shape=list(np.stack([j.d for j in dataset.jobs]).shape),
                                 rss_after_build=current_rss(), schedulers=None)
                    state['results'].append(entry)
                    print(json.dumps({k: v for k, v in entry.items()}), flush=True)
                    continue
                for k, name in enumerate(args.schedulers):
                    scheduler = SCHEDULERS[name]()
                    scheduler.id = k + 1
                    scheduler.jobs, scheduler.machines = dataset.get_copy()
                    w0, c0 = time.perf_counter(), time.process_time()
                    scheduler.process()
                    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
                    frame = scheduler.results_as_dataframe()
                    checks = validate(frame, dataset, M)
                    frames.append(frame)
                    entry['schedulers'][repr(scheduler)] = {
                        'awct': float(frame['C'].dot(frame['w']) / len(frame)),
                        'wall_seconds': wall, 'cpu_seconds': cpu,
                        'rss_after': current_rss(), 'peak_rss_so_far': peak_rss(), 'checks': checks}
                    print(json.dumps({'point': [factor, M, R, run], name: entry['schedulers'][repr(scheduler)]}),
                          flush=True)
                    del scheduler
                out = args.output_dir / FILE_PATTERNS[args.experiment].format(factor=factor, M=M, R=R, run=run)
                pd.concat(frames).to_parquet(out)
                entry['parquet'] = out.name
                entry['parquet_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
                state['results'].append(entry)
                manifest.write_text(json.dumps(state, indent=2) + '\n')
        checks = [c['checks'] for e in state['results'] for c in (e['schedulers'] or {}).values()]
        basic = all(c[k] for c in checks for k in ['all_jobs_once', 'valid_machine', 'no_start_before_release'])
        tolerant = all(c['capacity_respected'] or c.get('capacity_respected_within_pq_isclose_tolerance')
                       for c in checks)
        strict = all(c['capacity_respected'] for c in checks)
        # PASS_WITH_WARNINGS: only the documented upstream np.isclose overlap in scheduler/pq.py
        state['status'] = 'PASS' if basic and strict else 'PASS_WITH_WARNINGS' if basic and tolerant else 'FAIL'
    except Exception as exc:
        state.update(status='FAIL', error=repr(exc))
        raise
    finally:
        state['total_wall_seconds'] = time.perf_counter() - t0
        state['peak_rss_bytes'] = peak_rss()
        state['watchdog_max_rss_bytes'] = watchdog.max_seen
        manifest.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({'status': state['status'], 'total_wall_seconds': state['total_wall_seconds'],
                      'peak_rss_bytes': state['peak_rss_bytes']}))
    return 0 if state['status'] in ('PASS', 'PASS_WITH_WARNINGS') else 1


if __name__ == '__main__':
    sys.exit(main())
