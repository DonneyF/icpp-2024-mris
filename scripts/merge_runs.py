"""Merge chunked scripts/azure_pilot.py output folders into one folder for plotting.

Each chunk is one azure_pilot.py --output-dir (for example one machine count per
chunk). The merged folder holds every Parquet file plus a combined summary.json,
so scripts/plot_reduced_figures.py can plot the whole sweep.

Usage: python scripts/merge_runs.py --output results/<root>/fig4-machines results/<root>/chunks/m2 ...
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

POINT_KEY = {'jobs': 'factor', 'machines': 'M', 'resources': 'R'}
FIXED_KEYS = {'jobs': ('M', 'R'), 'machines': ('factor', 'R'), 'resources': ('factor', 'M')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('chunks', type=Path, nargs='+', help='azure_pilot.py output folders')
    parser.add_argument('--output', type=Path, required=True, help='New folder, e.g. results/<root>/fig4-machines')
    args = parser.parse_args()

    states = [json.loads((c / 'summary.json').read_text()) for c in args.chunks]
    experiments = {s['config']['experiment'] for s in states}
    seeds = {s['config']['seed'] for s in states}
    if len(experiments) != 1 or len(seeds) != 1:
        sys.exit(f'Chunks must share one experiment and seed, found {experiments} and {seeds}')
    if any(s['status'] not in ('PASS', 'PASS_WITH_WARNINGS') for s in states):
        sys.exit('A chunk did not finish with PASS or PASS_WITH_WARNINGS; rerun it before merging')
    if any(s['results'] and s['results'][0]['schedulers'] is None for s in states):
        sys.exit('Dry-run chunks have no schedules to merge')
    experiment = experiments.pop()
    key = POINT_KEY[experiment]

    results = [e for s in states for e in s['results']]
    for fixed in FIXED_KEYS[experiment]:
        values = {e[fixed] for e in results}
        if len(values) != 1:
            sys.exit(f'Chunks mix {fixed} = {sorted(values)}; keep different settings in separate folders')
    runs = {}
    for e in results:
        runs.setdefault(e[key], set()).add(e['run'])
    if len({frozenset(r) for r in runs.values()}) != 1 or any(r != set(range(1, len(r) + 1)) for r in runs.values()):
        sys.exit(f'Every point needs the same runs 1..n; found {runs}')

    args.output.mkdir(parents=True, exist_ok=False)
    for chunk, state in zip(args.chunks, states):
        for e in state['results']:
            src, dst = chunk / e['parquet'], args.output / e['parquet']
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if digest != e['parquet_sha256']:
                sys.exit(f'{src} does not match its recorded SHA-256')
            if dst.exists():
                sys.exit(f'Two chunks both contain {e["parquet"]}; each point must come from one chunk')
            shutil.copy2(src, dst)

    statuses = {s['status'] for s in states}
    merged = {
        'status': 'PASS_WITH_WARNINGS' if 'PASS_WITH_WARNINGS' in statuses else 'PASS',
        'scope': states[0]['scope'],
        'merged_from': [{'chunk': str(c), 'argv': s['argv'], 'runner_sha256': s['runner_sha256'],
                         'git_commit': s['git_commit'], 'status': s['status'],
                         'total_wall_seconds': s['total_wall_seconds'], 'peak_rss_bytes': s['peak_rss_bytes']}
                        for c, s in zip(args.chunks, states)],
        'config': {'experiment': experiment, 'seed': states[0]['config']['seed']},
        'total_wall_seconds': sum(s['total_wall_seconds'] for s in states),
        'peak_rss_bytes': max(s['peak_rss_bytes'] for s in states),
        'results': sorted(results, key=lambda e: (e[key], e['run'])),
    }
    (args.output / 'summary.json').write_text(json.dumps(merged, indent=2) + '\n')
    print(json.dumps({'status': merged['status'], 'points': sorted(runs), 'runs_per_point': len(next(iter(runs.values()))),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
