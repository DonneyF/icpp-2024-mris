"""Plot reduced-scale Figure 3/4/6 runs with the unchanged upstream plotting functions.

Reads the directories written by scripts/run_reduced_figures.sh and writes PNGs plus an
AWCT table (mean, min and max over the sampled offsets) next to them. Every PNG is stamped
"REDUCED-SCALE" because these are not paper reproductions.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments import generate_all_plots as plots

PAPER_LABELS = {'PQ-WSJF': 'CA-PQ-WSJF', 'OnlinePQ-WSJF': 'PQ-WSJF', 'Tetris-instantaneous': 'Tetris'}
FIGURES = {
    'fig3-jobs': ('Figure 3', 'factor', plots.scheduler_benchmark_jobs, '8,000-256,000 jobs'),
    'fig4-machines': ('Figure 4', 'M', plots.scheduler_benchmark_machines, '64,000 jobs'),
    'fig6-resources': ('Figure 6', 'R', plots.scheduler_benchmark_resources, '64,000 jobs'),
    'fig6-resources-loaded': ('Figure 6 M=2 load variant (not a paper configuration)', 'R',
                              plots.scheduler_benchmark_resources, '64,000 jobs on 20 machines'),
}


ORIGINAL_SAVEFIG = plt.savefig


def stamped_savefig(stamp):
    def savefig(name, **kwargs):
        fig = plt.gcf()
        for ax in fig.axes:
            legend = ax.get_legend()
            for text in legend.get_texts() if legend else []:
                text.set_text(text.get_text().replace(r'\textsc{Tetris}', 'Tetris'))  # no LaTeX installed
        fig.suptitle(stamp, fontsize=7, y=1.02)
        ORIGINAL_SAVEFIG(name, **kwargs)
    return savefig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Output root given to run_reduced_figures.sh')
    args = parser.parse_args()
    rows = []
    for folder, (figure, key, plot, paper) in FIGURES.items():
        summary = args.root / folder / 'summary.json'
        if not summary.exists():
            print(f'skip {folder}: no summary.json')
            continue
        state = json.loads(summary.read_text())
        if state['status'] not in ('PASS', 'PASS_WITH_WARNINGS'):
            print(f'skip {folder}: status {state["status"]}')
            continue
        results = state['results']
        values = sorted({e[key] for e in results})
        runs = max(e['run'] for e in results)
        jobs = sorted({e['jobs'] for e in results})
        machines = sorted({e['M'] for e in results})
        for e in results:
            for name, s in e['schedulers'].items():
                rows.append(dict(figure=figure, jobs=e['jobs'], M=e['M'], R=e['R'], run=e['run'],
                                 offset=e['offset'], scheduler=PAPER_LABELS.get(name, name), awct=s['awct'],
                                 wall_seconds=s['wall_seconds']))
        plots.PATH_TO_RESULTS = (args.root / folder).resolve()
        out = args.root / 'figures' / folder
        out.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        os.chdir(out)
        plt.savefig = stamped_savefig(
            f'REDUCED-SCALE {figure} validation: {jobs[0] if len(jobs) == 1 else "/".join(map(str, jobs))} jobs, '
            f'M={machines[0] if len(machines) == 1 else "/".join(map(str, machines))}, '
            f'{runs} offsets per point (paper: {paper}, 10 offsets) - not a paper reproduction')
        try:
            if key == 'factor':
                plot(4_096_000, np.array(values[::-1]), runs)  # upstream expects decreasing factors
            else:
                plot(np.array(values), runs)
        finally:
            plt.savefig = ORIGINAL_SAVEFIG
            os.chdir(cwd)
            plt.close('all')
    table = pd.DataFrame(rows)
    table.to_csv(args.root / 'figures' / 'awct_runs.csv', index=False)
    agg = table.groupby(['figure', 'jobs', 'M', 'R', 'scheduler'])['awct'].agg(['mean', 'min', 'max', 'count'])
    agg.to_csv(args.root / 'figures' / 'awct_summary.csv')
    print(agg.round(3).to_string())


if __name__ == '__main__':
    main()
