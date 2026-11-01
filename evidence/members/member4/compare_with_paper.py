"""Member 4: compare our load-matched Figure 6 sweep with the published Figure 6.

Inputs
  * merged summary.json from scripts/merge_runs.py (our runs: M = 5, N = 16,000, R = 4..20)
  * paper_fig6_digitized.json (published Figure 6, M = 20, N = 64,000, digitized from the PDF)
  * optional: Member 1's paper-scale summary.json (M = 20, N = 64,000, R = 4, one offset)

Outputs (in --out)
  * fig6_paper_vs_ours.png        published curves next to ours, same AWCT scale
  * fig6_relative_increase.png    AWCT change relative to R = 4, one panel per scheduler
  * fig6_comparison.csv           every plotted value (table view of both figures)

Usage, from the repository root:
  MPLBACKEND=Agg .venv/bin/python evidence/members/member4/compare_with_paper.py \
      --ours results/member4/fig6-resources/summary.json \
      --paper-64k evidence/members/member1/chunks/paper-64k/summary.json \
      --out results/member4/figures/comparison
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
# Runner names -> paper names (same mapping as scripts/plot_reduced_figures.py)
PAPER_LABELS = {'MRIS-WSJF': 'MRIS-WSJF', 'PQ-WSJF': 'CA-PQ-WSJF', 'OnlinePQ-WSJF': 'PQ-WSJF',
                'Tetris-instantaneous': 'Tetris', 'BF-EXEC': 'BF-EXEC'}
ORDER = ['MRIS-WSJF', 'CA-PQ-WSJF', 'PQ-WSJF', 'Tetris', 'BF-EXEC']
# Validated categorical palette (fixed order, light surface); markers and dashes are the secondary encoding
STYLE = {'MRIS-WSJF': ('#2a78d6', 'o', '-'), 'CA-PQ-WSJF': ('#eb6834', 's', '-'),
         'PQ-WSJF': ('#1baf7a', '^', '-'), 'Tetris': ('#eda100', 'D', '-'), 'BF-EXEC': ('#e87ba4', 'v', '-')}
SURFACE, INK, INK_2, GRID, PAPER_GREY = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8f8e89'
STAMP = ('REDUCED-SCALE, load-matched (not the paper configuration): 16,000 jobs on M = 5 machines '
         '(3,200 jobs/machine, as in the paper), 2 offsets per point. Paper: 64,000 jobs, M = 20, 10 offsets.')

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10,
    'legend.frameon': False, 'lines.linewidth': 1.8, 'lines.markersize': 5,
})


def load_ours(path):
    state = json.loads(Path(path).read_text())
    if state['status'] not in ('PASS', 'PASS_WITH_WARNINGS'):
        raise SystemExit(f'{path}: status {state["status"]}')
    rows = [dict(R=e['R'], run=e['run'], offset=e['offset'], jobs=e['jobs'], M=e['M'],
                 scheduler=PAPER_LABELS[name], awct=s['awct'])
            for e in state['results'] for name, s in e['schedulers'].items()]
    runs = pd.DataFrame(rows)
    agg = runs.groupby(['scheduler', 'R'])['awct'].agg(['mean', 'min', 'max', 'count']).reset_index()
    return runs, agg


def load_paper():
    d = json.loads((HERE / 'paper_fig6_digitized.json').read_text())
    return pd.DataFrame([dict(scheduler=s, R=r, awct=v) for s, vals in d['awct'].items()
                         for r, v in zip(d['R'], vals)])


def load_64k(path):
    if not path:
        return {}
    e = json.loads(Path(path).read_text())['results'][0]
    assert (e['jobs'], e['M'], e['R']) == (64000, 20, 4), 'expected the 64k, M=20, R=4 point'
    return {PAPER_LABELS[k]: v['awct'] for k, v in e['schedulers'].items()}


def label_ends(ax, series, xmax, min_gap):
    """Direct labels at the right end, nudged apart so they never overlap."""
    items = sorted(series, key=lambda t: t[1])
    placed = []
    for name, y in items:
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((name, y))
    for name, y in placed:
        ax.annotate(name, (xmax, y), xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=8, color=INK_2, annotation_clip=False)


def plot_side_by_side(paper, agg, k64, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for name in ORDER:
        color, marker, ls = STYLE[name]
        p = paper[paper.scheduler == name]
        left.plot(p.R, p.awct, ls, color=color, marker=marker, markersize=3.5, label=name)
        o = agg[agg.scheduler == name]
        right.fill_between(o.R, o['min'], o['max'], color=color, alpha=0.15, linewidth=0)
        right.plot(o.R, o['mean'], ls, color=color, marker=marker, label=name)
    if k64:
        for name in ORDER:
            left.plot(4, k64[name], marker=STYLE[name][1], markersize=8, markerfacecolor='none',
                      markeredgecolor=INK, markeredgewidth=1.0, linestyle='none')
        left.plot([], [], marker='o', markersize=8, markerfacecolor='none', markeredgecolor=INK,
                  linestyle='none', label='Our 64k run (Member 1, 1 offset)')
    left.set_title('Published Figure 6  (M = 20, N = 64,000; digitized)', loc='left')
    right.set_title('Ours  (M = 5, N = 16,000; mean, band = min-max of 2 offsets)', loc='left')
    top = max(paper.awct.max(), agg['max'].max())
    for ax, df, col in ((left, paper, 'awct'), (right, agg, 'mean')):
        ax.set_xlabel('Number of resource types R')
        ax.set_xticks(range(4, 21, 2))
        ax.set_xlim(3.5, 20.5)
        ends = df[df.R == 20].set_index('scheduler')[col]
        label_ends(ax, [(n, ends[n]) for n in ORDER], 20.5, top * 0.045)
    left.set_ylabel('Average weighted completion time')
    left.set_ylim(0, top * 1.08)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.text(0.01, 1.0, STAMP, fontsize=7, color=INK_2, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 0.93, 0.98))
    fig.savefig(out / 'fig6_paper_vs_ours.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def relative(df, col):
    base = df[df.R == 4].set_index('scheduler')[col]
    return df.assign(pct=100 * (df[col] / df.scheduler.map(base) - 1))


def plot_relative(paper_rel, ours_rel, out):
    fig, axes = plt.subplots(1, len(ORDER), figsize=(13, 3.3), sharey=True)
    for ax, name in zip(axes, ORDER):
        color, marker, _ = STYLE[name]
        p = paper_rel[paper_rel.scheduler == name]
        o = ours_rel[ours_rel.scheduler == name]
        ax.plot(p.R, p.pct, '--', color=PAPER_GREY, linewidth=1.4, label='Paper (M = 20, N = 64k)')
        ax.plot(o.R, o.pct, '-', color=color, marker=marker, label='Ours (M = 5, N = 16k)')
        ax.axhline(0, color=INK_2, linewidth=0.6)
        # End values go in the title so they never collide with the lines
        ours_end, paper_end = o[o.R == 20].pct.iloc[0], p[p.R == 20].pct.iloc[0]
        ax.set_title(f'{name}\nat R=20: ours {ours_end:+.0f}%, paper {paper_end:+.0f}%',
                     loc='left', fontsize=8.5, linespacing=1.5)
        ax.set_xticks([4, 8, 12, 16, 20])
        ax.set_xlabel('R')
    axes[0].set_ylabel('AWCT change vs R = 4 (%)')
    handles = [Line2D([], [], linestyle='--', color=PAPER_GREY, linewidth=1.4),
               Line2D([], [], linestyle='-', color=INK_2, marker='o')]
    fig.legend(handles, ['Paper (M = 20, N = 64k), dashed grey',
                         'Ours (M = 5, N = 16k), solid, coloured per scheduler'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.text(0.01, 1.0, STAMP, fontsize=7, color=INK_2, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out / 'fig6_relative_increase.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--paper-64k', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _, agg = load_ours(args.ours)
    paper = load_paper()
    k64 = load_64k(args.paper_64k)
    plot_side_by_side(paper, agg, k64, args.out)

    ours_rel = relative(agg, 'mean')
    paper_rel = relative(paper, 'awct')
    plot_relative(paper_rel, ours_rel, args.out)

    table = (ours_rel.rename(columns={'mean': 'ours_mean', 'min': 'ours_min', 'max': 'ours_max',
                                      'count': 'ours_offsets', 'pct': 'ours_pct_vs_R4'})
             .merge(paper_rel.rename(columns={'awct': 'paper_awct', 'pct': 'paper_pct_vs_R4'}),
                    on=['scheduler', 'R'], how='left'))
    table['scheduler'] = pd.Categorical(table.scheduler, ORDER, ordered=True)
    table = table.sort_values(['scheduler', 'R'])
    table.to_csv(args.out / 'fig6_comparison.csv', index=False, float_format='%.3f')

    print(table.round(2).to_string(index=False))
    best = agg.loc[agg.groupby('R')['mean'].idxmin()][['R', 'scheduler', 'mean']]
    print('\nBest scheduler at each R (ours):')
    print(best.round(2).to_string(index=False))
    if k64:
        print('\nMember 1 64k R=4 run vs paper R=4:',
              {n: (round(k64[n], 2), float(paper[(paper.scheduler == n) & (paper.R == 4)].awct.iloc[0]))
               for n in ORDER})


if __name__ == '__main__':
    main()
