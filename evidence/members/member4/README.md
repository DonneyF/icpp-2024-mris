# Member 4: load-matched reduced-scale Figure 6 (number of resource types)

All five planned chunks (r4, r8, r12, r16, r20) from [MEMBER_RUN_PLAN.md](../../../docs/MEMBER_RUN_PLAN.md) completed. The merged sweep covers R = 4, 8, 12, 16 and 20 resource types with 16,000 jobs on 5 machines (factor 256), using two sampled offsets per point: 10 sampled datasets and 50 scheduler results. This is **reduced-scale and load-matched, not the paper configuration**. 16,000 jobs on 5 machines gives 3,200 jobs per machine, the same load as the paper's 64,000 jobs on 20 machines. The paper uses ten offsets and every R from 4 to 20.

Hardware: Apple M3 Pro (`Mac15,6`), 18 GB RAM, macOS 26.6.2, Python 3.12.6 in the `.venv` from `requirements-mac.lock.txt`. Azure trace MD5 `fd7a6e124920eaeab25178f7363a5df3` matched. Runs were serial, one at a time, through `scripts/run_chunk.sh`.

| Chunk | Status | Elapsed time (wrapper) | Peak RSS |
|---|---|---|---|
| check-1000 | PASS | 28.38 seconds | 1,614,528,512 bytes |
| r4 | PASS_WITH_WARNINGS | 1,546.23 seconds (25.8 min) | 1,544,847,360 bytes |
| r8 | PASS_WITH_WARNINGS | 1,639.55 seconds (27.3 min) | 1,541,603,328 bytes |
| r12 | PASS_WITH_WARNINGS | 1,900.25 seconds (31.7 min) | 1,519,730,688 bytes |
| r16 | PASS_WITH_WARNINGS | 2,059.12 seconds (34.3 min) | 1,517,502,464 bytes |
| r20 | PASS_WITH_WARNINGS | 2,296.72 seconds (38.3 min) | 1,513,832,448 bytes |

Total for the five chunks: 2 h 37 min. The plan's estimate was about 1 h per chunk; each took 26–38 min, growing with R. The initial check matched offset 975 and every expected AWCT value (MRIS-WSJF 10.401, PQ-WSJF 14.820, OnlinePQ-WSJF / Tetris / BF-EXEC 6.966). Every chunk used offsets 60 and 163, so all R values are compared on the same two job samples. These are the same offsets as Member 2's 16k chunk, as expected for seed 2026 and factor 256. `merge_runs.py` checked the SHA-256 of all ten schedule files before merging.

PASS_WITH_WARNINGS records the known upstream CA-PQ (`PQ-WSJF` in the runner) capacity-tolerance issue. In all 10 datasets CA-PQ is the only scheduler that fails the strict capacity check. All schedulers passed the all-jobs-once, valid-machine and no-start-before-release checks. Scheduler, packing and experiment source files were unchanged.

## Results

Mean AWCT over the two offsets (lower is better; min and max are in `figures/awct_summary.csv`). Scheduler names follow the paper.

| R | MRIS-WSJF | CA-PQ-WSJF | PQ-WSJF | Tetris | BF-EXEC |
|---|---|---|---|---|---|
| 4 | **17.14** | 31.03 | 22.60 | 22.24 | 21.02 |
| 8 | **18.52** | 38.07 | 28.18 | 30.33 | 28.62 |
| 12 | **19.72** | 45.44 | 34.85 | 36.42 | 35.34 |
| 16 | **20.73** | 49.56 | 40.63 | 41.62 | 39.85 |
| 20 | **21.58** | 52.18 | 43.26 | 44.39 | 44.59 |

MRIS has the lowest AWCT in every one of the 10 sampled datasets. Its margin over the next-best scheduler grows with R, from 2.52 / 5.24 at R = 4 to 17.92 / 25.44 at R = 20 (offset 60 / offset 163).

## Comparison with the paper's claim

Paper, section 7.5.3: from R = 4 to 20, "the AWCT of Tetris increases by 80%, whereas for MRIS, this is only 17%", and MRIS stays best.

| AWCT change, R = 4 → 20 | MRIS-WSJF | CA-PQ-WSJF | PQ-WSJF | Tetris | BF-EXEC |
|---|---|---|---|---|---|
| Paper (digitized Figure 6) | +17% | +72% | +82% | +82% | +87% |
| Ours (load-matched) | +26% | +68% | +91% | +100% | +112% |

**The qualitative claim reproduces.** MRIS degrades far less than every baseline as resource types are added, and it is best at every R. In both the paper and our runs, Tetris's AWCT at R = 20 is about twice MRIS's (paper 37.4 / 20.1 = 1.86; ours 44.39 / 21.58 = 2.06). The **magnitudes differ**: both MRIS and the online baselines rise more steeply in our runs (MRIS +26% versus +17%; Tetris +100% versus +80%). At R = 4 our MRIS value (17.14) matches the paper (17.2). Our online baselines are 1.4–2.7 AWCT higher than the paper's (for example Tetris 22.24 versus 20.6), and CA-PQ is 4.3 higher (31.03 versus 26.7).

Likely reasons, not separated by these runs:
- **Fewer machines.** Load-matching keeps jobs per machine equal, but 5 machines give the online schedulers fewer places to put a job that arrives while others are busy. The paper's 20 machines can absorb bursts across more machines. This would hurt the baselines more than MRIS, as observed at R = 4.
- **Two offsets instead of ten.** The two offsets differ by up to 8.2 AWCT for the baselines at R = 16–20 (for example Tetris 40.27 versus 48.51 at R = 20). Ten offsets would narrow this. MRIS's spread is small (at most 0.87).
- **Smaller sample per offset.** Factor 256 takes every 256th job instead of every 64th. Each sample is a different, sparser slice of the same 12.39-day trace.
- **Augmentation randomness.** The extra resource columns follow the upstream method: each job's new demand is the CPU demand of a uniformly sampled job. The runner seeds this per trace index (documented in `scripts/azure_pilot.py`), while upstream uses one unseeded draw per R. The paper's text describes the sampling in the reverse direction ("j′'s demand ... is set to j's CPU demand"). The code's version is the one used, and it gives every job an independent draw from the CPU-demand distribution.
- **Digitization error.** Paper values were read from the PDF (see below), about ±0.5 AWCT.

## Figures

- `figures/comparison/fig6_paper_vs_ours.png`: the published Figure 6 (digitized) next to ours on the same AWCT scale. It also marks Member 1's paper-scale 64k run at R = 4.
- `figures/comparison/fig6_relative_increase.png`: AWCT change relative to R = 4 for each scheduler, paper versus ours. This is the scale-free test of the paper's claim.
- `figures/comparison/fig6_comparison.csv`: every value in both comparison figures.
- `figures/fig6-resources/5_scheduler_benchmark_resources.png`: the upstream plotting routine via `scripts/plot_reduced_figures.py`, stamped REDUCED-SCALE. Its shaded bands are upstream's 95% Student-t intervals. With only two offsets (1 degree of freedom, t = 12.7) they are very wide and extend below zero. They reflect the sample size, not the spread of results. Use the min–max bands in the comparison figure instead.

The comparison figures use a colour-blind-checked palette with distinct markers per scheduler. The paper's own colours fail a red-green (protan) colour-vision separation check (PQ orange versus CA-PQ green).

`compare_with_paper.py` regenerates the comparison figures. Run from the repository root:

```sh
MPLBACKEND=Agg .venv/bin/python evidence/members/member4/compare_with_paper.py \
    --ours results/member4/fig6-resources/summary.json \
    --paper-64k evidence/members/member1/chunks/paper-64k/summary.json \
    --out results/member4/figures/comparison
```

## Digitized paper Figure 6

`paper_fig6_digitized.json` holds the published Figure 6 values for all five schedulers at R = 4…20. Page 9 of the PDF was rendered at 400 dpi. The axes were calibrated from the major tick marks, and each curve was isolated by its line colour. Gaps in dashed lines were filled with local linear fits. As a check, the digitized curves give MRIS +16.9% and Tetris +81.6% from R = 4 to 20, matching the paper's text (17% and 80%).

## Independent review of FIGURE6_JOB_COUNT.md

I agree with its conclusion that the published Figure 6 used 64,000 jobs (factor 64), and the evidence is now stronger than the by-eye reading it rests on.

- **Visual readings.** The digitized R = 4 values (MRIS 17.2, CA-PQ 26.7, PQ 19.9, Tetris 20.6, BF-EXEC 19.6) agree with the document's readings (≈17, ≈26.5, ≈19.5–20.5).
- **Measured 64k point.** Member 1's paper-scale run (64,000 jobs, M = 20, R = 4, offset 15) gave MRIS 17.28, CA-PQ 26.14, PQ 20.63, Tetris 20.35, BF-EXEC 19.40. All five are within 0.73 of the published R = 4 values, with the same ranking (MRIS best).
- **Measured 32k point.** Member 2's 32,000-job runs (M = 20, R = 4, two offsets) average MRIS 13.84, CA-PQ 20.15, PQ 11.84, Tetris 11.87, BF-EXEC 11.66. These are 3.4–8.7 AWCT away from the published R = 4 values, and the ranking is reversed (three baselines beat MRIS).
- **Limits.** One 64k offset is not ten, so this is still not proof. It is consistent only with the 64,000-job configuration in the caption.

Not done: the full factor-64 R sweep (64,000 jobs, M = 20) and the factor-128 upstream-script variant. At about 8 h per 64k point on a laptop (Member 1's measured time), a 5-point sweep would need about 40 h per offset. These load-matched runs replace it under the instructor's 2026-10-30 approval of reduced-scale simulations.

## Reproduce

```sh
scripts/run_chunk.sh results/member4 check-1000 --experiment jobs --factors 4096 --runs 1
scripts/run_chunk.sh results/member4 r4 --experiment resources --factors 256 --machines 5 --resources 4 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r8 --experiment resources --factors 256 --machines 5 --resources 8 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r12 --experiment resources --factors 256 --machines 5 --resources 12 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r16 --experiment resources --factors 256 --machines 5 --resources 16 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r20 --experiment resources --factors 256 --machines 5 --resources 20 --runs 2 --max-jobs 16000
python scripts/merge_runs.py --output results/member4/fig6-resources results/member4/chunks/r4 results/member4/chunks/r8 results/member4/chunks/r12 results/member4/chunks/r16 results/member4/chunks/r20
MPLBACKEND=Agg python scripts/plot_reduced_figures.py results/member4
```

Parquet schedules are not committed. Their SHA-256 hashes are in each `chunks/*/summary.json` and in `merged-summary.json`.

## Contribution note

The runs were executed on Member 4's laptop. Following the team's convention in TEAM_CONTRIBUTIONS.md, Member 4 reviews these results and adds their own explanation before this evidence is attributed to them.
