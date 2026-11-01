# Member 1: shared paper-scale point (64,000 jobs, M = 20, R = 4)

One run, one sample set (the paper uses 10), seed 2026, offset 15, original MRIS code. Run with
`scripts/run_chunk.sh results/member1 paper-64k --experiment jobs --factors 64 --machines 20 --resources 4 --runs 1 --max-jobs 64000 --max-rss-mb 3000`.
Not a full paper reproduction: a single point, not a curve.

| Scheduler (paper name) | AWCT | Paper (by eye, ±1) | Wall time |
|---|---|---|---|
| MRIS-WSJF | 17.285 | ≈17 | 3 h 54 m |
| PQ-WSJF (CA-PQ-WSJF) | 26.136 | ≈26.5 | 30 m |
| OnlinePQ-WSJF (PQ-WSJF) | 20.630 | ≈19.5–20 | 1 h 55 m |
| Tetris-instantaneous | 20.355 | ≈19.5–20 | 1 h 08 m |
| BF-EXEC | 19.399 | ≈19.5–20 | 40 m |

- Status `PASS_WITH_WARNINGS`: only the known CA-PQ `np.isclose` capacity issue (max excess 5.0 without tolerance; passes with it). All other checks passed for all schedulers.
- Total 29,228 s (8 h 07 m), peak RSS 1.10 GB, exit 0, started 2026-10-31 23:55, ended 2026-11-01 07:03.
- Parquet SHA-256 (file not committed): `33d3d6769b56f93f09e7dd5ba46051269add9b3e3daf912a033f7b4f944a9cb0`.
