# Member run plan: larger reduced-scale runs

**For each member.** Follow your own section exactly. Steps are in order. Everything here runs on an 8 GB Apple Silicon Mac.

The instructor approved reduced-scale simulations on 2026-10-30, so this plan never claims to be a full paper reproduction. Its goal is to move each figure closer to the paper's load, using the original MRIS code, and to record honest evidence.

Background: [AZURE_VALIDATION.md](AZURE_VALIDATION.md) (what has already been run and measured) and [FIGURE_REPRODUCTION.md](FIGURE_REPRODUCTION.md) (paper parameters).

## Rules for everyone (do not skip)

1. **Never edit** `scheduler/`, `packing/`, `experiments/` or `run.sh`. They are the authors' original code.
2. **Never run** `run.sh`, `download_data.sh` or `experiments/*.py` directly. They start one worker per CPU core, each loading the full trace, which exceeds 8 GB. Use only the scripts named below.
3. **One run at a time per laptop.** `scripts/run_chunk.sh` refuses to start a second one.
4. **Keep the lid open and the charger in** while a chunk runs. The wrapper uses `caffeinate -i`, but closing the lid can still sleep the Mac. Light use (browser, notes) is fine; each run uses about 1.2–1.5 GB of RAM and one CPU core.
5. **Use a new chunk name every time.** Never delete or overwrite earlier output.
6. If a chunk ends with status `FAIL`, **stop and report** the last lines of `logs/<chunk>.stderr.txt`. `PASS_WITH_WARNINGS` is expected: it is the known CA-PQ (`PQ-WSJF`) capacity issue described in AZURE_VALIDATION.md.
7. **Time estimates are extrapolated** from measured 8,000-job runs, not measured at 16,000 jobs. Record the real times.
8. Label every plot and table **"reduced-scale"**. Label Fig. 4 and Fig. 6 runs also **"load-matched, not paper configuration"**, because they use fewer machines than the paper.

## Step 0: everyone (about 15 minutes, once)

1. Get the code. This plan needs PR #14 merged into `main`.
   ```sh
   git fetch origin
   git switch <your branch>          # setup/mac-support, experiments/jobs, experiments/machines or experiments/resources
   git merge --ff-only origin/main
   ```
2. Environment: follow [MAC_SETUP.md](MAC_SETUP.md) sections 2 and 4.
   - Python 3.12 `.venv` from `requirements-mac.lock.txt`.
   - Azure trace at `experiments/data/packing_trace_zone_a_v1.sqlite`.
   - `md5 -q` must print `fd7a6e124920eaeab25178f7363a5df3`.
3. Repeatability check (about 20 s, about 1.2 GB):
   ```sh
   scripts/run_chunk.sh results/<you> check-1000 --experiment jobs --factors 4096 --runs 1
   ```
   Open `results/<you>/chunks/check-1000/summary.json`. Expect `"offset": 975`, MRIS-WSJF AWCT 10.401, PQ-WSJF 14.820, and OnlinePQ-WSJF / Tetris / BF-EXEC 6.966. Report any mismatch before continuing.

`<you>` is `member1`, `member2`, `member3` or `member4`. Run every command from the repository root.

## Member 1: shared paper-scale point (overnight)

This point (64,000 jobs, 20 machines, 4 resource types) appears in all three figures: Fig. 3 at 64k, Fig. 4 at M = 20 and Fig. 6 at R = 4. One run gives every member a real paper-scale comparison point.

1. Wait until at least one 16,000-job chunk has finished (yours, or from Members 2–4) and check that its time is near the estimate. If 16k took much longer than about 1 h per run, tell the group before starting.
2. Overnight (estimated 4–12 h or more, not measured; memory at this size not measured):
   ```sh
   scripts/run_chunk.sh results/member1 paper-64k --experiment jobs --factors 64 --machines 20 --resources 4 --runs 1 --max-jobs 64000
   ```
   If the memory watchdog stops it (`RSS ... exceeded limit` in stderr), report it. Do not raise `--max-rss-mb` above 3000 on an 8 GB Mac.
3. Paper values for comparison (read by eye, ±1): MRIS ≈17, CA-PQ ≈26.5, PQ / Tetris / BF-EXEC ≈19.5–20. `summary.json` has the AWCT values. Do not plot this single run as a curve.
4. Also: review Members 2–4's evidence PRs for complete logs, RAM and time records.

## Member 2: Figure 3 (jobs)

| Chunk | Jobs | Est. time | Paper value at that size (±1) |
|---|---|---|---|
| `small` | 1k / 2k / 4k / 8k | 17 min (measured) | 8k: MRIS ≈10, CA-PQ ≈15.5, others ≈7 |
| `f256` | 16k | ~1 h | 16k: MRIS ≈11, CA-PQ ≈17, others ≈8.5 |
| `f128` (optional, evening or overnight) | 32k | ~3–5 h | 32k: MRIS ≈14, CA-PQ ≈20, others ≈12 |

```sh
scripts/run_chunk.sh results/member2 small --experiment jobs --factors 4096 2048 1024 512 --runs 2
scripts/run_chunk.sh results/member2 f256 --experiment jobs --factors 256 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member2 f128 --experiment jobs --factors 128 --runs 2 --max-jobs 32000
```

The `small` chunk uses the same seed and offsets as the committed sweep, so its AWCT values should equal `evidence/reduced-figures/fig3-jobs/summary.json` exactly. This is a good repeatability check. All chunks use the paper's own settings (M = 20, R = 4) with 2 sample sets instead of 10. The paper's crossover, where MRIS becomes best, is near 45k jobs. Look for the gap between MRIS and the others to shrink as jobs increase.

Merge and plot. Leave out `f128` if it was not run.

```sh
python scripts/merge_runs.py --output results/member2/fig3-jobs results/member2/chunks/small results/member2/chunks/f256 results/member2/chunks/f128
MPLBACKEND=Agg python scripts/plot_reduced_figures.py results/member2
```

## Member 3: Figure 4 (machines), load-matched

16,000 jobs on fewer machines gives the paper's jobs-per-machine load. The paper uses 64,000 jobs on 5, 10, 20 and 40 machines.

| Chunk | Jobs per machine | Paper equivalent | Est. time |
|---|---|---|---|
| `m10` (run first, for timing) | 1,600 | M = 40 | ~1 h |
| `m5` | 3,200 | M = 20 | ~1 h |
| `m3` | 5,333 | between M = 10 and 20 | ~1 h+ |
| `m2` | 8,000 | between M = 5 and 10 | ~1 h+ (heavier load may be slower) |

Run one per session, in this order:

```sh
scripts/run_chunk.sh results/member3 m10 --experiment machines --factors 256 --machines 10 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member3 m5 --experiment machines --factors 256 --machines 5 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member3 m3 --experiment machines --factors 256 --machines 3 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member3 m2 --experiment machines --factors 256 --machines 2 --runs 2 --max-jobs 16000
```

Paper claim to test: MRIS beats Tetris, PQ and BF-EXEC under heavy load (the paper's M = 5–20), and the online schedulers catch up at light load (M = 40). The earlier 4,000-job run reached only 2,000 jobs per machine and MRIS did not win there.

```sh
python scripts/merge_runs.py --output results/member3/fig4-machines results/member3/chunks/m10 results/member3/chunks/m5 results/member3/chunks/m3 results/member3/chunks/m2
MPLBACKEND=Agg python scripts/plot_reduced_figures.py results/member3
```

## Member 4: Figure 6 (resource types), load-matched

16,000 jobs on **5 machines** = 3,200 jobs per machine, exactly the paper's load (64,000 jobs on 20 machines). Use factor 256 throughout; do not mix factors in one folder.

Five chunks of about 1 h each. Run one per session, `r4` first for timing:

```sh
scripts/run_chunk.sh results/member4 r4 --experiment resources --factors 256 --machines 5 --resources 4 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r8 --experiment resources --factors 256 --machines 5 --resources 8 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r12 --experiment resources --factors 256 --machines 5 --resources 12 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r16 --experiment resources --factors 256 --machines 5 --resources 16 --runs 2 --max-jobs 16000
scripts/run_chunk.sh results/member4 r20 --experiment resources --factors 256 --machines 5 --resources 20 --runs 2 --max-jobs 16000
```

Paper claim to test: from R = 4 to 20, Tetris AWCT rises ≈80% and MRIS only ≈17%, and MRIS stays best. The earlier 2,000-jobs-per-machine test gave MRIS +15.8% and Tetris +150%.

```sh
python scripts/merge_runs.py --output results/member4/fig6-resources results/member4/chunks/r4 results/member4/chunks/r8 results/member4/chunks/r12 results/member4/chunks/r16 results/member4/chunks/r20
MPLBACKEND=Agg python scripts/plot_reduced_figures.py results/member4
```

Also review the 64k-versus-32k evidence in [FIGURE6_JOB_COUNT.md](FIGURE6_JOB_COUNT.md) independently.

## Saving evidence and opening a PR (everyone, at the end)

Parquet files are not committed (Git ignores them). Commit the small records:

```sh
mkdir -p evidence/members/<you>/logs
cp -R results/<you>/chunks evidence/members/<you>/chunks && find evidence/members/<you>/chunks -name "*.parquet" -delete
cp results/<you>/logs/*.timing.txt results/<you>/logs/*.memory.txt results/<you>/logs/*.stdout.txt evidence/members/<you>/logs/
# Members 2-4 only (the merged figure folder and plots):
cp results/<you>/fig*/summary.json evidence/members/<you>/merged-summary.json
cp -R results/<you>/figures evidence/members/<you>/figures
git add evidence/members/<you>
git commit -m "<Member N>: <figure> load-matched reduced-scale runs"
git push
gh pr create --base main --title "<Member N>: <figure> runs" --body "<what ran, times, peak RAM, statuses, comparison with paper>"
```

Before committing, check that no file under `evidence/members/<you>` contains your home folder path (`grep -r "/Users/" evidence/members/<you>`). If one does, remove the path.

Do not merge your own PR. Another member reviews it, as listed in [TEAM_CONTRIBUTIONS.md](TEAM_CONTRIBUTIONS.md).

## What to report to the group

- Each chunk: status, wall time and peak RSS (from `logs/<chunk>.timing.txt`).
- The `awct_summary.csv` table and the PNG.
- One sentence: does the trend match the paper's claim at this load? If not, why might it differ (scale, 2 sample sets, fewer machines)?
