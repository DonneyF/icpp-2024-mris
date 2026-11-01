# Figure 6 job count: 64,000 versus 32,000

**Finding:** the published Figure 6 is consistent with **64,000 jobs (downsample factor 64)**, as its caption states, and not with the 32,000 jobs (factor 128) in upstream `run.sh`. Use factor 64 as the primary Figure 6 configuration. Factor 128 stays available as a clearly labelled upstream-script variant. This conclusion rests on the caption plus a cross-figure consistency check. The authors' raw Figure 6 files are not available, so it is strong indirect evidence, not proof.

## Sources checked

| Source | Job count for the resource sweep | Notes |
|---|---|---|
| Paper, page 9, Figure 6 caption | **64,000** (M = 20) | "Effect of the number of resource types on AWCT, with M = 20 machines and N = 64000 jobs." |
| Paper, section 7.1 | Method only | Uses the first 4.096M jobs, downsampling factor *f*, and 10 sampled offsets per point. Its worked example is 64,000 jobs with *f* = 64. |
| Upstream `run.sh`, `scheduler_benchmark_resources()` | **32,000** (`DOWNSAMPLE_FACTOR=128`) | The only executable sweep definition in the repository. |
| Upstream `experiments/5_scheduler_benchmark_resources.ipynb`, first cell | **32,000** | The markdown text says "downsample this to get a total of 32000 jobs". |
| Upstream `experiments/4_scheduler_benchmark_machines.ipynb`, first cell | 32,000 | The same sentence appears here, but `run.sh` (factor 64) and the Figure 4 caption both say 64,000. The sentence is copy-pasted text, so it is weak evidence. |
| `5_scheduler_benchmark_resources.py` argparse default | 8,000 (`--downsample_factor 512`) | Not used by `run.sh`. Not relevant to the figure. |
| Upstream `generate_all_plots.py` and the notebook plotting code | Not encoded | Output files are named `5_scheduler_benchmark_resources_{R}_{run}.parquet`. The name has no factor, so the plot cannot tell which factor produced the data. |

## Why the two numbers differ

The repository records the sweep as 32,000 jobs in two places: `run.sh` and the resources notebook text. The paper caption says 64,000. Neither the paper nor the repository explains the difference. The likely explanation is a stale or copy-pasted configuration. The machines notebook carries the same "32000" sentence even though its own sweep runs at 64,000. Because the factor is not in the result file names, a figure made from factor-64 files and one made from factor-128 files cannot be told apart from the code alone.

## Cross-figure consistency check (evidence that the published figure used 64,000)

In the upstream code, the R = 4 point of the resource sweep is the same experiment as the M = 20 point of the machine sweep and as one point of the jobs sweep. All three use 20 machines, the same unaugmented four-resource Azure cache, the same five schedulers, and the same `[offset::f][:4096000//f]` slice. The only thing that can differ is the job count. So the published R = 4 values should match the published values at whichever job count Figure 6 actually used.

Values below were read by eye from page 9 of the PDF, rendered at 6x. They are approximate, to about ±1 AWCT unit.

| Scheduler (paper label) | Fig. 6 at R = 4 | Fig. 4 at M = 20 (N = 64,000) | Fig. 3 at N = 64,000 | Fig. 3 at N = 32,000 |
|---|---|---|---|---|
| MRIS-WSJF | ≈17 | ≈17 | ≈17 | ≈14 |
| CA-PQ-WSJF | ≈26.5 | ≈27 | ≈26.5 | ≈20 |
| PQ-WSJF / Tetris / BF-EXEC | ≈19.5–20.5 | ≈19.5–20 | ≈19.5–20 | ≈12 |
| Ranking | MRIS best | MRIS best | MRIS best | Tetris/PQ/BF-EXEC best; MRIS second |

The Figure 6 R = 4 values match the 64,000-job points in Figures 3 and 4. They do not match the 32,000-job point, where the ranking is also reversed: at 32,000 jobs, Tetris, PQ and BF-EXEC beat MRIS. The paper's Figure 6 text gives "Tetris +80%" and "MRIS +17%" from 4 to 20 resources. That text applies to the published 64,000-job data.

What this check cannot rule out:
- that the authors re-ran with a different setting that happened to give the same values;
- reading error of about one unit.

The rankings and the gap between the two job counts (about 3 units for MRIS, 6 for CA-PQ, 8 for the online baselines) are much larger than that reading error.

## What changed in this repository

- `docs/FIGURE_REPRODUCTION.md` now lists factor 64 (64,000 jobs) as the primary Figure 6 configuration. It cites this evidence and keeps factor 128 as a separate, labelled variant.
- `run.sh` and all upstream scripts and notebooks are **not edited**. Upstream `./run.sh 5` still runs factor 128. Run Figure 6 through the commands in `FIGURE_REPRODUCTION.md`, or pass `--downsample_factor 64` to `5_scheduler_benchmark_resources.py`.
- Keep factor-64 and factor-128 outputs in **separate directories**, because their file names are identical.
- Both configurations are prepared in `scripts/azure_pilot.py`: `--experiment resources --factors 64 --max-jobs 64000 …` (primary) or `--factors 128 --max-jobs 64000 …` (upstream-script variant). A dry run built both input sets: 64,000 and 32,000 jobs, each spanning 12.39 days, at R = 4 and 20, with peak RSS 1.10 GiB. Neither has been scheduled yet, because each 64,000-job MRIS run is extrapolated at about 4–12 h on the laptop. See `docs/AZURE_VALIDATION.md`.
- Reduced-scale support for the published trend: in a load-matched variant (4,000 jobs on 2 machines), MRIS rose 15.8% from R = 4 to 20 (paper: 17%) while Tetris rose 150% (paper: 80%). At 20 machines with 4,000 jobs, the system is too lightly loaded to show any effect. This does not distinguish 64k from 32k. It only shows the code reproduces the qualitative effect.
