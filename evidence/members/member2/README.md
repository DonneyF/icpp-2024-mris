# Member 2: completed reduced-scale Figure 3 experiments

All three planned chunks (small, f256, and optional f128) completed. The merged sweep covers 1,000, 2,000, 4,000, 8,000, 16,000 and 32,000 jobs on 20 machines with four resource types, using two sampled offsets per point. There are 12 sampled datasets and 60 scheduler results. This is reduced-scale validation, not a full paper reproduction; the paper uses ten offsets and extends to larger job counts.

| Chunk | Status | Elapsed time (wrapper) | Peak RSS |
|---|---|---|---|
| check-1000 | PASS | 11.62 seconds | 1,211,744,256 bytes |
| small | PASS_WITH_WARNINGS | 570.36 seconds | 1,239,449,600 bytes |
| f256 | PASS_WITH_WARNINGS | 1,833.22 seconds | 1,189,134,336 bytes |
| f128 | PASS_WITH_WARNINGS | 8,686.00 seconds | 1,282,686,976 bytes |

The 32k chunk elapsed time is 2 h 24 min 46 s, measured by the wrapper. The displayed start and end clock times span the daylight-saving fallback; elapsed timing is used for the duration.

The initial check matched offset 975 and all expected AWCT values. All 40 small-sweep AWCT values exactly matched the committed reference. The merge tool checked the SHA-256 of all twelve schedule files before merging.

At 32,000 jobs, mean AWCT was 13.843 for MRIS, 20.145 for CA-PQ, 11.838 for PQ, 11.873 for Tetris, and 11.655 for BF-EXEC. Lower AWCT is better. These are close to the plan's approximate paper values of 14, 20 and 12 at this job count. MRIS beats CA-PQ but trails the other three schedulers. Its gap to the best baseline narrows from about 2.79 at 16k to 2.19 at 32k. This sweep does not test the paper's crossover near 45k jobs and does not establish that MRIS becomes best.

PASS_WITH_WARNINGS records the known upstream CA-PQ capacity-tolerance issue. It is not a clean strict-capacity pass. Scheduler, packing, and experiment source files were unchanged.

The supplied plots use the upstream plotting routine and are labelled REDUCED-SCALE. Only two offsets per point are used, so uncertainty estimates rely on very few samples. The CSV supplies mean, minimum and maximum directly.

The complete merged data is in results/member2-complete/fig3-jobs; original chunk outputs remain in results/member2/chunks. This evidence supersedes the earlier package ending at 16k. Evidence is submitted on the experiments/jobs branch for team review.
