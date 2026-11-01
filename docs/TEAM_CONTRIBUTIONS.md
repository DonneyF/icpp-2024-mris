# Four-member contributions

These are provisional role numbers, not invented identities or account assignments. Add actual names, email addresses, R numbers and completed work when supplied. Each student must understand the paper, run their own commands, explain their result, and review someone else's work.

| Member | Primary responsibility | Issues | Review duty | Evidence to contribute |
|---|---|---|---|---|
| 1 | Mac compatibility, Azure pilot, shared testing documentation | #2–#5, coordinates #12 | Runner, memory and evidence records in every figure PR | Second-Mac pilot run, RAM/time measurements, review links |
| 2 | Figure 3: number of jobs / arrival rate | #6, shares #9 | Member 3's Figure 4 | Manifest (`summary.json`), raw Parquet hashes, plot, comparison with paper |
| 3 | Figure 4: number of machines | #7, shares #9 | Member 4's Figure 6 | Same evidence for the machine-count sweep |
| 4 | Figure 6: number of resource types | #8, shares #9 | Member 2's Figure 3 | Same evidence; factor-64 primary and factor-128 variant kept separate |

## Larger runs: per-member plan

The step-by-step commands for each member's larger, load-matched runs are in [MEMBER_RUN_PLAN.md](MEMBER_RUN_PLAN.md). It is written so each member can follow it directly. Only Member 1's 64,000-job point needs an overnight run; the other runs are split into chunks of about 1 h.

## Independent work: verified starting points (2026-10-29)

Each figure has its own runner step, output folder and Parquet name pattern, so the work can proceed in parallel without shared state. Every `run_reduced_figures.sh` step and the pilot command below was executed on the 8 GB M2 (see AZURE_VALIDATION.md); suggested next steps are not yet run. Each member should first bring their branch up to date with `main` after this PR merges (`git switch <branch> && git merge --ff-only origin/main`) and use a new output root each time.

- **Member 1 (`setup/mac-support`):** repeat `scripts/azure_pilot.py --experiment jobs --factors 4096 --output-dir results/pilot-<laptop>` on a second Mac. Compare the AWCT values with `evidence/azure-pilot/1000-jobs/summary.json` (same seed 2026 and offset 975 should give identical values) and record RSS and time. Keep the runner, troubleshooting and evidence conventions consistent across figure PRs.
- **Member 2 (`experiments/jobs`, Figure 3):** `scripts/run_reduced_figures.sh results/<root> fig3` (~17 min) reproduces the committed reduced sweep. The next step towards the paper is a 16,000-job point (`--factors 256`, one offset per invocation; not yet run, extrapolated at roughly 20–30 min per offset with MRIS dominating) to start testing the published crossover.
- **Member 3 (`experiments/machines`, Figure 4):** `scripts/run_reduced_figures.sh results/<root> fig4` (~14 min). The reduced sweep only reaches the paper's light-load side. Next, design a higher per-machine-load run (for example 8,000 jobs on M = 2 / 3 / 5) and document it as not a paper configuration, or plan a 64,000-job single point on HPC.
- **Member 4 (`experiments/resources`, Figure 6):** `scripts/run_reduced_figures.sh results/<root> fig6 fig6-loaded fig6-config` (~26 min). Use factor 64 (64,000 jobs) as the primary configuration and keep factor 128 in a separate folder. Read FIGURE6_JOB_COUNT.md and review its visual readings independently.

All four contribute their own verified sections to #10/#11 later and demonstrate their own workflow in #12. The bounded serial Azure path now exists (`scripts/azure_pilot.py`); Members 2–4 review it for their figure before extending it. Member 4's additional augmentation work should be balanced by Member 1 handling packaging and Member 2 handling the shared plot conventions.

## Contribution log (fill with real work)

| Member/name | Date | Commit or PR | Command/result or review | Own explanation of contribution |
|---|---|---|---|---|
| Pending | Pending | Pending | Pending | Pending |

The initial setup and tests were done as shared setup work; do not attribute them to a student who has not performed or reviewed them. Track actual student contributions as work proceeds.

Assignment checklist from the supplied documents: 1–5 group members; paper published 2022–2026 and OS-relevant; slides due November 6, 2026 at 11:59 pm; final package due December 1, 2026 at 11:59 pm, including original paper, report, and code/results demo video. The project description says presentations occur in the final two weeks, so November 6 is a slide deadline, not a confirmed presentation date. Confirm course timezone in the LMS.

Slides eventually need background, problem, novelty, method, simulation comparisons, conclusion. Report template needs detailed contributions, problem statement, solution, simulation results with at least three original/generated figure pairs and explanations, conclusion, and references. This initial package deliberately does not draft those deliverables.

Suggested checkpoints: Oct 30–Nov 1 environment and serial pilot; Nov 2–4 measured comparisons; Nov 5 review and disclose remaining gaps; Nov 6 submit slides. Complete broader validation and recorded walkthrough before the Dec 1 final deadline. These are planning targets, not evidence that results exist.
