# GitHub workflow for four students

Team repository: https://github.com/Vishwabelaganti/mris-os-research-reproduction

PR #1 merged as `b9f469e5ae1e660bcc59bd98e2554d91058d870f`, after local checks and both hosted smoke checks passed. `main` contains the reviewed setup. `origin` is the team fork; `upstream` is the original authors' repository. Never push team work to upstream.

## Access

The owner invites the other three students through Settings → Collaborators → Add people. Each student must accept the invitation and have write access before pushing team branches. No invitations or GitHub account assignments are made by this setup. Role labels identify responsibilities only. Until access is accepted, students can fork the team repository and submit pull requests from their own forks.

## Branch responsibilities

| Member | Branch | Scope |
|---|---|---|
| 1 | `setup/mac-support` | Installation, CI, memory measurements, integration and demo coordination |
| 2 | `experiments/jobs` | Paper Figure 3: job arrival rate |
| 3 | `experiments/machines` | Paper Figure 4: machine count |
| 4 | `experiments/resources` | Paper Figure 6: resource count |

The figure mappings have been verified against the paper. These branches start from the shared tested main. Keep changes in your branch and request a peer review. Members 2, 3, and 4 review each other's experiments in a cycle; Member 1 checks shared setup and evidence.

## First checkout

```sh
git clone https://github.com/Vishwabelaganti/mris-os-research-reproduction.git
cd mris-os-research-reproduction
git remote add upstream https://github.com/DonneyF/icpp-2024-mris.git
git fetch origin
git switch experiments/jobs  # choose your assigned branch
```

Use `git remote -v` first if working in an existing checkout; do not recreate existing remotes. GitHub passwords do not authenticate HTTPS pushes; use your existing Git credential setup or `gh auth login`. Never put a token in a file or command copied into the report.

## Daily work and pull requests

```sh
git status
git fetch origin
git merge origin/main
# Make your changes, then run setup/simulation checks.
git diff --check
git add docs/EXPERIMENTS.md  # replace with your intended small source/doc files
git diff --cached
git commit -m 'Describe the concrete experiment change'
git push -u origin HEAD
```

On GitHub, choose Compare & pull request with base `main` in the team repository. Describe the trigger, change, exact command, input checksum/seed, output files, time/RAM and limitations. Link the relevant issue. Ask a teammate to review. Merge only after the `smoke` check passes and review is complete. Keep long-lived team branches after merge and merge origin/main into them before the next task. Do not force-push shared branches or add environments, trace data or large results.

Recommend main protection requiring `smoke`, one peer approval, up-to-date branches, no force pushes, and no deletion. Protection status is recorded in the setup verification notes; a recommendation alone does not mean enforcement is enabled.

## Existing issue ownership

| Issues | Responsibility | Current work |
|---|---|---|
| #2, #3, #4 | Member 1 | Repo/access, second-Mac check, dependency validation |
| #5 | Member 1, Members 2–4 review | Serial Azure pilot passed (1,000 and 8,000 jobs); see AZURE_VALIDATION.md |
| #6 | Member 2 | Figure 3 |
| #7 | Member 3 | Figure 4 |
| #8 | Member 4 | Figure 6; 64k (factor 64) primary per FIGURE6_JOB_COUNT.md, factor 128 kept separate |
| #9 | All four | Each analyzes own work and cross-reviews |
| #10, #11 | All four | Later slides/report with actual contribution records |
| #12 | Member 1 coordinates; all demonstrate | Later demonstration video |

No issues are assigned to GitHub accounts until usernames are provided. The upstream Docker publishing workflow is retained but is not the validation workflow; do not trigger it for ordinary testing.
