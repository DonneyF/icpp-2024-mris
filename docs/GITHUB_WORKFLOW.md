# Team GitHub structure and workflow

Local setup branch: `setup/mac-support`; `main` remains the cloned baseline. Remote `upstream` points to `https://github.com/DonneyF/icpp-2024-mris.git`. The team origin is `https://github.com/Vishwabelaganti/mris-os-research-reproduction.git`. Setup is published through a pull request. The owner handles collaborator invitations in GitHub.

The confirmed owner is `Vishwabelaganti`. The repository is a fork retaining original history. No upstream LICENSE was found; do not add a license claiming ownership of the research code or copy it into an independently licensed repository without clarifying terms.

Proposed branches: `main` for reviewed working code; `setup/mac-support` for environments; `experiments/reproduction` for wrappers/manifests; `docs/report-presentation` for later deliverables. Keep algorithm changes separate from infrastructure changes, with evidence and a mathematical explanation.

For another checkout, configure the team remote if it is missing:

```sh
git remote add origin https://github.com/Vishwabelaganti/mris-os-research-reproduction.git
git diff --check
git status --short
# Review and commit only intended source, documentation, and small evidence files.
git push -u origin setup/mac-support
```

Open a pull request into team main. Recommend one peer approval, passing lightweight checks, no force pushes, and no direct main pushes, subject to the account's available branch protection features. The existing upstream release workflow publishes Docker images; do not trigger it as a smoke test. A separate read-only smoke workflow is included.

## Issue backlog

| Issue | Acceptance condition |
|---|---|
| Repository setup | Confirm owner/collaborators; fork, preserve attribution, review reuse terms, configure origin and protection |
| Mac compatibility | Second ARM64 member repeats setup; record Intel result if available |
| Dependency testing | pip check and GLPK integer solve pass with locked environment |
| First simulation | Serial Azure pilot has feasible schedules and memory/time logs |
| Figure reproduction 1 | Figure 3 manifest, raw outputs, mean/CI plot, scale disclosure |
| Figure reproduction 2 | Figure 4 same evidence |
| Figure reproduction 3 | Figure 6 resolves 32k/64k discrepancy and records resource seed |
| Result analysis | Explain differences and identify noncomparable settings |
| Presentation slides | Required sections and validated original/generated comparisons; due Nov 6 |
| Final report | Template sections, contribution detail, three figure pairs, references |
| Demonstration video | Clean environment-to-results walkthrough; no secrets |
