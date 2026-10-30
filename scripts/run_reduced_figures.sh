#!/bin/bash
# Reduced-scale, strictly serial Figure 3/4/6 validation on the Azure trace.
# NOT a paper reproduction: fewer jobs and 2 instead of 10 sampled offsets per point.
# Usage: scripts/run_reduced_figures.sh <output-root> [fig3|fig4|fig6|fig6-loaded|fig6-config ...]
# fig6-loaded is NOT a paper configuration: 2 machines raise per-machine load toward the paper's level.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?output root required, e.g. results/reduced-2026-10-29}"
shift
STEPS=("${@:-fig3 fig4 fig6 fig6-config}")
PY="$ROOT/.venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$OUT/logs"

run() {  # run <name> <runner args...>
  local name="$1"; shift
  echo "== $name: $(date '+%F %T')"
  memory_pressure | tail -1 | tee "$OUT/logs/$name.memory-before.txt"
  /usr/bin/time -l "$PY" "$ROOT/scripts/azure_pilot.py" --output-dir "$OUT/$name" "$@" \
    > "$OUT/logs/$name.stdout.txt" 2> "$OUT/logs/$name.stderr.txt"
  local status=$?
  memory_pressure | tail -1 | tee "$OUT/logs/$name.memory-after.txt"
  grep -E "real|maximum resident set size|peak memory footprint" "$OUT/logs/$name.stderr.txt"
  tail -1 "$OUT/logs/$name.stdout.txt"
  echo "exit=$status"
  return $status
}

for step in ${STEPS[*]}; do
  case "$step" in
    fig3) run fig3-jobs --experiment jobs --factors 4096 2048 1024 512 --machines 20 --resources 4 --runs 2 ;;
    fig4) run fig4-machines --experiment machines --factors 1024 --machines 2 5 10 20 40 --resources 4 --runs 2 ;;
    fig6) run fig6-resources --experiment resources --factors 1024 --machines 20 --resources 4 8 12 16 20 --runs 2 ;;
    fig6-loaded) run fig6-resources-loaded --experiment resources --factors 1024 --machines 2 --resources 4 12 20 --runs 2 ;;
    fig6-config) run fig6-config-dry-run --experiment resources --factors 64 128 --machines 20 --resources 4 20 \
                   --runs 1 --max-jobs 64000 --dry-run ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac || exit $?
done
