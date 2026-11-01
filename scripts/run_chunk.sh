#!/bin/bash
# Run ONE scripts/azure_pilot.py chunk with logging, timing and sleep prevention.
# Usage: scripts/run_chunk.sh <output-root> <chunk-name> <azure_pilot.py arguments...>
# Writes <output-root>/chunks/<chunk-name>/ and <output-root>/logs/<chunk-name>.*
# Example: scripts/run_chunk.sh results/member3 m10 --experiment machines --factors 256 --machines 10 --runs 2 --max-jobs 16000
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?output root required}"
NAME="${2:?chunk name required}"
shift 2
cd "$ROOT" || exit 2  # paths below are relative to the repository root
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$OUT/chunks" "$OUT/logs"
LOG="$OUT/logs/$NAME"
if pgrep -f "scripts/azure_pilot.py" > /dev/null; then
  echo "Another azure_pilot.py is already running on this machine; run chunks one at a time." >&2
  exit 2
fi
echo "start $(date '+%F %T')" | tee "$LOG.timing.txt"
memory_pressure | tail -1 | tee "$LOG.memory.txt"
# caffeinate -i keeps the Mac awake while the run lasts (keep the lid open and the charger in)
caffeinate -i /usr/bin/time -l "$ROOT/.venv/bin/python" scripts/azure_pilot.py --output-dir "$OUT/chunks/$NAME" "$@" \
  > "$LOG.stdout.txt" 2> "$LOG.stderr.txt"
STATUS=$?
memory_pressure | tail -1 | tee -a "$LOG.memory.txt"
grep -E " real| user| sys|maximum resident set size|peak memory footprint" "$LOG.stderr.txt" | sed 's/.*\[A//' | tee -a "$LOG.timing.txt"
echo "end $(date '+%F %T') exit=$STATUS" | tee -a "$LOG.timing.txt"
tail -1 "$LOG.stdout.txt"
exit $STATUS
