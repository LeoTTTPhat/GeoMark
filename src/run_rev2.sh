#!/bin/bash
# Stage 8 supervisor: runs a comma-separated list of rev2 steps under ONE memguard job and retries unfinished steps.
# Usage (detached): nohup src/run_rev2.sh <group-name> <step1,step2,...> > logs/rev2_<group>.log 2>&1 &
# Done-markers: results/rev2/.done/<step>. Stop cleanly: touch shared/.memguard/STOP-p1-geomark-trajectory-watermark-rev2-<group>
cd "$(dirname "$0")/.." || exit 1
G=$1; STEPS=$2; ROUNDS=${ROUNDS:-3}
LOCK=logs/run_rev2_$G.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "rev2 $G already running"; exit 0; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
PY=${PY:-python}
GUARD=${MEMGUARD:-tools/memguard.py}
for r in $(seq 1 "$ROUNDS"); do
  echo "[$(date '+%F %T')] rev2 $G round $r/$ROUNDS: $STEPS"
  $GUARD --name "p1-geomark-trajectory-watermark-rev2-$G" -- $PY src/rev2_steps.py --steps "$STEPS"
  rc=$?
  [ $rc -eq 0 ] && { echo "[$(date '+%F %T')] rev2 $G: all steps done"; exit 0; }
  [ $rc -eq 130 ] && { echo "rev2 $G: stopped"; exit 130; }
  echo "[$(date '+%F %T')] rev2 $G round $r rc=$rc; retry in 120 s"; sleep 120
done
echo "rev2 $G: steps still failing"; exit 1
