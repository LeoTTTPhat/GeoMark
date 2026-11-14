#!/bin/bash
# Stage 10 supervisor: runs a comma-separated list of rev3 steps under ONE memguard job and retries unfinished steps.
# Usage (detached): nohup src/run_rev3.sh <group-name> <step1,step2,...> > logs/rev3_<group>.log 2>&1 &
# Done-markers: results/rev3/.done/<step>. Stop cleanly: touch shared/.memguard/STOP-p1-geomark-trajectory-watermark-rev3-<group>
cd "$(dirname "$0")/.." || exit 1
G=$1; STEPS=$2; ROUNDS=${ROUNDS:-2}
LOCK=logs/run_rev3_$G.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "rev3 $G already running"; exit 0; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
PY=${PY:-python}
GUARD=${MEMGUARD:-tools/memguard.py}
for r in $(seq 1 "$ROUNDS"); do
  echo "[$(date '+%F %T')] rev3 $G round $r/$ROUNDS: $STEPS"
  $GUARD --name "p1-geomark-trajectory-watermark-rev3-$G" -- $PY src/rev3_steps.py --steps "$STEPS"
  rc=$?
  [ $rc -eq 0 ] && { echo "[$(date '+%F %T')] rev3 $G: all steps done"; exit 0; }
  [ $rc -eq 130 ] && { echo "rev3 $G: stopped"; exit 130; }
  echo "[$(date '+%F %T')] rev3 $G round $r rc=$rc; retry in 60 s"; sleep 60
done
echo "rev3 $G: steps still failing"; exit 1
