#!/bin/bash
# Full experiment matrix (PLAN.md Sec. 6). Supervisor: re-runs the (idempotent, checkpointed) pipeline until every step
# has succeeded (done-markers in results/.done/), at most $ROUNDS rounds. Every step runs under memguard with
#   --name p1-geomark-trajectory-watermark-full-<step>
# Long: run detached, e.g.  nohup src/run_full.sh > logs/full.log 2>&1 &
# Individual steps: src/run_full.sh train gen ...  (FORCE=1 re-runs steps that have a done-marker; they skip finished work)
# Stop cleanly: touch .memguard_stop (in the project folder) and/or touch shared/.memguard/STOP-<name>
cd "$(dirname "$0")/.." || exit 1
LOCK=logs/run_full.lock
mkdir -p logs
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "run_full.sh already running (pid $(cat "$LOCK")); exiting"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
ROUNDS=${ROUNDS:-4}
for r in $(seq 1 "$ROUNDS"); do
  echo "[$(date '+%F %T')] ===== run_full round $r/$ROUNDS ====="
  SLOT_WATCH=${SLOT_WATCH:-0} TICKETS=${TICKETS:-1} CFG=src/configs/full.yaml TAG=full bash src/pipeline.sh "$@"
  rc=$?
  [ $rc -eq 0 ] && { echo "[$(date '+%F %T')] run_full: all steps done"; exit 0; }
  [ $rc -eq 130 ] || [ $rc -eq 3 ] && { echo "run_full: stopped (rc=$rc)"; exit $rc; }
  echo "[$(date '+%F %T')] round $r incomplete (rc=$rc); retrying unfinished steps in 5 min"
  sleep 300
done
echo "run_full: steps still failing after $ROUNDS rounds; see logs/full.log"; exit 1
