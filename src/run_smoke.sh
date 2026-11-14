#!/bin/bash
# Smoke test: tiny subsets + smallest models, every step under memguard (names p1-geomark-trajectory-watermark-smoke-<step>).
# Outputs: results/smoke/ (summary.json, step_times.tsv, compute_times.tsv, figures/). Usage: src/run_smoke.sh [step ...]
# Re-running skips steps with a done-marker in results/smoke/.done/ (FORCE=1 to redo; outputs are checkpointed anyway).
cd "$(dirname "$0")/.." || exit 1
for r in 1 2; do
  SLOT_WATCH=${SLOT_WATCH:-0} TICKETS=${TICKETS:-1} CFG=src/configs/geomark_st.yaml TAG=smoke bash src/pipeline.sh "$@" && exit 0
  rc=$?; [ $rc -eq 130 ] || [ $rc -eq 3 ] && exit $rc
  echo "smoke round $r incomplete; retrying unfinished steps"; sleep 60
done
exit 1
