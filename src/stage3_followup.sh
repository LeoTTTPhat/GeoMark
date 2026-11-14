#!/bin/bash
# Stage-3 follow-up (one guarded job): E9 key-randomised null -- smoke check first (20 keys), then the full run
# (500 keys), then re-aggregate + plots. Launch:  python src/tools/slot_watch.py --name p1-geomark-trajectory-watermark-full-followup -- bash src/stage3_followup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
$PY src/run_radio_keynull.py --cfg src/configs/geomark_st.yaml --n_keys 20
$PY src/run_steps.py --cfg src/configs/full.yaml --steps radio-keynull
$PY src/run_steps.py --cfg src/configs/full.yaml --steps aggregate,plots --force
