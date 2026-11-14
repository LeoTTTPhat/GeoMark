#!/bin/bash
# GeoMark pipeline driver. Steps (defined in src/run_steps.py) are executed in GROUPS; every group is ONE memguard job
#   memguard --name p1-geomark-trajectory-watermark-<TAG>-<group> -- python src/run_steps.py --cfg CFG --steps s1,s2,...
# (one slot acquisition per group: slots are shared machine-wide and waits are long). Every step is resumable
# (checkpointed outputs; done-markers in <out_dir>/.done/<step>), so re-running is safe. Each group has a retry loop.
# Usage: CFG=src/configs/full.yaml TAG=full src/pipeline.sh [group-spec ...]
#   group-spec = <step-or-alias>  or  <groupname>=<step>,<step>,...   (default: the group plan below for TAG)
# Steps: prep tests train-gpt train-<llm> gen-gpt gen-<llm> baselines-prep mapmatch regen-gpt regen-<llm> eval calib
#        baselines-eval utility steal radio capacity aggregate plots   (aliases: train, gen, regen)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=${PY:-python}
MG=${MEMGUARD:-tools/memguard.py}
CFG=${CFG:-src/configs/full.yaml}
TAG=${TAG:-full}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUT=$($PY -c "import sys; sys.path.insert(0,'src'); from geomark.common import load_cfg; print(load_cfg('$CFG')['out_dir'])")
DONE_DIR="$OUT/.done"
mkdir -p "$OUT" logs "$DONE_DIR"
FAILED=0

run_guarded() {  # run_guarded <guard-name> <cmd ...>: run cmd under memguard. SLOT_WATCH=1: start the guard the moment a
  # slot is released (src/tools/slot_watch.py, fair race against other tenants' supervisors). TICKETS>1: queue that many staggered
  # waiting guards ("tickets") for the same job; the first to obtain a slot runs it (src/tools/first.py), the others are
  # withdrawn (SIGINT while still waiting). Needed because freed slots are re-taken within ~1 s by other tenants' chains
  # while waiting guards poll every 15 s. Never more than one of our tickets holds a slot for longer than a second.
  local name=$1; shift
  local n=${TICKETS:-1}
  if [ "${SLOT_WATCH:-0}" = 1 ]; then MG="$MG" "$PY" src/tools/slot_watch.py --name "$name" -- "$@"; return $?; fi
  if [ "$n" -le 1 ]; then "$MG" --name "$name" -- "$@"; return $?; fi
  local tdir i
  tdir=$(mktemp -d "$ROOT/logs/.tickets.XXXXXX")
  for i in $(seq 1 "$n"); do
    ( "$MG" --name "$name" -- "$PY" src/tools/first.py "$tdir" -- "$@"; echo $? > "$tdir/rc.$i" ) &
    [ "$i" -lt "$n" ] && sleep $(( 15 / n ))
  done
  while [ ! -e "$tdir/started" ] && [ "$(ls "$tdir" | grep -c '^rc\.')" -lt "$n" ]; do sleep 2; done
  if [ -e "$tdir/started" ]; then
    local winner p
    winner=$(cat "$tdir/started")
    for p in $(pgrep -f "first.py $tdir"); do
      [ "$p" != "$winner" ] && ps -o command= -p "$p" | grep -q "memguard.py --name" && kill -INT "$p" 2>/dev/null
    done
  fi
  wait
  local rc=130 r              # 130 if every ticket was interrupted; else 0 unless some ticket's guard failed
  for i in $(seq 1 "$n"); do
    r=$(cat "$tdir/rc.$i" 2>/dev/null || echo 1)
    [ "$r" = 130 ] && continue
    if [ "$r" != 0 ]; then rc=$r; elif [ "$rc" = 130 ]; then rc=0; fi
  done
  rm -rf "$tdir"
  return "$rc"
}

group() {  # group <name> <comma-separated steps/aliases>
  local name=$1 steps=$2
  local expanded
  expanded=$($PY src/run_steps.py --cfg "$CFG" --steps "$steps" --list) || { echo "bad group $name: $steps"; FAILED=1; return 0; }
  local todo="" s
  for s in $expanded; do
    { [ -e "$DONE_DIR/$s" ] && [ "${FORCE:-0}" != 1 ]; } || todo="$todo $s"
  done
  if [ -z "$todo" ]; then
    echo "[$(date '+%F %T')] === group $name: all steps done ($expanded)"; return 0
  fi
  todo=$(echo $todo | tr ' ' ',')
  local attempt rc=1
  for attempt in $(seq 1 "${STEP_RETRIES:-3}"); do
    [ -e "$ROOT/.memguard_stop" ] && { echo "stop file $ROOT/.memguard_stop present; aborting"; exit 3; }
    local t0=$(date +%s)
    echo "[$(date '+%F %T')] >>> group $name: $todo (attempt $attempt)"
    if PIPELINE_FAILED=$FAILED run_guarded "p1-geomark-trajectory-watermark-$TAG-$name" \
         "$PY" src/run_steps.py --cfg "$CFG" --steps "$todo" $([ "${FORCE:-0}" = 1 ] && echo --force); then
      rc=0
    else
      rc=$?
    fi
    echo "[$(date '+%F %T')] <<< group $name rc=$rc $(( $(date +%s) - t0 ))s (incl. slot wait)"
    [ $rc -eq 0 ] && break
    [ $rc -eq 130 ] && { echo "interrupted (SIGINT); stopping"; exit 130; }
    sleep "${RETRY_SLEEP:-120}"
  done
  [ $rc -ne 0 ] && FAILED=1
  return 0
}

llms() {
  $PY -c "
import sys; sys.path.insert(0,'src')
from geomark.common import load_cfg
c = load_cfg('$CFG')
print(' '.join(g for g, v in c['generators'].items() if v['type'] == 'qwen'))"
}

if [ $# -gt 0 ]; then
  for spec in "$@"; do
    case $spec in
      *=*) group "${spec%%=*}" "${spec#*=}" ;;
      *)   group "$spec" "$spec" ;;
    esac
  done
elif [ "$TAG" = smoke ]; then
  group all all                                   # smoke: everything in one guarded job
else
  group prep prep,tests,capacity
  group train-gpt train-gpt
  for q in $(llms); do group "train-$q" "train-$q"; done
  group gen-gpt gen-gpt
  for q in $(llms); do group "gen-$q" "gen-$q"; done
  group attacks baselines-prep,mapmatch,regen-gpt
  for q in $(llms); do group "regen-$q" "regen-$q"; done
  group analysis eval,calib,baselines-eval,utility,steal
  group radio radio,radio-keynull
  group final aggregate,plots
fi
echo "pipeline finished (failed=$FAILED); step times in $OUT/step_times.tsv, compute times in $OUT/compute_times.tsv"
exit $FAILED
