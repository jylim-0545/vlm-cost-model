#!/usr/bin/env bash
# InternVL-4B & 14B, 2-pass (cold_vt + kv), NExT-QA, frame[16..128], batch[1,4,8], cudagraph.
# Output appended to results/nextqa/reuse_real.csv. Freeze watchdog + EngineCore reap.
set -u
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf
export OUTPUT_DIR="$HOME/VLM/results/nextqa"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
CSVIN=results/nextqa/sample.csv
FR="16 32 64 128"; BA="1 4 8"
STALL=1200
LOG=/tmp/intern_4b14b_logs; mkdir -p "$LOG"

reap(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qi EngineCore && kill -9 "$pid" 2>/dev/null && echo "  reaped $pid"; done; }

run(){ local m=$1 p=$2; local lf="$LOG/${m}__${p}.log"; : > "$lf"
  python -u -m measure.reuse_real --model "$m" --pass "$p" --videos-csv "$CSVIN" \
      --frames $FR --batches $BA --runs 5 --warmup 2 --cudagraph > "$lf" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do sleep 60
    kill -0 "$pid" 2>/dev/null || break
    local idle=$(( $(date +%s) - $(stat -c %Y "$lf" 2>/dev/null || date +%s) ))
    [ "$idle" -gt "$STALL" ] && { echo "  !! FREEZE ${idle}s — kill $pid"; kill -9 "$pid" 2>/dev/null; break; }
  done; wait "$pid" 2>/dev/null
  echo "  $m/$p done=$(grep -cE '^\[reuse_real\] (cold_vt|kv) ' "$lf") fail=$(grep -cE 'FAILED' "$lf")"; reap; sleep 3; }

echo "######## InternVL 4B/14B start $(date) ########"; reap
for m in internvl3.5-4b internvl3.5-14b; do
  echo "==== $m cold_vt ===="; run "$m" cold_vt
  echo "==== $m kv ====";      run "$m" kv
done
echo "######## done $(date) ########"
echo "CSV models: $(awk -F, 'NR>1{print \$1}' results/nextqa/reuse_real.csv | sort -u | tr '\n' ' ')"
