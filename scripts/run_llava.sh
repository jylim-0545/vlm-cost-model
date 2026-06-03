#!/usr/bin/env bash
# LLaVA-OV-7B, 2-pass (cold_vt + kv), NExT-QA, frame[16..128], batch[1,4,8].
# 196 tok/frame fixed (128f=25088 < 32768 ctx). dims == Qwen2.5 (KV 56KB, ratio 8x).
# Appends to results/nextqa/reuse_real.csv. H100(GPU1) by default — timing must be
# H100-normalized. For a Blackwell(GPU0) PRELIM only: CUDA_VISIBLE_DEVICES=0 ALLOW_GPU0=1.
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" HF_HOME=/mnt/nas/VLM/hf
export OUTPUT_DIR="$HOME/VLM/results/nextqa"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
CSVIN=results/nextqa/sample.csv
FR="16 32 64 128"; BA="1 4 8"; STALL=1200; MML=32768
GID="${CUDA_VISIBLE_DEVICES}"
LOG=/tmp/llava_logs; mkdir -p "$LOG"

reap(){ for pid in $(nvidia-smi --id=$GID --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qi EngineCore && kill -9 "$pid" 2>/dev/null && echo "  reaped $pid"; done; }

run(){ local p=$1; local lf="$LOG/llava__${p}.log"; : > "$lf"
  python -u -m measure.reuse_real --model llava-ov-7b --pass "$p" --videos-csv "$CSVIN" \
      --frames $FR --batches $BA --runs 5 --warmup 2 --max-model-len $MML --cudagraph > "$lf" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do sleep 60
    kill -0 "$pid" 2>/dev/null || break
    local idle=$(( $(date +%s) - $(stat -c %Y "$lf" 2>/dev/null || date +%s) ))
    [ "$idle" -gt "$STALL" ] && { echo "  !! FREEZE ${idle}s — kill $pid"; kill -9 "$pid" 2>/dev/null; break; }
  done; wait "$pid" 2>/dev/null
  echo "  llava/$p done=$(grep -cE '^\[reuse_real\] (cold_vt|kv) ' "$lf") fail=$(grep -cE 'FAILED' "$lf")"; reap; sleep 3; }

echo "######## LLaVA-OV start $(date) on GPU$GID ########"; reap
echo "==== cold_vt ===="; run cold_vt
echo "==== kv ====";      run kv
echo "######## done $(date) ########"
