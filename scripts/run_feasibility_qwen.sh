#!/usr/bin/env bash
# Qwen2.5 & Qwen3 feasibility frontier: max feasible n_vis per (model, batch) on H100.
# One subprocess per (model, batch) so an OOM (EngineDead) is isolated. Appends results/feasibility.csv.
# n_vis ceiling is resolution-independent (verified), so this gives max_n_vis; per-resolution
# max_frames = max_n_vis / tok_per_frame(res). Uses the capped 1280x720 video (~360 tok/frame).
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" HF_HOME=/mnt/nas/VLM/hf HF_HUB_OFFLINE=1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
GID="${CUDA_VISIBLE_DEVICES}"; PER_CFG_TIMEOUT=1200
reap(){ for pid in $(nvidia-smi --id=$GID --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qiE "EngineCore|feasibility" && kill -9 "$pid" 2>/dev/null && echo "  reaped $pid"; done; }

echo "######## qwen feasibility start $(date) on GPU$GID ########"; reap
for model in qwen3-vl-8b qwen2.5-vl-7b; do
  for b in 1 4 8 16 32; do
    echo "==== $model b$b ===="
    timeout $PER_CFG_TIMEOUT python -u -m measure.feasibility "$model" "$b"
    rc=$?; [ $rc -eq 124 ] && echo "  TIMEOUT ($model b$b) after ${PER_CFG_TIMEOUT}s"
    echo "  rc=$rc"; reap; sleep 3
  done
done
echo "######## done $(date) ########"
