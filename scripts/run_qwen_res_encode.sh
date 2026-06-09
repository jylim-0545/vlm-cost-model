#!/usr/bin/env bash
# Matched-n_vis encode-vs-resolution experiment for Qwen2.5 & Qwen3.
# Per (model, resolution) we sweep frame counts chosen so n_vis ~ 2500/5000/7500,
# then encode = cold_ttft - vt_inject is compared ACROSS resolutions at matched n_vis.
# If encode-vs-n_vis curves overlay -> n_vis is a sufficient x-axis; if the capped
# (1280x720) curve sits above -> resolution is an independent axis (within-frame attn).
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" HF_HOME=/mnt/nas/VLM/hf HF_HUB_OFFLINE=1
export OUTPUT_DIR="$HOME/VLM/results/qwen_res_encode"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
GID="${CUDA_VISIBLE_DEVICES}"; MML=40960; MNBT=32768
reap(){ for pid in $(nvidia-smi --id=$GID --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qiE "EngineCore|reuse_real" && kill -9 "$pid" 2>/dev/null && echo "  reaped $pid"; done; }

# (model, resolution-tag, frames) — frames picked per model's measured tok/frame
run(){ local model="$1" tag="$2" frames="$3"
  echo "==== $model  $tag  frames=$frames ===="
  python -u -m measure.reuse_real --model "$model" --pass cold_vt \
    --videos-csv "results/qwen_res_encode/vid_${tag}.csv" --frames $frames --batches 1 \
    --runs 5 --warmup 2 --max-model-len $MML --max-num-batched-tokens $MNBT --cudagraph
  echo "  rc=$?"; reap; sleep 3; }

echo "######## qwen res-encode start $(date) on GPU$GID ########"; reap
# Qwen3 (tok/f: 320x240=40, 640x480=150, 1280x720=360)
run qwen3-vl-8b   320x240  "62 124 188"
run qwen3-vl-8b   640x480  "16 34 50"
run qwen3-vl-8b   1280x720 "8 14 20"
# Qwen2.5 (tok/f: 320x240=70, 640x480=195, 1280x720=360)
run qwen2.5-vl-7b 320x240  "36 72 108"
run qwen2.5-vl-7b 640x480  "12 26 38"
run qwen2.5-vl-7b 1280x720 "8 14 20"
echo "######## done $(date) ########"
