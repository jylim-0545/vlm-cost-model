#!/usr/bin/env bash
# Qwen2.5 single-video (5396384503) full sweep -> completes the batch/figure video (legacy
# multi-video data lacked 128f on 5396384503). No longest_edge (Qwen2.5 = per-frame cap, no saturation).
set -u
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf HF_HUB_OFFLINE=1
export OUTPUT_DIR="$HOME/VLM/results/nextqa"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
FR="16 32 64 128"; BA="1 4 8 16"; MNBT=32768
reap(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p $pid -o cmd= 2>/dev/null); echo "$c"|grep -qiE "EngineCore|reuse_real" && kill -9 $pid 2>/dev/null && echo "  reaped $pid"; done; }
echo "######## Qwen2.5 single-video start $(date) ########"; reap
echo "==== cold_vt ===="
python -u -m measure.reuse_real --model qwen2.5-vl-7b --pass cold_vt \
  --videos-csv results/nextqa/sample_1vid.csv --frames $FR --batches $BA --runs 5 --warmup 2 --max-num-batched-tokens $MNBT --cudagraph
echo "  cold_vt rc=$?"; reap; sleep 3
echo "==== kv ===="
python -u -m measure.reuse_real --model qwen2.5-vl-7b --pass kv \
  --videos-csv results/nextqa/sample_1vid.csv --frames $FR --batches $BA --runs 5 --warmup 2 --max-num-batched-tokens $MNBT --cudagraph
echo "  kv rc=$?"; reap
echo "######## Qwen2.5 single-video done $(date) ########"
