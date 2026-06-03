#!/usr/bin/env bash
# Qwen2.5-VL two-pass batch sweep (same-video x B). cold_vt (prefix OFF) then kv (prefix ON).
set -u
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
FR="16 32 64 128"; BA="1 4 8 16"
reap(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p $pid -o cmd= 2>/dev/null); echo "$c"|grep -qi EngineCore && kill -9 $pid 2>/dev/null && echo "  reaped $pid"; done; }

echo "######## Qwen2.5 2-pass start $(date) ########"
reap
echo "==== PASS cold_vt (prefix OFF) ===="
python -u -m measure.reuse_real --model qwen2.5-vl-7b --pass cold_vt \
    --frames $FR --batches $BA --runs 5 --warmup 2 --cudagraph
echo "  cold_vt rc=$?"; reap; sleep 3
echo "==== PASS kv (prefix ON) ===="
python -u -m measure.reuse_real --model qwen2.5-vl-7b --pass kv \
    --frames $FR --batches $BA --runs 5 --warmup 2 --cudagraph
echo "  kv rc=$?"; reap
echo "######## Qwen2.5 2-pass done $(date) ########"
