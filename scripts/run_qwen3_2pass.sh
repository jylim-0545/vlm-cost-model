#!/usr/bin/env bash
# Qwen3-VL two-pass batch sweep — MIRRORS run_qwen25_2pass.sh (same frames/batches/CSV) so the
# 6 models are directly comparable. Adds --qwen3-longest-edge (default 154M, auto) to lift the
# 12288-token video-budget saturation -> n_vis ∝ frames like the others; --max-num-batched-tokens
# 32768 for the encoder budget at 128f (raised n_vis can exceed the 16384 default).
set -u
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf
export OUTPUT_DIR="$HOME/VLM/results/nextqa"   # unified 6-model CSV
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
FR="16 32 64 128"; BA="1 4 8 16"; MNBT=32768
reap(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p $pid -o cmd= 2>/dev/null); echo "$c"|grep -qiE "EngineCore|reuse_real" && kill -9 $pid 2>/dev/null && echo "  reaped $pid"; done; }

echo "######## Qwen3 2-pass start $(date) ########"
reap
echo "==== PASS cold_vt (prefix OFF) ===="
python -u -m measure.reuse_real --model qwen3-vl-8b --pass cold_vt \
    --videos-csv results/nextqa/sample.csv --frames $FR --batches $BA --runs 5 --warmup 2 --max-num-batched-tokens $MNBT --cudagraph
echo "  cold_vt rc=$?"; reap; sleep 3
echo "==== PASS kv (prefix ON) ===="
python -u -m measure.reuse_real --model qwen3-vl-8b --pass kv \
    --videos-csv results/nextqa/sample.csv --frames $FR --batches $BA --runs 5 --warmup 2 --max-num-batched-tokens $MNBT --cudagraph
echo "  kv rc=$?"; reap
echo "######## Qwen3 2-pass done $(date) ########"
