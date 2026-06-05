#!/usr/bin/env bash
# LLaVA-OV vt_reuse via mm-cache HIT (encode-skip lower bound, no inject overhead).
# Isolated CSV results/nextqa_llava_ideal/ ; cold & kv re-measured for consistency.
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" HF_HOME=/mnt/nas/VLM/hf
export OUTPUT_DIR="$HOME/VLM/results/nextqa_llava_ideal"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM; mkdir -p "$OUTPUT_DIR"
CSVIN=results/nextqa/sample.csv; FR="16 32 64"; BA="1 4 8"; MML=32768; GID="${CUDA_VISIBLE_DEVICES}"
reap(){ for pid in $(nvidia-smi --id=$GID --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qi EngineCore && kill -9 "$pid" 2>/dev/null && echo "  reaped $pid"; done; }

echo "######## LLaVA mmhit start $(date) on GPU$GID ########"; reap
echo "==== cold_vt (vt-mode=mmhit) ===="
python -u -m measure.reuse_real --model llava-ov-7b --pass cold_vt --vt-mode mmhit \
    --videos-csv "$CSVIN" --frames $FR --batches $BA --runs 5 --warmup 2 --max-model-len $MML --cudagraph
echo "  cold_vt rc=$?"; reap; sleep 3
echo "==== kv ===="
python -u -m measure.reuse_real --model llava-ov-7b --pass kv \
    --videos-csv "$CSVIN" --frames $FR --batches $BA --runs 5 --warmup 2 --max-model-len $MML --cudagraph
echo "  kv rc=$?"; reap; sleep 3
echo "==== figures (dataset=nextqa_llava_ideal, frame=64) ===="
python analyze/fig_internvl8b.py --model llava-ov-7b --dataset nextqa_llava_ideal --frame 64 \
    2>&1 | grep -E "^\[fig\]|Fig8|baseline/|kv/|vt/|saved"
echo "######## LLaVA mmhit done $(date) ########"
