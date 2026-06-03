#!/usr/bin/env bash
# Full NExT-QA experiment driver (CLAUDE.md Sections 5/7).
# Stage A (GPU): per-model Layer-1 primitives on REAL videos via vLLM (batch=1,
# prefix caching OFF, --text-baseline so encode/prefill/decode split). Stage B
# (no GPU): break-even family + figures. Each model runs as a FRESH process (no
# VRAM leak); a model that fails is logged and skipped so the rest proceed.
#
# Usage:
#   bash scripts/run_nextqa.sh                 # all models, 16-video sample, 16 frames
#   NSAMPLE=40 FRAMES=32 bash scripts/run_nextqa.sh
#   MODELS="qwen2.5-vl-7b qwen3-vl-8b" bash scripts/run_nextqa.sh
#
# NOTE: vLLM video path is verified for Qwen2.5-VL / Qwen3-VL. InternVL3.5 has no
# HF processor (apply_chat_template fails), so its vLLM-video run will likely fail
# here and be skipped — measure InternVL on real video via the transformers path
# once that's added (TODO). byte_sizes / break-even for InternVL still work.
set -uo pipefail

source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=/mnt/nas/VLM/hf
export HF_HUB_OFFLINE=1
export OUTPUT_DIR="$HOME/VLM/results"

SAMPLE="$OUTPUT_DIR/nextqa_sample.csv"
PRIMS="$OUTPUT_DIR/stage_timing_vllm.csv"
FRAMES="${FRAMES:-16}"
NSAMPLE="${NSAMPLE:-16}"
RUNS="${RUNS:-5}"
MODELS="${MODELS:-qwen2.5-vl-7b qwen3-vl-8b internvl3.5-8b internvl3.5-14b}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "[run] sample=$SAMPLE frames=$FRAMES models=($MODELS)"

# 1. prepare a short-end video sample if missing (extracts to LOCAL_SCRATCH)
if [ ! -f "$SAMPLE" ]; then
  echo "[run] preparing NExT-QA sample (n=$NSAMPLE) ..."
  python -m data.prepare_nextqa --n "$NSAMPLE"
fi

# 2. fresh primitives file for this run (keep prior as .bak)
[ -f "$PRIMS" ] && mv "$PRIMS" "$PRIMS.bak.$STAMP" && echo "[run] archived old primitives -> $PRIMS.bak.$STAMP"

# 3. Stage A — Layer-1 primitives per model (real videos), each a fresh process
for m in $MODELS; do
  echo "=== [run] stage_timing_vllm: $m ==="
  if python -m measure.stage_timing_vllm --model "$m" --videos-csv "$SAMPLE" \
        --frames "$FRAMES" --text-baseline --warmup 2 --runs "$RUNS"; then
    echo "[run] OK: $m"
  else
    echo "[run] FAILED: $m (skipped — see error above)"
  fi
done

# 4. Stage B — break-even family + figures (no GPU; re-runnable after editing prices)
if [ -f "$PRIMS" ]; then
  echo "=== [run] analyze.plots ==="
  python -m analyze.plots --primitives "$PRIMS"
  echo "[run] DONE -> $OUTPUT_DIR/figures/  (edit config/prices.yaml or storage_tiers.yaml and re-run 'python -m analyze.plots --primitives $PRIMS' anytime — no GPU)"
else
  echo "[run] no primitives produced; nothing to analyze."
fi
