#!/usr/bin/env bash
# AUTO-PILOT (overnight): wait for GPU1 InternVL to finish -> make InternVL 4B/14B figures
# -> wait for H100(GPU1) idle (shared w/ ljh) -> run LLaVA-OV on H100 -> make LLaVA figures.
# GPU0 LLaVA Blackwell prelim runs independently (its CSV is isolated) and is NOT touched.
set -u
export HF_HOME=/mnt/nas/VLM/hf OUTPUT_DIR="$HOME/VLM/results/nextqa"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

reap1(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qi EngineCore && kill -9 "$pid" 2>/dev/null && log "  reaped EngineCore $pid"; done; }

echo "######## AUTO-PILOT start $(date) ########"

# 1) wait for our InternVL run on GPU1 to finish (4B already done; 14B in progress)
log "waiting for GPU1 InternVL (reuse_real --model internvl ...) to finish..."
while pgrep -f "reuse_real --model internvl" >/dev/null 2>&1; do sleep 120; done
log "InternVL finished."

# 2) InternVL 4B/14B figures (frame=128; skip gracefully if data missing)
for m in internvl3.5-4b internvl3.5-14b; do
  log "figures for $m ..."
  python analyze/fig_internvl8b.py --model "$m" --frame 128 >/tmp/fig_${m}.log 2>&1 \
    && log "  $m figures OK" || log "  $m figures FAILED (see /tmp/fig_${m}.log)"
done

# 3) reap our orphan EngineCore, then wait until H100(GPU1) is genuinely free (ljh shares it)
reap1; sleep 5
log "waiting for GPU1 to be free (<10GB used; ljh may share)..."
while true; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$used" ] && [ "$used" -lt 10000 ] && { log "GPU1 free (${used}MiB)."; break; }
  log "  GPU1 busy (${used}MiB) — waiting 5min..."
  sleep 300
done

# 4) run LLaVA-OV on H100(GPU1) — real (H100-normalized) numbers into results/nextqa/
log "launching run_llava.sh on H100(GPU1)..."
CUDA_VISIBLE_DEVICES=1 bash scripts/run_llava.sh
log "run_llava.sh returned (rc=$?)."
reap1; sleep 3

# 5) LLaVA figures (196 tok/frame fixed -> 128f fits 32768 ctx)
log "figures for llava-ov-7b ..."
python analyze/fig_internvl8b.py --model llava-ov-7b --frame 128 >/tmp/fig_llava.log 2>&1 \
  && log "  LLaVA figures OK" || log "  LLaVA figures FAILED (see /tmp/fig_llava.log)"

echo "######## AUTO-PILOT done $(date) ########"
log "CSV models: $(awk -F, 'NR>1{print \$1}' results/nextqa/reuse_real.csv | sort -u | tr '\n' ' ')"
log "LLaVA rows: $(awk -F, '$1=="llava-ov-7b"' results/nextqa/reuse_real.csv | wc -l)"
