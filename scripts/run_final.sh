#!/usr/bin/env bash
# Final paper experiment — 3-way (cold / vt_reuse / kv_reuse) x 4 models x 6 videos x 4 frames x 4 batches.
#   cold + vt_pre/vt_post  <- measure/preproj_vllm.py        (in-process; cold-only for Qwen3)
#   vt_reuse (Qwen3)       <- measure/reuse_lmcache.py --mode ec   --tier dram  (EC post-projector)
#   kv_reuse (all 4)       <- measure/reuse_lmcache.py --mode lmcache --tier dram
# Measurement tier = DRAM (real DRAM->GPU load); S3/local storage->DRAM is COMPUTED in TCO (CLAUDE.md §7).
# H100 ONLY. Reaps lingering jylim VLLM::EngineCore between vLLM (multiproc) runs. OOM/overflow configs
# are caught & skipped inside each script. Run in background; logs in results/final/.
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf PYTHONHASHSEED=0 PYTHONPATH="$HOME/VLM"
cd "$HOME/VLM" || exit 1

VIDS=final_videos.csv
FR="16 32 64 128"
BA="1 4 8 16"
OUT=results/final
mkdir -p "$OUT"

reap() {   # kill OUR lingering EngineCore only (never another user's)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null); c=$(ps -o cmd= -p "$p" 2>/dev/null)
    if [ "$u" = jylim ] && echo "$c" | grep -q "EngineCore"; then kill -9 "$p" 2>/dev/null && echo "[reap] killed $p"; fi
  done
}
check_gpu() {  # warn if ANOTHER user is on the H100 (timing contamination)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null)
    [ -n "$u" ] && [ "$u" != jylim ] && echo "[WARN] GPU in use by '$u' (pid $p) — timing may be contaminated"
  done
}

MODELS="internvl3.5-8b llava-ov-7b qwen2.5-vl-7b qwen3-vl-8b"

for M in $MODELS; do
  MML=40960; [ "$M" = "llava-ov-7b" ] && MML=32768
  check_gpu
  echo "==================== $M : cold + vt (preproj_vllm) ===================="
  python -m measure.preproj_vllm --model "$M" --videos-csv "$VIDS" --frames $FR --batches $BA \
      --max-model-len "$MML" --csv "$OUT/preproj_vllm.csv" 2>"$OUT/log_preproj_${M}.err"
  reap
  echo "==================== $M : kv_reuse (LMCache, dram) ===================="
  python -m measure.reuse_lmcache --model "$M" --videos-csv "$VIDS" --frames $FR --batches $BA \
      --mode lmcache --tier dram --max-model-len "$MML" --csv "$OUT/reuse_lmcache.csv" 2>"$OUT/log_kv_${M}.err"
  reap
done

echo "==================== qwen3-vl-8b : vt_reuse via EC (post-projector) ===================="
check_gpu
python -m measure.reuse_lmcache --model qwen3-vl-8b --videos-csv "$VIDS" --frames $FR --batches $BA \
    --mode ec --tier dram --csv "$OUT/reuse_ec.csv" 2>"$OUT/log_ec_qwen3.err"
reap

echo "[done] results/final/  (cold/vt in preproj_vllm.csv, kv in reuse_lmcache.csv, qwen3 vt in reuse_ec.csv)"
