#!/usr/bin/env bash
# Qwen3-VL-8B RESUME — fill only the gaps left when the final run was killed (mid movie101_87 / pre-EC).
#   cold(preproj) + kv(lmcache): only xiaoliyu_9 (4K) was never reached; other 5 videos complete
#                                 (game_33 128f / movie101_87 64-128f are real context-overflow skips).
#   vt(EC): game_33 64f + movie101_87(16/32) + xiaoliyu_9 were missing (EC stopped at game_33 32f).
# Subset CSVs keep dups out. Appends to the SAME results/final/*.csv. H100 only; reaps our EngineCore.
set -u
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf PYTHONHASHSEED=0 PYTHONPATH="$HOME/VLM"
cd "$HOME/VLM" || exit 1

M=qwen3-vl-8b; MML=40960; FR="16 32 64 128"; BA="1 4 8 16"; OUT=results/final

reap() {   # kill OUR lingering EngineCore only (never another user's)
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null); c=$(ps -o cmd= -p "$p" 2>/dev/null)
    if [ "$u" = jylim ] && echo "$c" | grep -q "EngineCore"; then kill -9 "$p" 2>/dev/null && echo "[reap] killed $p"; fi
  done
}
check_gpu() {  # warn if ANOTHER user is on the H100
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null)
    [ -n "$u" ] && [ "$u" != jylim ] && echo "[WARN] GPU in use by '$u' (pid $p) — timing may be contaminated"
  done
}

echo "==================== $M : cold (preproj) — xiaoliyu_9 4K ===================="
check_gpu
python -m measure.preproj_vllm --model "$M" --videos-csv final_videos_resume_4k.csv --frames $FR --batches $BA \
    --max-model-len "$MML" --csv "$OUT/preproj_vllm.csv" 2>"$OUT/log_resume_preproj_qwen3.err"
reap

echo "==================== $M : kv_reuse (LMCache dram) — xiaoliyu_9 4K ===================="
check_gpu
python -m measure.reuse_lmcache --model "$M" --videos-csv final_videos_resume_4k.csv --frames $FR --batches $BA \
    --mode lmcache --tier dram --max-model-len "$MML" --csv "$OUT/reuse_lmcache.csv" 2>"$OUT/log_resume_kv_qwen3.err"
reap

echo "==================== $M : vt EC — game_33 64f (gap) ===================="
check_gpu
python -m measure.reuse_lmcache --model "$M" --videos-csv final_videos_resume_game.csv --frames 64 --batches $BA \
    --mode ec --max-model-len "$MML" --csv "$OUT/reuse_ec.csv" 2>"$OUT/log_resume_ec_game.err"
reap

echo "==================== $M : vt EC — movie101_87 + xiaoliyu_9 ===================="
check_gpu
python -m measure.reuse_lmcache --model "$M" --videos-csv final_videos_resume_ecbig.csv --frames $FR --batches $BA \
    --mode ec --max-model-len "$MML" --csv "$OUT/reuse_ec.csv" 2>"$OUT/log_resume_ec_big.err"
reap

echo "[done] qwen3 resume — gaps filled in $OUT/{preproj_vllm,reuse_lmcache,reuse_ec}.csv"
