#!/usr/bin/env bash
# AUTOPILOT (user left): wait for the running Qwen3 sweep -> Qwen2.5 feasibility -> figures +
# break-even. Best-effort analysis (|| true) so one failure doesn't abort. Logs to stdout.
set -u
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf HF_HUB_OFFLINE=1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
CSV=results/nextqa/reuse_real.csv
reap(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qiE "EngineCore|reuse_real|measure.feasibility" \
    && kill -9 "$pid" 2>/dev/null && echo "  [autopilot] reaped $pid"; done; }

echo "######## AUTOPILOT start $(date) ########"

# ---- 1. wait for the Qwen3 sweep to finish (done-marker OR sweep process gone); max ~3h ----
echo "[autopilot] waiting for Qwen3 sweep to finish..."
for i in $(seq 1 360); do
  grep -q "Qwen3 2-pass done" /tmp/qwen3_2pass.log 2>/dev/null && { echo "[autopilot] sweep done-marker seen"; break; }
  pgrep -f run_qwen3_2pass >/dev/null 2>&1 || { echo "[autopilot] sweep process gone (i=$i)"; break; }
  sleep 30
done
echo "[autopilot] Qwen3 sweep wait over. cold_vt configs=$(grep -c 'cold_vt .*n_vis=' /tmp/qwen3_2pass.log 2>/dev/null), kv configs=$(grep -c 'kv .*n_vis=\|kv_reuse' /tmp/qwen3_2pass.log 2>/dev/null)"
sleep 5; reap; sleep 5

# ---- 2. Qwen2.5 feasibility (b1/4/8/16/32). non-saturating per-frame cap -> n_vis ∝ frames ----
echo "[autopilot] === Qwen2.5 feasibility ==="
for b in 1 4 8 16 32; do
  echo "[autopilot] feasibility qwen2.5-vl-7b b$b"
  timeout 1500 python -u -m measure.feasibility qwen2.5-vl-7b "$b"
  rc=$?; [ $rc -eq 124 ] && echo "[autopilot]   TIMEOUT b$b"
  reap; sleep 3
done

# ---- 3. figures (6 models, best-effort) ----
echo "[autopilot] === figures ==="
for m in internvl3.5-4b internvl3.5-8b internvl3.5-14b llava-ov-7b qwen2.5-vl-7b qwen3-vl-8b; do
  for fr in 64 128; do
    python analyze/fig_internvl8b.py --model "$m" --frame "$fr" --dataset nextqa >/dev/null 2>&1 \
      && echo "[autopilot]   fig OK $m f$fr" || echo "[autopilot]   fig SKIP $m f$fr"
  done
done

# ---- 4. break-even (all models; gpu-stall on & off), best-effort ----
echo "[autopilot] === break-even ==="
python analyze/breakeven_reuse.py --csv "$CSV" > /tmp/breakeven_all.txt 2>&1 \
  && echo "[autopilot]   breakeven OK -> /tmp/breakeven_all.txt" || echo "[autopilot]   breakeven FAIL (see /tmp/breakeven_all.txt)"
python analyze/breakeven_reuse.py --csv "$CSV" --no-gpu-stall > /tmp/breakeven_nostall.txt 2>&1 \
  && echo "[autopilot]   breakeven(no-stall) OK -> /tmp/breakeven_nostall.txt" || echo "[autopilot]   breakeven(no-stall) FAIL"

echo "######## AUTOPILOT done $(date) ########"
