#!/usr/bin/env bash
# Full sweep, unattended. LAYER 1 (reuse_real) for all (model x dataset), THEN
# LAYER 2 (throughput batch sweep). Each run is an INDEPENDENT process so an
# OOM/engine crash skips only that combo (rows flush per-config). Orphan
# VLLM::EngineCore is reaped after every process (pkill doesn't reap it).
#
# Frame sweep (CLAUDE.md): 16 32 64 128 for BOTH datasets, ALL models.
#   (MLVU is 21-49min; 256 frames * 256 tok = 65536 > 40960 ctx for InternVL, and the
#    user dropped 256 for Qwen too -> 128 is the cap everywhere.)
#
# NO hard time cap: a single MLVU config (32768-tok prefill + decode) is legitimately
# slow. Instead a FREEZE WATCHDOG kills a process only if its log is IDLE for
# STALL_LIMIT seconds (real hang: CUDA deadlock / NFS stall), never a busy one.
#
# Usage:  nohup bash scripts/run_full.sh > /tmp/run_full.log 2>&1 &
set -u
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/mnt/nas/VLM/hf
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM

MODELS=(qwen2.5-vl-7b qwen3-vl-8b internvl3.5-4b internvl3.5-8b internvl3.5-14b)
NEXTQA=results/nextqa_sample.csv
MLVU=results/mlvu_sample.csv
FRAMES="16 32 64 128"
RUNS=5
WARMUP=2
STALL_LIMIT=900           # kill ONLY if log idle > 15min (freeze), not just slow
TPUT_FRAMES=32            # representative input for the batch sweep
TPUT_BATCHES="1 4 8 16"
LOGDIR=/tmp/run_full_logs
mkdir -p "$LOGDIR"

kill_engine() {           # reap orphan VLLM::EngineCore (ours), leave other users' procs
  for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    cmd=$(ps -p "$pid" -o cmd= 2>/dev/null)
    echo "$cmd" | grep -qi 'EngineCore' && kill -9 "$pid" 2>/dev/null && echo "  [reaped EngineCore $pid]"
  done
}

# launch a module, watchdog its log: kill only on freeze (log idle > STALL_LIMIT).
# args: logfile, python-module-args...
run_watched() {
  local log=$1; shift
  : > "$log"
  python -u "$@" > "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    kill -0 "$pid" 2>/dev/null || break
    local idle=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ))
    if [ "$idle" -gt "$STALL_LIMIT" ]; then
      echo "  !! FREEZE: log idle ${idle}s > ${STALL_LIMIT}s — killing pid $pid"
      kill -9 "$pid" 2>/dev/null; break
    fi
  done
  wait "$pid" 2>/dev/null
  return $?
}

echo "######## FULL RUN start $(date) ########"
kill_engine

echo "==== LAYER 1: reuse_real (5 models x 2 datasets) ===="
for m in "${MODELS[@]}"; do
  for ds in "nextqa:$NEXTQA" "mlvu:$MLVU"; do
    name=${ds%%:*}; csv=${ds##*:}
    log="$LOGDIR/L1_${m}__${name}.log"
    echo "---- $(date +%H:%M:%S)  L1 $m x $name  frames=[$FRAMES] ----"
    run_watched "$log" -m measure.reuse_real --model "$m" --videos-csv "$csv" \
        --frames $FRAMES --runs "$RUNS" --warmup "$WARMUP"
    echo "     rc=$?  configs_done=$(grep -cE '^\[reuse_real\] [0-9]' "$log")  (log: $log)"
    kill_engine; sleep 3
  done
done
echo "==== LAYER 1 done. rows in reuse_real.csv: $(wc -l < results/reuse_real.csv 2>/dev/null) ===="

echo "==== LAYER 2: throughput batch sweep (5 models, NExT-QA rep video) ===="
for m in "${MODELS[@]}"; do
  log="$LOGDIR/L2_${m}.log"
  echo "---- $(date +%H:%M:%S)  L2 $m  frames=$TPUT_FRAMES batches=[$TPUT_BATCHES] ----"
  run_watched "$log" -m measure.throughput --model "$m" --videos-csv "$NEXTQA" \
      --frames "$TPUT_FRAMES" --batches $TPUT_BATCHES
  echo "     rc=$?  (log: $log)"
  kill_engine; sleep 3
done

echo "######## FULL RUN done $(date) ########"
echo "L1 rows: $(wc -l < results/reuse_real.csv 2>/dev/null)   L2 rows: $(wc -l < results/throughput.csv 2>/dev/null)"
