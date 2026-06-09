#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/nas/VLM/hf
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost
cd ~/VLM
reap(){ for pid in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  c=$(ps -p "$pid" -o cmd= 2>/dev/null); echo "$c"|grep -qi EngineCore && kill -9 "$pid" 2>/dev/null; done; }
echo "######## feasibility sweep $(date) ########"
for m in qwen2.5-vl-7b internvl3.5-4b internvl3.5-8b internvl3.5-14b llava-ov-7b qwen3-vl-8b; do
  for b in 1 4 8 16 32; do
    echo "==== $m b$b ===="
    timeout 700 python -u -m measure.feasibility "$m" "$b" 2>&1 | grep -E "^\[feas\]"
    reap; sleep 2
  done
done
echo "######## done $(date) ########"
echo "=== SUMMARY: max frame per (model,batch) ==="
awk -F, 'NR>1 && $5=="OK"{k=$1" b"$2; if($3>m[k])m[k]=$3} END{for(x in m)print x, m[x]"f"}' results/feasibility.csv | sort
