#!/usr/bin/env bash
# Vision-token pruning demos (see pruning/README.md).
#
#   COST demo  — GPU-FREE, runs in the vlmcost env (or anywhere the repo imports).
#                Shows storage + break-even N* shrinking as we prune.
#   GPU demos  — need a GPU (transformers 4.57 or 5.9; the repo's vlmcost env works).
#                Commands printed below.
#
# Usage:
#   scripts/run_pruning_demo.sh                       # cost demo, representative base
#   scripts/run_pruning_demo.sh path/to/reuse_real.csv  # cost demo off real measurements
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_CSV="${1:-}"
MODEL="${MODEL:-internvl3.5-8b}"

echo "=== unit checks (GPU-free) ==="
python -m pruning.test_methods

echo
echo "=== cost demo (GPU-free) ==="
if [[ -n "$BASE_CSV" ]]; then
  python -m pruning.demo_cost --model "$MODEL" --base-csv "$BASE_CSV"
else
  python -m pruning.demo_cost --model "$MODEL"
fi

cat <<'EOF'

=== GPU demos (run separately) ===
Need a GPU (transformers 4.57 or 5.9; this repo's vlmcost env works).
On the transformers-4.57 box:
  conda activate <efficientvlm-env>        # transformers==4.57.6
  cd /path/to/vlm-cost-model
  # real latency breakdown (encode vs prefill, video-scale n_vis):
  CUDA_VISIBLE_DEVICES=<gpu> python -m pruning.demo_latency  --which internvl --n-vis 8192
  # accuracy sanity (does pruning change accuracy?):
  CUDA_VISIBLE_DEVICES=<gpu> python -m pruning.demo_accuracy --which internvl \
      --tsv <textvqa.tsv> --bench textvqa --n 30 --keeps 1.0,0.5,0.25,0.1
EOF
