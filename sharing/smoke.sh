#!/usr/bin/env bash
# Tiny end-to-end smoke for ALL sharing/ paths BEFORE a full run (analog of
# EfficientVLM/scripts/smoke_B.sh). GPU-free checks first, then 4 tiny GPU jobs in parallel.
# These are crash/plumbing checks — accuracy is meaningless at this scale (use sweep.py for
# real recovery numbers).
set -uo pipefail
cd "$(dirname "$0")/.."
L=logs/smoke_share
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU-free: unit tests + cost demo ==="
python -m sharing.test_adapters
python -m sharing.demo_cost >/dev/null && echo "  demo_cost OK"

echo
echo "=== GPU: 4 tiny trainer smokes (one per GPU) ==="
# fine: raw (no train) / mlp_recon (pretrain) / mlp_e2e (pretrain+ft+anchor+cosine)
CUDA_VISIBLE_DEVICES=${G0:-0} python -u -m sharing.demo_train --task mmstar --mode raw \
  --pre-samples 24 --n-eval 12 > "$L/raw_mmstar.log" 2>&1 &
CUDA_VISIBLE_DEVICES=${G1:-1} python -u -m sharing.demo_train --task mmstar --mode mlp_recon \
  --pre-samples 24 --pre-steps 40 --n-eval 12 > "$L/recon_mmstar.log" 2>&1 &
CUDA_VISIBLE_DEVICES=${G2:-2} python -u -m sharing.demo_train --task mmstar --mode mlp_e2e \
  --pre-samples 24 --pre-steps 40 --ft-steps 16 --recon-lambda 8 --n-eval 12 > "$L/e2e_mmstar.log" 2>&1 &
# holistic video loader (decord)
CUDA_VISIBLE_DEVICES=${G3:-3} python -u -m sharing.demo_train --task nextqa --mode raw --frames 4 \
  --pre-samples 8 --n-eval 6 > "$L/raw_nextqa.log" 2>&1 &
wait

echo
echo "=== smoke results ([result] line per job) ==="
for f in raw_mmstar recon_mmstar e2e_mmstar raw_nextqa; do
  line=$(grep -h "\[result\]" "$L/$f.log" 2>/dev/null | tail -1)
  echo "  $f: ${line:-FAILED (see $L/$f.log)}"
done
