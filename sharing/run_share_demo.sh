#!/usr/bin/env bash
# Vision-token sharing demos (see sharing/README.md).
#   - unit checks: GPU-free (only torch).
#   - GPU demos: need a GPU + transformers (the study's box; commands printed below).
#
# Usage:  bash sharing/run_share_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== unit checks (GPU-free) ==="
python -m sharing.test_adapters

cat <<'EOF'

=== GPU demos (run separately) ===
Need a GPU + transformers (the token-sharing study's box: transformers 4.57.6, 4×RTX 4090;
e.g. conda env vlmeval). Pick a free GPU with CUDA_VISIBLE_DEVICES.

  # recovery vs native — adapter ladder on LLaVA-OV (fine = MMStar / holistic = NExT-QA):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode raw       --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode ridge     --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_recon --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --recon-lambda 8 --ft-steps 600 --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task nextqa --mode raw       --frames 4 --n-eval 200
  # multi-task (trade-off + forgetting):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --multitask aokvqa --recon-lambda 8 --forget aokvqa
  # re-evaluate a shipped pre-trained adapter (no training; run `git lfs pull` first):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --n-eval 400 \
      --load-adapter sharing/adapters_pretrained/fine_e2e_s0.pt
  # how cheap is the adapter vs the encoder:
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_latency --backbone llavaov --runs 20

  # full variant × 3-seed sweep over 4 GPUs (LONG):
  python -m sharing.sweep --gpus 0,1,2,3 --seeds 0,1,2 --tasks mmstar,nextqa \
      --modes raw,mlp_recon,mlp_e2e --out-csv logs/share_sweep/recovery.csv
EOF
