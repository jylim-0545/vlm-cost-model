#!/usr/bin/env bash
# Vision-token sharing demos (see sharing/README.md).
#
#   COST demo  — GPU-FREE: encode "once, serve N" + canonical-TokenStore + break-even.
#                Runs anywhere the repo imports (no torch needed for cost.py).
#   GPU demos  — need a GPU + transformers (the study's 4.57 box; commands printed below).
#
# Usage:
#   bash sharing/run_share_demo.sh                 # unit tests + cost demo
#   bash sharing/run_share_demo.sh --hub-encode-ms 17.8   # cost demo w/ measured encode
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== unit checks (GPU-free) ==="
python -m sharing.test_adapters

echo
echo "=== cost demo (GPU-free) ==="
python -m sharing.demo_cost "$@"

cat <<'EOF'

=== GPU demos (run separately) ===
Need a GPU + transformers (the token-sharing study's box: transformers 4.57.6, 4×RTX 4090;
e.g. conda env vlmeval). The cost-model H100/vlmcost box can run them too if SigLIP + OV are
present. Pick a free GPU with CUDA_VISIBLE_DEVICES.

  # recovery vs native — adapter ladder on LLaVA-OV (fine = MMStar / holistic = NExT-QA):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode raw       --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode ridge     --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_recon --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --recon-lambda 8 --ft-steps 600 --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task nextqa --mode raw       --frames 4 --n-eval 200
  # multi-task (trade-off + forgetting):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --multitask aokvqa --recon-lambda 8 --forget aokvqa

  # measured latency (adapter ~1% of ViT; feeds demo_cost --hub-encode-ms):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_latency --backbone llavaov --runs 20

  # full variant × 3-seed sweep over 4 GPUs (LONG):
  python -m sharing.sweep --gpus 0,1,2,3 --seeds 0,1,2 --tasks mmstar,nextqa \
      --modes raw,mlp_recon,mlp_e2e --out-csv logs/share_sweep/recovery.csv
EOF
