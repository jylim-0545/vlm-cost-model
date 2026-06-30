"""Train ONE hub->backbone adapter and report recovery vs native (REAL model, GPU).

Reproduces the study's recovery ladder on LLaVA-OV (vtpatch): pick a task (mmstar fine /
nextqa holistic) and a mode (raw / ridge / mlp_recon / mlp_e2e), optionally mix a second
task for multi-task training, and print adapter% / native% / recovery (= adapter/native).

  # GPU env (the study's transformers-4.57 box / 4×4090):
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode ridge --n-eval 100
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e \
       --recon-lambda 8 --ft-steps 600 --n-eval 400
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task nextqa --mode mlp_recon --frames 4
  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e \
       --multitask aokvqa --recon-lambda 8         # multi-task trade-off + forgetting

Expected (FINDINGS.md, recovery = adapter/native): holistic raw≈0.96; fine raw≈0.83 ->
mlp_recon≈0.88 -> mlp_e2e≈0.91. Absolute numbers are noisy at small n_eval (report uses the
recovery RATIO). See sharing/sweep.py for the 3-seed × variant sweep.
"""
from __future__ import annotations

import argparse

# default data paths (the EfficientVLM checkout this repo is nested under; override via args)
MMSTAR_TSV = "/home/yhlee/EfficientVLM/LMUData/MMStar.tsv"
NEXTQA_CSV = "/home/yhlee/EfficientVLM/data/nextqa_local_mc.csv"
NEXTQA_DIR = "/mnt/nas/yhlee/nextqa/NExTVideo"
AOKVQA_GLOB = ("/home/yhlee/EfficientVLM/hf_cache/hub/datasets--HuggingFaceM4--A-OKVQA/"
               "snapshots/*/data/train-*.parquet")


def _make_task(name, args, n_train=None):
    from sharing import train
    if name == "mmstar":
        return train.mmstar_task(args.mmstar_tsv, n_eval=args.n_eval, seed=args.seed, n_train=n_train)
    if name == "nextqa":
        return train.nextqa_task(args.nextqa_csv, args.nextqa_dir, n_frames=args.frames,
                                 n_eval=args.n_eval, seed=args.seed, n_train=n_train)
    if name == "aokvqa":
        return train.aokvqa_task(args.aokvqa_glob, n_eval=min(args.n_eval, 300), seed=args.seed,
                                 n_train=n_train)
    raise ValueError(f"unknown task '{name}'")


def main() -> None:
    from sharing.methods import HubShare
    from sharing.train import AdapterTrainer, ShareTrainConfig

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="llavaov")
    ap.add_argument("--task", default="mmstar", choices=["mmstar", "nextqa", "aokvqa"])
    ap.add_argument("--mode", default="ridge", choices=["raw", "ridge", "mlp_recon", "mlp_e2e"])
    ap.add_argument("--multitask", default=None, choices=[None, "aokvqa", "mmstar", "nextqa"])
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--recon-lambda", type=float, default=0.0)
    ap.add_argument("--ft-steps", type=int, default=600)
    ap.add_argument("--ft-lr", type=float, default=3e-5)
    ap.add_argument("--pre-steps", type=int, default=2000)
    ap.add_argument("--pre-samples", type=int, default=400)
    ap.add_argument("--sched", default="cosine", choices=["const", "cosine"])
    ap.add_argument("--frames", type=int, default=4, help="frames per video (nextqa)")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=None, help="cap train items (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forget", default=None, choices=[None, "aokvqa", "mmstar"],
                    help="also eval this OTHER task (forgetting)")
    ap.add_argument("--load-adapter", default=None,
                    help="eval a PRE-TRAINED study adapter (.pt: {state_dict,xm,xs,adapter,MLP_H}) "
                         "instead of training — e.g. EfficientVLM/logs/d6_B_fine_recon_s0_adapter.pt")
    ap.add_argument("--mmstar-tsv", default=MMSTAR_TSV)
    ap.add_argument("--nextqa-csv", default=NEXTQA_CSV)
    ap.add_argument("--nextqa-dir", default=NEXTQA_DIR)
    ap.add_argument("--aokvqa-glob", default=AOKVQA_GLOB)
    a = ap.parse_args()

    cfg = ShareTrainConfig(mode=a.mode, hidden=a.hidden, recon_lambda=a.recon_lambda,
                           ft_steps=a.ft_steps, ft_lr=a.ft_lr, pre_steps=a.pre_steps,
                           pre_samples=a.pre_samples, sched=a.sched, n_eval=a.n_eval, seed=a.seed)
    share = HubShare(a.backbone)
    task = _make_task(a.task, a, n_train=a.n_train)
    tr = AdapterTrainer(share, cfg)
    if a.load_adapter:
        from sharing.adapters import load_study_adapter
        ad, kind = load_study_adapter(a.load_adapter)
        print(f"[load] {a.load_adapter}  (kind={kind})", flush=True)
        res = tr.evaluate(ad, task)
    else:
        mt = _make_task(a.multitask, a, n_train=a.n_train) if a.multitask else None
        res = tr.run(task, multitask=mt)
    if a.forget:
        f = tr.forgetting(_make_task(a.forget, a))
        print(f"  forgetting on {f['task']}: adapter={f['adapter_acc']:.1f}% "
              f"native={f['native_acc']:.1f}% (Δ {f['forgetting']:+.1f})", flush=True)
    print(f"\n[result] {res}", flush=True)


if __name__ == "__main__":
    main()
