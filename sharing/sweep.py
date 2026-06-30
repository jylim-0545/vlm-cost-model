"""Variant × seed sweep for the adapter recovery study (multi-GPU), the library form of
`EfficientVLM/scripts/run_B.sh`.

Runs `sharing.demo_train` for each (task, mode, seed) across the available GPUs (one job per
GPU at a time, parallel across GPUs), parses the printed `[result]` dict, and aggregates
recovery (= adapter/native) mean±std per (task, mode) into a table + CSV. This is the
3-seed error-bar run behind FINDINGS.md's recovery ladder.

  # ALL variants × 3 seeds on 4 GPUs (LONG — this is the human's full run, not a smoke):
  python -m sharing.sweep --gpus 0,1,2,3 --seeds 0,1,2 \
      --tasks mmstar,nextqa --modes raw,mlp_recon,mlp_e2e --out-csv logs/share_sweep.csv

  # quick smoke (tiny):
  python -m sharing.sweep --gpus 0,1 --seeds 0 --tasks mmstar --modes raw,ridge \
      --n-eval 40 --pre-samples 30 --ft-steps 32

mode->flags mapping (matches the study recipe): mlp_e2e gets recon-anchor lambda=8 + cosine;
recon = mlp_recon (no fine-tune); raw/ridge need no fine-tune. holistic (nextqa) uses --frames.
GPU + transformers (the study's 4.57 box). See sharing/README.md.
"""
from __future__ import annotations

import argparse
import ast
import csv as csvmod
import os
import queue
import statistics
import subprocess
import sys
import threading


def _job_cmd(py, task, mode, seed, a):
    cmd = [py, "-u", "-m", "sharing.demo_train", "--task", task, "--mode", mode,
           "--seed", str(seed), "--n-eval", str(a.n_eval), "--pre-samples", str(a.pre_samples),
           "--pre-steps", str(a.pre_steps)]
    if task == "nextqa":
        cmd += ["--frames", str(a.frames)]
    if mode == "mlp_e2e":
        cmd += ["--ft-steps", str(a.ft_steps), "--recon-lambda", str(a.recon_lambda),
                "--sched", "cosine"]
    if a.multitask:
        cmd += ["--multitask", a.multitask]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default="mmstar,nextqa")
    ap.add_argument("--modes", default="raw,mlp_recon,mlp_e2e")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--pre-samples", type=int, default=400)
    ap.add_argument("--pre-steps", type=int, default=2000)
    ap.add_argument("--ft-steps", type=int, default=600)
    ap.add_argument("--recon-lambda", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--multitask", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--logdir", default="logs/share_sweep")
    ap.add_argument("--out-csv", default="logs/share_sweep/recovery.csv")
    a = ap.parse_args()

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    os.makedirs(a.logdir, exist_ok=True)

    jobs = [(t, m, s) for t in tasks for m in modes for s in seeds]
    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    results = []
    lock = threading.Lock()
    print(f"[sweep] {len(jobs)} jobs over {len(gpus)} GPUs: tasks={tasks} modes={modes} seeds={seeds}",
          flush=True)

    def worker(gpu):
        while True:
            try:
                task, mode, seed = q.get_nowait()
            except queue.Empty:
                return
            tag = f"{task}_{mode}_s{seed}"
            logp = os.path.join(a.logdir, tag + ".log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            print(f"[sweep] GPU{gpu} start {tag}", flush=True)
            with open(logp, "w") as f:
                p = subprocess.run(_job_cmd(a.python, task, mode, seed, a), env=env,
                                   stdout=f, stderr=subprocess.STDOUT)
            rec = None
            for line in reversed(open(logp).read().splitlines()):
                if line.strip().startswith("[result]"):
                    try:
                        rec = ast.literal_eval(line.split("[result]", 1)[1].strip())
                    except Exception:
                        pass
                    break
            with lock:
                if rec:
                    results.append(rec)
                    print(f"[sweep] GPU{gpu} done  {tag}: recovery={rec['recovery']:.3f} "
                          f"(adapter {rec['adapter_acc']:.1f} / native {rec['native_acc']:.1f})",
                          flush=True)
                else:
                    print(f"[sweep] GPU{gpu} FAILED {tag} (rc={p.returncode}); see {logp}", flush=True)
            q.task_done()

    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    if not results:
        print("[sweep] no results parsed; check logs")
        return
    with open(a.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\n[sweep] wrote {a.out_csv} ({len(results)} rows)")

    # aggregate recovery mean±std per (task, mode)
    print(f"\n  {'task':>8} {'mode':>10} {'n':>3} {'recovery (mean±std)':>22} {'adapter%':>9} {'native%':>8}")
    agg = {}
    for r in results:
        agg.setdefault((r["task"], r["mode"]), []).append(r)
    for (task, mode), rs in sorted(agg.items()):
        recs = [x["recovery"] for x in rs]
        ad = statistics.mean(x["adapter_acc"] for x in rs)
        na = statistics.mean(x["native_acc"] for x in rs)
        sd = statistics.pstdev(recs) if len(recs) > 1 else 0.0
        print(f"  {task:>8} {mode:>10} {len(rs):>3} {statistics.mean(recs):>14.3f} ± {sd:<5.3f}"
              f"   {ad:>8.1f} {na:>8.1f}")


if __name__ == "__main__":
    main()
