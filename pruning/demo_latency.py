"""Real measured latency breakdown of vision-token pruning (GPU).

Times the actual stages with cuda events on a real model, per keep-ratio, and composes
them the way the cost model's three variants do:
  cold      TTFT = encode + prefill(full n_vis)
  vt_reuse  TTFT = prefill(full n_vis)        (encode skipped — vision tokens are stored)
  vt+prune  TTFT = prefill(keep · n_vis)      (fewer vision tokens prefilled)

n_vis defaults to VIDEO scale (--n-vis 8192 ≈ 32 InternVL frames), because that is where
prefill is super-linear in n_vis and pruning's latency win shows; a single 256-token
image is weight-bandwidth bound and prefill barely moves. prefill is measured on the real
LLM over (synthetic) vision embeddings — latency is content-independent. encode (ViT) is
measured on one real frame and scaled ×frames (encode is linear in frames; CLAUDE.md §3).

  # GPU env (transformers 4.57 or 5.9, e.g. the repo's vlmcost):
  CUDA_VISIBLE_DEVICES=<gpu> python -m pruning.demo_latency --which internvl --n-vis 8192

Runs on transformers 4.57 and 5.9 (incl. the vlmcost/vLLM env); needs a GPU. See pruning/README.md.
"""
from __future__ import annotations

import argparse


def main() -> None:
    import numpy as np
    from PIL import Image
    from pruning.methods import Pruner

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="internvl", choices=["internvl", "llava15"])
    ap.add_argument("--n-vis", type=int, default=8192, help="full vision-token count (video scale)")
    ap.add_argument("--keeps", default="1.0,0.5,0.25,0.1")
    ap.add_argument("--runs", type=int, default=7)
    a = ap.parse_args()
    keeps = [float(x) for x in a.keeps.split(",")]
    N = a.n_vis

    pr = Pruner(a.which)

    # encode (ViT+projector) measured on ONE real frame, then scaled by the frame count.
    sz = pr.resize or (448, 448)
    img = Image.fromarray(np.random.randint(0, 255, (sz[1], sz[0], 3), dtype="uint8"))
    inp = pr.build(img, "Describe the image.")
    tok_per_frame = pr.feats_of(inp).shape[0]
    enc_per_frame = pr.encode_latency(inp, runs=a.runs)
    frames = max(1, round(N / tok_per_frame))
    enc_full = enc_per_frame * frames

    print(f"\nlatency breakdown — {a.which}  n_vis(full)={N}  (~{frames} frames @ "
          f"{tok_per_frame} tok/frame)  runs={a.runs}, cuda events")
    print(f"  encode/frame = {enc_per_frame:.1f} ms  → encode(full {frames}f) ≈ {enc_full:.0f} ms "
          f"(measured/frame × frames; vt_reuse skips this entirely)\n")

    # prefill at each keep (real LLM forward over n_vis synthetic vision tokens)
    rows = []
    for keep in keeps:
        n_k = max(1, round(N * keep))
        pre_ms, seq = pr.prefill_latency_nvis(n_k, runs=a.runs)
        rows.append((keep, n_k, seq, pre_ms))
    pre_full = rows[0][3]

    print(f"  {'keep':>6} {'n_vis':>7} {'prefill_ms':>11} | "
          f"{'cold TTFT':>10} {'vt TTFT':>9} {'vt+prune':>9}")
    for keep, n_k, seq, pre_ms in rows:
        cold = enc_full + pre_full          # baseline: encode + full prefill
        vt = pre_full                       # vt_reuse: full prefill, encode skipped
        vtp = pre_ms                         # vt_reuse + prune: shorter prefill
        print(f"  {keep:>6.2f} {n_k:>7} {pre_ms:>11.1f} | "
              f"{cold:>10.1f} {vt:>9.1f} {vtp:>9.1f}")
    print(f"\n  prefill(full)={pre_full:.1f}ms → at smallest keep={rows[-1][3]:.1f}ms "
          f"({100*rows[-1][3]/pre_full:.0f}% of full).")
    print("  cold→vt: skip encode; vt→vt+prune: prefill fewer vision tokens (the pruning win).")


if __name__ == "__main__":
    main()
