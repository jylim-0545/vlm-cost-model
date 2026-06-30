"""Measured latency breakdown for vision-token sharing (REAL model, GPU).

Times, with cuda events on a real LLaVA-OV + SigLIP hub:
  - native vision tower (the encoder a standalone backbone runs)   E_native
  - shared SigLIP hub encode                                       E_hub
  - the adapter apply (ridge affine and 2-layer MLP)               a

and composes the cost model's "encode once, serve N" the way sharing.cost does:
  baseline (no sharing) = N x E_native ;  shared = E_hub + N x a.
This validates the report's "adapter ~1% of the ViT" and the ~74%@N=4 encode saving on
real hardware, and prints a measured --hub-encode-ms to feed `sharing.demo_cost`.

  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_latency --backbone llavaov --runs 20

The vt_reuse (skip-encode) latency win is the SAME mechanism the cost repo already measures
for one model; sharing extends it across N backbones (encode amortized) — see sharing/cost.py.
GPU + transformers (the study's 4.57 box). See sharing/README.md.
"""
from __future__ import annotations

import argparse


def main() -> None:
    import numpy as np
    from PIL import Image
    from sharing import adapters, cost
    from sharing.methods import HubShare

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="llavaov")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ns", default="1,2,4,8,16")
    a = ap.parse_args()

    share = HubShare(a.backbone)
    torch = share.torch
    ns = [int(x) for x in a.ns.split(",")]

    img = Image.fromarray(np.random.randint(0, 255, (share.res, share.res, 3), dtype="uint8"))
    inp = share.build(img, "x")
    pv = inp["pixel_values"]

    def cuda_ms(fn):
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.runs):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2]

    # build adapters of each kind, fit on a single image's tokens (latency is content-free)
    x = share.hub_on_pixels(pv).reshape(-1, 1152)
    y = share.native_vt_on_pixels(pv).reshape(-1, share.vt_dim)
    ridge = adapters.RidgeAffine.fit(x, y).to(share.device).float()
    mlp = adapters.ZScoreMLP(1152, 2048, share.vt_dim)
    m, s = adapters.zscore_stats(x); mlp.set_stats(m, s)
    mlp = mlp.to(share.device).float()
    hub_tok = x.float()

    e_native = cuda_ms(lambda: share.native_vt_on_pixels(pv))
    e_hub = cuda_ms(lambda: share.hub_on_pixels(pv))
    a_ridge = cuda_ms(lambda: ridge(hub_tok))
    a_mlp = cuda_ms(lambda: mlp(hub_tok))

    n_tok = x.shape[0]
    print(f"\nlatency breakdown — {a.backbone}  (per image, {n_tok} hub tokens, "
          f"runs={a.runs}, cuda events)")
    print(f"  native VT encode   E_native = {e_native:7.2f} ms")
    print(f"  shared hub encode  E_hub    = {e_hub:7.2f} ms")
    print(f"  adapter (ridge)    a        = {a_ridge:7.3f} ms  ({100*a_ridge/e_native:.2f}% of E_native)")
    print(f"  adapter (mlp)      a        = {a_mlp:7.3f} ms  ({100*a_mlp/e_native:.2f}% of E_native)")

    print(f"\n  encode 'once, serve N' — baseline N×E_native vs shared E_hub + N×a (mlp)")
    print(f"  {'N':>4} {'baseline_ms':>12} {'shared_ms':>11} {'saving%':>9}")
    for n in ns:
        base = n * e_native
        shared = e_hub + n * a_mlp
        print(f"  {n:>4d} {base:>12.1f} {shared:>11.1f} {100*(base-shared)/base:>8.1f}%")

    # FLOP-based cross-check + a measured hub-encode-ms for demo_cost
    g4 = cost.encode_share(4)
    print(f"\n  (FLOP model: adapter {cost.ADAPTER_MLP_GFLOPS:.2f} GFLOPs = "
          f"{100*cost.ADAPTER_MLP_GFLOPS/cost.HUB_VIT_GFLOPS:.1f}% of ViT; N=4 saving "
          f"{100*g4['saving_frac']:.0f}%)")
    print(f"  feed measured value:  python -m sharing.demo_cost --hub-encode-ms {e_hub:.1f}")


if __name__ == "__main__":
    main()
