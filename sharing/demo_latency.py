"""Measure how cheap the adapter is vs the vision encoder (REAL model, GPU).

Times, with cuda events on a real LLaVA-OV + SigLIP hub, the per-image latency of:
  - the backbone's native vision tower (the encoder a standalone model runs)   E_native
  - the shared SigLIP hub encode                                               E_hub
  - the adapter apply (ridge affine, and 2-layer MLP)                          a

and reports the adapter as a % of the encoder. This is the empirical basis for "the adapter
is ~1% of the ViT" — raw input for whoever models the serving cost; this module itself does
NOT compute cost/break-even.

  CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_latency --backbone llavaov --runs 20

GPU + transformers (the study's box). See sharing/README.md.
"""
from __future__ import annotations

import argparse


def main() -> None:
    import numpy as np
    from PIL import Image
    from sharing import adapters
    from sharing.methods import HubShare

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="llavaov")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    share = HubShare(a.backbone)
    torch = share.torch

    img = Image.fromarray(np.random.randint(0, 255, (share.res, share.res, 3), dtype="uint8"))
    pv = share.build(img, "x")["pixel_values"]

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

    # adapters (latency is content-free, so fit on this one image's tokens)
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

    print(f"\nadapter vs encoder latency — {a.backbone}  (per image, {x.shape[0]} hub tokens, "
          f"runs={a.runs}, cuda events)")
    print(f"  native VT encode   E_native = {e_native:7.2f} ms")
    print(f"  shared hub encode  E_hub    = {e_hub:7.2f} ms")
    print(f"  adapter (ridge)    a        = {a_ridge:7.3f} ms  ({100*a_ridge/e_native:.2f}% of E_native)")
    print(f"  adapter (mlp)      a        = {a_mlp:7.3f} ms  ({100*a_mlp/e_native:.2f}% of E_native)")
    print("\n  → the adapter is a tiny fraction of one ViT encode; sharing one hub across N "
          "backbones replaces N encodes with 1 encode + N (cheap) adapters.")


if __name__ == "__main__":
    main()
