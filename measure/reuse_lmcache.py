"""LMCache kv_reuse measurement — the REAL KV-offload-and-retrieve path, to compare against
our analytical kv_reuse (measure/reuse_real.py, vLLM 0.22 prefix-cache-warm + COMPUTED retrieval).

This measures, per (model, video, frames) on a storage TIER:
  - cold      : first request for this (video,frames) key -> LMCache MISS -> recompute encode+prefill
                (+ store KV to the tier). The within-engine baseline.
  - kv_lmcache: subsequent request -> LMCache HIT -> KV LOADED BACK from the tier (real retrieval),
                prefill skipped. This is LMCache's kv_reuse.

CRITICAL — force real retrieval: vLLM's own prefix cache is DISABLED (enable_prefix_caching=False)
so KV is NOT GPU-resident across requests; the only reuse path is LMCache, which therefore must
actually LOAD from the tier (DRAM / local disk). With prefix caching ON, the smoke test showed
'need to load: 0' (served from GPU cache) — that would measure a GPU hit, not tier retrieval.

Runs in the isolated `lmcache` conda env (vLLM 0.18 / lmcache 0.4.4). H100 ONLY — LMCache's
prebuilt c_ops has no Blackwell (sm_120) kernel. Writes results/lmcache/ (isolated; never touches
results/nextqa/reuse_real.csv). ONE tier per process (LMCACHE_* env is read at engine init); use
the launcher to sweep tiers.
"""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np


def configure_tier(tier: str, disk_path: str) -> dict:
    """Set LMCACHE_* env BEFORE vllm import. Returns a dict of the knobs (for logging)."""
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    if tier == "dram":                       # CPU-DRAM tier: large CPU pool, no disk
        cfg = {"LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "20.0"}
    elif tier == "disk":                     # local-NVMe tier: tiny CPU pool forces spill to disk,
        cfg = {"LMCACHE_LOCAL_CPU": "True",  # so reuse loads disk->CPU->GPU (real NVMe retrieval)
               "LMCACHE_MAX_LOCAL_CPU_SIZE": "0.5",
               "LMCACHE_LOCAL_DISK": f"file://{disk_path}",
               "LMCACHE_MAX_LOCAL_DISK_SIZE": "80.0"}
        Path(disk_path).mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"unknown tier {tier!r}")
    os.environ.update(cfg)
    return cfg


def assert_h100() -> None:
    import torch
    name = torch.cuda.get_device_name(0)
    assert "H100" in name, (f"LMCache track is H100-only (its c_ops has no Blackwell kernel); "
                            f"visible device is {name!r}. Set CUDA_VISIBLE_DEVICES=1.")
    print(f"[gpu] OK pinned to {name}")


def build_llava_video_request(processor, path: str, n_frames: int,
                              query: str = "Describe this video in detail."):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path)
    idx = np.linspace(0, len(vr) - 1, num=min(n_frames, len(vr))).round().astype(int)
    frames = vr.get_batch(idx).asnumpy()
    content = [{"type": "video"}, {"type": "text", "text": query}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt, "multi_modal_data": {"video": frames}}, len(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    ap.add_argument("--model-tag", default="llava-ov-7b", help="tag written to CSV (match reuse_real)")
    ap.add_argument("--video", default=os.path.expanduser("~/VLM/scratch/nextqa_videos/5396384503.mp4"))
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--mode", choices=["lmcache", "ours", "ec"], default="lmcache",
                    help="lmcache=LMCache KV tier offload+load; ours=vanilla vLLM prefix-cache warm "
                         "(GPU-resident KV); ec=LMCache Encoder Cache (vision-token reuse, encode-skip)")
    ap.add_argument("--tier", choices=["dram", "disk"], help="required for --mode lmcache")
    ap.add_argument("--disk-path", default=os.path.expanduser("~/VLM/scratch/lmcache_disk"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--csv", default="results/lmcache/reuse_lmcache.csv")
    a = ap.parse_args()
    if a.mode in ("lmcache", "ec") and not a.tier:
        ap.error(f"--tier is required for --mode {a.mode}")

    # 'ours' = OUR kv_reuse (reuse_real mechanism): vanilla vLLM prefix-cache warm, NO LMCache,
    #          KV stays GPU-RESIDENT across requests (no real tier retrieval). 'lmcache' = LMCache
    #          offload to a storage tier + real LOAD-back. Same engine/model/video/frames otherwise
    #          -> apples-to-apples head-to-head (the warm DELTA = real cost of going to a tier).
    is_ours = a.mode == "ours"
    is_ec = a.mode == "ec"
    cfg = {} if is_ours else configure_tier(a.tier, a.disk_path)
    tier_label = "gpu_resident" if is_ours else a.tier
    warm_variant = {"ours": "kv_ours", "lmcache": "kv_lmcache", "ec": "ec_reuse"}[a.mode]

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig, ECTransferConfig

    assert_h100()
    assert Path(a.video).exists(), f"video not found: {a.video}"
    print(f"[mode] {a.mode}  tier={tier_label}  warm_variant={warm_variant}  cfg={cfg}")

    proc = AutoProcessor.from_pretrained(a.model)
    # max_num_batched_tokens=32768 raises the encoder-cache budget so 128f (25088 vis tok) fits.
    common = dict(model=a.model, max_model_len=32768, enforce_eager=True,
                  gpu_memory_utilization=0.85, limit_mm_per_prompt={"video": 1},
                  max_num_batched_tokens=32768)
    if is_ours:
        # prefix caching ON -> 2nd identical request HITS the GPU-resident KV (= our kv_reuse).
        llm = LLM(enable_prefix_caching=True, **common)
    elif is_ec:
        # LMCache ENCODER CACHE: cache the encoder output (vision tokens) to a tier; on mm_hash
        # hit, reload it and SKIP the encoder. prefix caching OFF so prefill ALWAYS runs ->
        # matches our vt_reuse semantics (encode-skip only). Connector lives in THIS repo
        # (measure/lmcache_ec_connector.py); vLLM resolves it via ec_connector_module_path.
        etc = ECTransferConfig(ec_connector="LMCacheECConnector", ec_role="ec_both",
                               ec_connector_module_path="measure.lmcache_ec_connector")
        llm = LLM(enable_prefix_caching=False, ec_transfer_config=etc, **common)
    else:
        # prefix caching OFF -> no GPU-resident reuse -> LMCache must LOAD KV from the tier.
        ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both",
                               kv_load_failure_policy="recompute")
        llm = LLM(enable_prefix_caching=False, kv_transfer_config=ktc, **common)
    sp_ttft = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    vid = Path(a.video).stem
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.csv).exists()
    fcsv = open(a.csv, "a", newline="")
    W = csv.DictWriter(fcsv, fieldnames=["model", "video_id", "frames", "n_frames_real", "n_vis",
                                         "n_prompt_tokens", "tier", "variant", "metric",
                                         "value_ms", "run_idx", "engine", "lmcache_ver"])
    if new:
        W.writeheader()

    def gen_ttft(req) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([req], sp_ttft)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        return dt, len(out[0].prompt_token_ids)

    import vllm as _vllm
    lmver = "none" if is_ours else __import__("lmcache").__version__
    # Pre-loop warmup on a THROWAWAY frame count (not measured) to absorb first-call lazy
    # init / compilation, so each nf's COLD is a clean miss (not polluted like a 5.3s first gen).
    try:
        wreq, _ = build_llava_video_request(proc, a.video, 8)
        gen_ttft(wreq)
        print("[warmup] done (throwaway 8f)")
    except Exception as e:
        print(f"[warmup] skipped: {e}")

    for nf in a.frames:
        try:
            req, nfr = build_llava_video_request(proc, a.video, nf)
        except Exception as e:
            print(f"  frames={nf}: build FAILED, skip ({e})"); continue
        n_vis = nfr * 196                         # LLaVA-OV: 196 tok/frame FIXED
        # run 0 = COLD (miss -> recompute; lmcache mode also stores KV to the tier)
        try:
            cold_ms, n_prompt = gen_ttft(req)
        except Exception as e:
            print(f"  frames={nf} n_vis={n_vis}: cold gen FAILED, skip ({type(e).__name__})"); continue
        W.writerow({"model": a.model_tag, "video_id": vid, "frames": nf, "n_frames_real": nfr,
                    "n_vis": n_vis, "n_prompt_tokens": n_prompt, "tier": tier_label, "variant": "cold",
                    "metric": "ttft", "value_ms": round(cold_ms, 3), "run_idx": 0,
                    "engine": f"vllm-{_vllm.__version__}", "lmcache_ver": lmver})
        # warmup (discard) then timed WARM runs = reuse (ours: GPU hit | lmcache: tier LOAD)
        for _ in range(a.warmup):
            gen_ttft(req)
        warm = []
        for r in range(a.runs):
            w_ms, _ = gen_ttft(req)
            warm.append(w_ms)
            W.writerow({"model": a.model_tag, "video_id": vid, "frames": nf, "n_frames_real": nfr,
                        "n_vis": n_vis, "n_prompt_tokens": n_prompt, "tier": tier_label,
                        "variant": warm_variant, "metric": "ttft", "value_ms": round(w_ms, 3),
                        "run_idx": r, "engine": f"vllm-{_vllm.__version__}", "lmcache_ver": lmver})
        fcsv.flush()
        print(f"  frames={nf:>3} n_vis={n_vis:>5} | cold {cold_ms:8.1f}ms | "
              f"{warm_variant}(median) {statistics.median(warm):7.1f}ms | "
              f"saving {cold_ms - statistics.median(warm):7.1f}ms  ({tier_label})")
    fcsv.close()
    print(f"\n[done] wrote {a.csv}  (watch stderr for LMCache 'need to load' > 0 = real tier retrieval)")


if __name__ == "__main__":
    main()
