"""SMOKE TEST: vLLM 0.18 + LMCacheConnectorV1 KV reuse for VLM video (LLaVA-OV-7B).

Goal (functional, not a price-model number): confirm the LMCache track is viable —
  (1) vLLM 0.18 loads LLaVA-OV-7B video,
  (2) LMCacheConnectorV1 initializes (kv_both) with a CPU-DRAM backend,
  (3) the SAME (video, prompt) on the 2nd request REUSES KV from LMCache (hit logs +
      lower TTFT), i.e. KV is offloaded to a storage tier and read back.

Runs ONLY in the isolated `lmcache` conda env (vLLM 0.18 / lmcache 0.4.4). Writes nothing
to results/nextqa/reuse_real.csv. H100 only (CUDA_VISIBLE_DEVICES=1, asserted by name).
"""
from __future__ import annotations
import os

# ---- LMCache backend = CPU DRAM tier (set BEFORE vllm import; connector reads from_env) ----
os.environ.setdefault("LMCACHE_CHUNK_SIZE", "256")
os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")
os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "15.0")   # GB of CPU DRAM for KV

import time
from pathlib import Path

import numpy as np
import torch


def assert_gpu() -> None:
    """H100 only — UNLESS ALLOW_GPU0=1 (functional validation on Blackwell; timings are
    NOT H100-normalized, per CLAUDE.md §8 escape hatch). This smoke test prints latency for
    a functional check only (writes no CSV), so GPU0 is acceptable here."""
    name = torch.cuda.get_device_name(0)
    if os.environ.get("ALLOW_GPU0") == "1":
        print(f"[gpu] ALLOW_GPU0=1 — running on {name!r} for FUNCTIONAL validation only "
              f"(timings NOT H100-normalized)")
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", \
        f"pin H100: CUDA_VISIBLE_DEVICES must be '1', got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    assert "H100" in name, f"visible device is {name!r}, not H100 — refusing (would contaminate)"
    print(f"[gpu] OK pinned to {name}")


def build_llava_video_request(processor, path: str, n_frames: int,
                              query: str = "Describe this video in detail."):
    import decord                                              # same decoder as the main pipeline
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path)
    idx = np.linspace(0, len(vr) - 1, num=min(n_frames, len(vr))).round().astype(int)
    frames = vr.get_batch(idx).asnumpy()                       # (T,H,W,C) uint8 RGB
    content = [{"type": "video"}, {"type": "text", "text": query}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt, "multi_modal_data": {"video": frames}}, len(idx)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    ap.add_argument("--video", default=os.path.expanduser("~/VLM/scratch/nextqa_videos/5396384503.mp4"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    assert_gpu()
    assert Path(a.video).exists(), f"video not found: {a.video}"

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"[load] processor {a.model}")
    proc = AutoProcessor.from_pretrained(a.model)

    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both",
                           kv_load_failure_policy="recompute")
    print("[load] LLM (vLLM 0.18 + LMCacheConnectorV1, kv_both)")
    llm = LLM(model=a.model, max_model_len=32768, enforce_eager=True,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"video": 1},
              kv_transfer_config=ktc)

    req, n_frames_real = build_llava_video_request(proc, a.video, a.frames)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, detokenize=False)
    print(f"[run] video={Path(a.video).stem} frames={n_frames_real} "
          f"prompt_chars={len(req['prompt'])} max_tokens={a.max_tokens}")

    lats = []
    for i in range(a.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([req], sp)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        n_prompt = len(out[0].prompt_token_ids)
        lats.append(dt)
        tag = "COLD (store)" if i == 0 else "WARM (expect LMCache hit)"
        print(f"  run {i}: {dt:8.1f} ms   n_prompt_tokens={n_prompt}   [{tag}]")

    print("\n[result]")
    print(f"  run0 (cold):  {lats[0]:8.1f} ms")
    for i in range(1, len(lats)):
        print(f"  run{i} (warm): {lats[i]:8.1f} ms   ({100*(lats[0]-lats[i])/lats[0]:+.1f}% vs cold)")
    print("\n  -> look ABOVE for LMCache 'store'/'retrieve'/'hit tokens' log lines to confirm "
          "KV was offloaded to CPU-DRAM and read back on warm runs.")


if __name__ == "__main__":
    main()
