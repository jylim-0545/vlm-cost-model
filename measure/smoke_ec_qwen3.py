"""Smoke: does LMCache Encoder Cache (EC) work for Qwen3-VL (post-projector vt_reuse)?

Qwen3's DeepStack makes pre-projector reuse ill-defined, so its vt_reuse = EC (cache the full
vision-tower output, skip the tower on mm_hash hit). EC is model-agnostic (caches whatever
get_multimodal_embeddings returns), so it should work for Qwen3 too. This confirms: EC stores the
embeds (EC put) and the warm request skips the tower (TTFT drops). vlmcost (vLLM 0.22 + lmcache
0.4.6); EC connector lives in measure/lmcache_ec_connector.py -> need PYTHONPATH=repo for the worker.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")
os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "20.0")
os.environ.setdefault("LMCACHE_CHUNK_SIZE", "256")

import sys, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--video", default=os.path.expanduser("~/VLM/scratch/nextqa_videos/5396384503.mp4"))
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.config import ECTransferConfig
    from measure.stage_timing_vllm import build_video_request

    assert "H100" in torch.cuda.get_device_name(0), "H100 only"
    proc = AutoProcessor.from_pretrained(a.repo)
    etc = ECTransferConfig(ec_connector="LMCacheECConnector", ec_role="ec_both",
                           ec_connector_module_path="measure.lmcache_ec_connector")
    # NOTE: do NOT set mm_processor_cache_gb=0 here — that disables content-based mm_hash, so EC
    # keys by a positional id (renderer0-mm-N) and never hits. Default mm cache -> stable content
    # hash -> warm requests skip the tower (via EC / mm cache). We measure the tower-skip TTFT.
    llm = LLM(model=a.repo, max_model_len=40960, enforce_eager=True, gpu_memory_utilization=0.85,
              enable_prefix_caching=False,
              max_num_batched_tokens=32768, limit_mm_per_prompt={"video": 1},
              ec_transfer_config=etc)
    sp = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    def ttft(req):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        llm.generate([req], sp); torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    # global warmup (absorb first-call lazy init so the per-frame cold isn't 11s-polluted)
    _wreq, _ = build_video_request(proc, a.video, n_frames=8, with_metadata=True)
    ttft(_wreq)

    print(f"\n{'fr':>4} | {'cold(EC store)':>15}{'warm(EC hit)':>14} | saving")
    for nf in a.frames:
        req, _ = build_video_request(proc, a.video, n_frames=nf, with_metadata=True)  # Qwen3 needs metadata
        cold = ttft(req)                       # EC miss -> tower runs + EC stores
        warms = [ttft(req) for _ in range(a.runs)]   # EC hit -> tower skipped
        w = statistics.median(warms)
        print(f"{nf:>4} | {cold:>15.1f}{w:>14.1f} | {cold-w:>+7.1f}  (-> EC tower-skip works if warm << cold)")
    print("\n[check stderr for 'EC put: stored' (store) + warm<<cold (tower skipped)]")


if __name__ == "__main__":
    main()
