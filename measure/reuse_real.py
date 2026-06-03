"""Real-scenario reuse measurement via vLLM — TWO-PASS, batch = SAME video x B.

A batch is B COPIES of one video (n_vis identical -> no per-video heterogeneity, no
embeds-split mismatch). The two reuse caches need OPPOSITE engine settings, so we run
two passes (separate processes, --pass):

  --pass cold_vt  (enable_prefix_caching=False, mm cache OFF):
      baseline cold (encode+prefill+decode) + vt_reuse (embeds-inject, encoder skipped).
      prefix OFF so B identical copies each truly prefill (no dedup -> real cold batch).
  --pass kv       (enable_prefix_caching=True, mm cache ON):
      kv_reuse warm: populate once then repeat -> all B hit the shared KV (real warm).

Per-request cost = whole-batch wall / B. CUDA graphs (--cudagraph) for real-serving
throughput. H2D (DRAM->GPU) measured separately (retrieval added in the price model).
Qwen frame resolution capped via VIDEO_MAX/MIN_PIXELS. CUDA_VISIBLE_DEVICES=1.

Usage (run both passes, then analyze):
  python -m measure.reuse_real --model qwen2.5-vl-7b --pass cold_vt --frames 16 32 64 128 \
      --batches 1 4 8 16 --cudagraph
  python -m measure.reuse_real --model qwen2.5-vl-7b --pass kv      --frames 16 32 64 128 \
      --batches 1 4 8 16 --cudagraph
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402


def h2d_ms(nbytes: int) -> float:
    """Measured DRAM->GPU transfer (ms) for nbytes (pinned host -> cuda)."""
    import torch
    host = torch.empty(max(nbytes // 2, 1), dtype=torch.bfloat16, pin_memory=True)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); g = host.to("cuda:0", non_blocking=True); e.record()
    torch.cuda.synchronize()
    del g, host
    return s.elapsed_time(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pass", dest="pass_", required=True, choices=["cold_vt", "kv"])
    ap.add_argument("--videos-csv", default="results/nextqa_sample.csv")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--video-max-patches", type=int, default=768,
                    help="Qwen only: VIDEO_MAX_PIXELS = N*28*28 (Qwen-native 768).")
    ap.add_argument("--video-min-patches", type=int, default=128)
    ap.add_argument("--cudagraph", action="store_true")
    a = ap.parse_args()

    # H100(GPU1) is the default; ALLOW_GPU0=1 permits GPU0 (Blackwell) for FUNCTIONAL
    # validation only — those timings are NOT H100-normalized, do not use for the price model.
    if os.environ.get("ALLOW_GPU0") != "1":
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "must run on the H100 (GPU 1)"
    cfg = load_models(); spec = cfg.models[a.model]
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams
    from measure.stage_timing_vllm import build_video_request, build_video_request_internvl
    import torch

    is_internvl = spec.key.startswith("internvl")
    is_llava = spec.key.startswith("llava")
    needs_meta = spec.key.startswith("qwen3")
    img_tid = None
    if is_internvl:
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        vtid = tok.convert_tokens_to_ids("<|video_pad|>")
        filler = tok.encode("the")[-1]
        proc = None
        make_video = lambda p, nf, q: build_video_request_internvl(tok, p, nf, query=q)
    else:
        # pixel capping only for Qwen (dynamic resolution); LLaVA-OV is fixed 196 tok/frame.
        vpx = {} if is_llava else {"max_pixels": a.video_max_patches * 28 * 28,
                                   "min_pixels": a.video_min_patches * 28 * 28}
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **vpx)
        _cfg = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        vtid = getattr(_cfg, "video_token_id", None) or getattr(_cfg, "video_token_index", None)
        if is_llava:  # vt_reuse goes through the image-embeds path (video path ignores embeds)
            img_tid = getattr(_cfg, "image_token_index", None)
        filler = proc.tokenizer.encode("the")[-1]
        make_video = lambda p, nf, q: build_video_request(proc, p, n_frames=nf, with_metadata=needs_meta, query=q)

    rows = []
    with open(os.path.expanduser(a.videos_csv)) as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], (r.get("question") or "Describe in detail.").strip(),
                         float(r.get("duration_s") or 0)))

    # cold_vt pass: prefix OFF + mm OFF (so B identical copies each prefill; no dedup).
    # kv pass: prefix ON + mm ON (populate once -> all B hit shared KV; warm skips enc+pre).
    cold_vt = a.pass_ == "cold_vt"
    mm_kwargs = None if (is_internvl or is_llava) else {
        "max_pixels": a.video_max_patches * 28 * 28, "min_pixels": a.video_min_patches * 28 * 28}
    print(f"[reuse_real] pass={a.pass_} loading {spec.repo_id} "
          f"(prefix={'OFF' if cold_vt else 'ON'}, cudagraph={a.cudagraph}, batches={a.batches}) ...", flush=True)
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code,
              max_model_len=a.max_model_len, gpu_memory_utilization=0.85, enforce_eager=not a.cudagraph,
              max_num_seqs=max(a.batches),
              enable_prefix_caching=not cold_vt, mm_processor_cache_gb=(0 if cold_vt else 8),
              enable_mm_embeds=True, mm_processor_kwargs=mm_kwargs,
              limit_mm_per_prompt={"video": 1, "image": 1})
    HID = spec.hidden_size
    sp1 = SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1, ignore_eos=True, detokenize=False)
    spD = SamplingParams(temperature=0.0, min_tokens=a.decode_tokens, max_tokens=a.decode_tokens,
                         ignore_eos=True, detokenize=False)

    def gen(reqs, sp):
        rl = reqs if isinstance(reqs, list) else [reqs]
        t = time.perf_counter(); outs = llm.generate(rl, sp, use_tqdm=False)
        return (time.perf_counter() - t) * 1e3, outs

    def reset():
        llm.reset_prefix_cache()
        try:
            llm.reset_mm_cache()
        except Exception:
            pass

    out_path = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "reuse_real.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fcsv = open(out_path, "a", newline="")
    W = csv.DictWriter(fcsv, fieldnames=["model", "video_id", "frames", "batch", "variant", "metric",
                                         "run_idx", "value_ms", "n_vis", "prompt_tokens",
                                         "token_bytes", "kv_bytes", "duration_s", "timestamp"])
    if new:
        W.writeheader()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def emit(vid, nf, B, variant, metric, vals, n_vis, n_prompt, tb, kb, dur):
        for i, v in enumerate(vals):
            W.writerow({"model": spec.key, "video_id": vid, "frames": nf, "batch": B,
                        "variant": variant, "metric": metric, "run_idx": i,
                        "value_ms": round(v, 4), "n_vis": n_vis, "prompt_tokens": n_prompt,
                        "token_bytes": tb, "kv_bytes": kb, "duration_s": dur, "timestamp": ts})

    def measure(row, nf, B):
        """B identical copies of ONE video. per-request cost = whole-batch wall / B."""
        path, query, dur = row
        req = make_video(path, nf, query)[0]
        reqs = [req] * B
        vid = Path(path).stem + (f"_b{B}" if B > 1 else "")
        reset(); _, outs = gen([req], sp1)
        o = outs[0]
        n_prompt = len(o.prompt_token_ids)
        n_vis = sum(1 for t in o.prompt_token_ids if t == vtid)
        tb = cfg.vision_bytes(spec.key, n_vis); kb = cfg.kv_bytes(spec.key, n_vis)

        if cold_vt:
            text_reqs = [{"prompt_token_ids": [filler] * n_prompt} for _ in range(B)]
            for _ in range(a.warmup):
                reset(); gen(reqs, sp1)
            c_ttft, c_full, c_pre = [], [], []
            for _ in range(a.runs):
                reset(); w1, _ = gen(reqs, sp1)
                reset(); wf, _ = gen(reqs, spD)
                reset(); wp, _ = gen(text_reqs, sp1)
                c_ttft.append(w1 / B); c_full.append(wf / B); c_pre.append(wp / B)
            emit(vid, nf, B, "cold", "ttft", c_ttft, n_vis, n_prompt, tb, kb, dur)
            emit(vid, nf, B, "cold", "full", c_full, n_vis, n_prompt, tb, kb, dur)
            emit(vid, nf, B, "cold", "prefill_textbase", c_pre, n_vis, n_prompt, tb, kb, dur)
            # vt_reuse: B identical embeds (same video -> same n_vis -> split matches)
            if is_internvl:
                ip = tok.apply_chat_template([{"role": "user", "content": "<image>\n" + query}],
                                             tokenize=False, add_generation_prompt=True)
                er = {"prompt": ip, "multi_modal_data": {"image": torch.randn(1, n_vis, HID, dtype=torch.bfloat16)}}
            elif is_llava:  # reuse cold prompt 1:1, swap video placeholder -> image placeholder, inject image_embeds
                ids = [img_tid if t == vtid else t for t in o.prompt_token_ids]
                er = {"prompt_token_ids": ids,
                      "multi_modal_data": {"image": torch.randn(1, n_vis, HID, dtype=torch.bfloat16)}}
            else:
                _v = req["multi_modal_data"]["video"]; _f = _v[0] if isinstance(_v, tuple) else _v
                _raw = proc(text=[req["prompt"]], videos=[_f], return_tensors="pt")
                er = {"prompt": req["prompt"], "multi_modal_data": {"video": {
                    "video_embeds": torch.randn(n_vis, HID, dtype=torch.bfloat16),
                    "video_grid_thw": _raw["video_grid_thw"]}}}
            ereqs = [er] * B
            for _ in range(a.warmup):
                reset(); gen(ereqs, sp1)
            t_ttft, t_full = [], []
            for _ in range(a.runs):
                reset(); w1, _ = gen(ereqs, sp1)
                reset(); wf, _ = gen(ereqs, spD)
                t_ttft.append(w1 / B); t_full.append(wf / B)
            emit(vid, nf, B, "vt_reuse", "ttft_inject", t_ttft, n_vis, n_prompt, tb, kb, dur)
            emit(vid, nf, B, "vt_reuse", "full_inject", t_full, n_vis, n_prompt, tb, kb, dur)
            emit(vid, nf, B, "vt_reuse", "h2d_tok", [h2d_ms(tb) for _ in range(a.runs)], n_vis, n_prompt, tb, kb, dur)
            print(f"[reuse_real] cold_vt {vid}_{nf}f b{B} n_vis={n_vis}: "
                  f"cold_ttft={statistics.median(c_ttft):.0f} cold_pre={statistics.median(c_pre):.0f} "
                  f"vt={statistics.median(t_ttft):.0f} (per-req ms)", flush=True)
        else:  # kv pass
            for _ in range(a.warmup):
                gen(reqs, sp1)
            reset(); gen(reqs, spD)                       # populate KV (B copies dedup to 1)
            k_ttft, k_full = [], []
            for _ in range(a.runs):
                w1, _ = gen(reqs, sp1); wf, _ = gen(reqs, spD)
                k_ttft.append(w1 / B); k_full.append(wf / B)
            emit(vid, nf, B, "kv_reuse", "ttft_warm", k_ttft, n_vis, n_prompt, tb, kb, dur)
            emit(vid, nf, B, "kv_reuse", "full_warm", k_full, n_vis, n_prompt, tb, kb, dur)
            emit(vid, nf, B, "kv_reuse", "h2d_kv", [h2d_ms(kb) for _ in range(a.runs)], n_vis, n_prompt, tb, kb, dur)
            print(f"[reuse_real] kv {vid}_{nf}f b{B} n_vis={n_vis}: "
                  f"kv_warm={statistics.median(k_ttft):.0f} (per-req ms)", flush=True)
        fcsv.flush()

    for nf in a.frames:
        for B in a.batches:
            groups = ([[r] for r in rows] if B == 1 else [rows[:1]])  # B>1: representative video x B
            targets = [g[0] for g in groups]
            for row in targets:
                try:
                    measure(row, nf, B)
                except Exception as e:
                    gid = Path(row[0]).stem
                    print(f"[reuse_real] {gid}_{nf}f b{B} ({a.pass_}): FAILED "
                          f"({type(e).__name__}: {str(e)[:80]}) — skip", flush=True)
                    try:
                        reset()
                    except Exception:
                        pass
    fcsv.close()
    print(f"[reuse_real] pass={a.pass_} wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
