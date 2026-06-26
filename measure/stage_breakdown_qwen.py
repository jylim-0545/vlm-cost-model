"""Qwen2.5-VL cold stage breakdown — REAL measured: preprocessing / encoding(ViT) / projector(merger)
/ prefill (= TTFT segments) + decoding shown as TPOT. b=8, 128f, one video, eager (mm_cache=0 so vision
really runs). CUDA events time the vision transformer (incl merger) and the merger; preprocessing =
generate-start -> vision-forward entry (CPU pixel transform + sched, NOT video decode which is outside).
prefill = ttft - preprocessing - vision_total. TPOT = (full - ttft)/decode_tokens. Plots a stacked TTFT
bar + a TPOT bar. video DECODE excluded (done in make_req, outside timing)."""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models                                  # noqa: E402

STAGE = {"vis": [], "merger": []}
GEN = {"t0": None, "pre": []}


def patch_qwen25():
    import torch
    import vllm.model_executor.models.qwen2_5_vl as qv
    cls = next(c for c in vars(qv).values() if isinstance(c, type) and c.__name__ == "Qwen2_5_VisionTransformer")
    orig = cls.forward

    def patched(self, *args, **kw):
        if GEN["t0"] is not None:
            GEN["pre"].append((time.perf_counter() - GEN["t0"]) * 1e3); GEN["t0"] = None
        if not getattr(self.merger, "_probed", False):
            mo = self.merger.forward

            def mw(*a, **k):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); r = mo(*a, **k); e.record(); torch.cuda.synchronize()
                STAGE["merger"].append(s.elapsed_time(e)); return r
            self.merger.forward = mw; self.merger._probed = True
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = orig(self, *args, **kw); e.record(); torch.cuda.synchronize()
        STAGE["vis"].append(s.elapsed_time(e)); return r
    cls.forward = patched
    print(f"[probe] qwen2.5: timing {cls.__name__}.forward + merger", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--videos-csv", default="final_videos_pin.csv")
    ap.add_argument("--video-id", default="5396384503")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--video-max-patches", type=int, default=768)
    ap.add_argument("--video-min-patches", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--out", default="results/figs/fig_stage_qwen25.png")
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from measure.stage_timing_vllm import build_video_request

    assert "H100" in torch.cuda.get_device_name(0)
    spec = load_models().models[a.model]
    patch_qwen25()
    from measure.preproj_vllm import video_token_id
    vtid = video_token_id(spec, "qwen2.5")          # compute ONCE at startup (avoid end-of-run hang)
    mmk = {"max_pixels": a.video_max_patches * 28 * 28, "min_pixels": a.video_min_patches * 28 * 28}
    proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **mmk)
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False,
              mm_processor_cache_gb=0, mm_processor_kwargs=mmk, max_num_seqs=a.batch,
              max_num_batched_tokens=32768, limit_mm_per_prompt={"video": 1})
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    spD = SamplingParams(temperature=0.0, max_tokens=a.decode_tokens, detokenize=False)

    row = next(r for r in csv.DictReader(open(a.videos_csv)) if r["video_id"] == a.video_id)
    req = build_video_request(proc, row["path"], n_frames=a.frames, with_metadata=False)[0]
    reqs = [req] * a.batch
    B = a.batch

    def run(sp, want_stage):
        for _ in range(a.warmup):
            llm.generate(reqs, sp)
        walls, pre_b, vis_b, mrg_b = [], [], [], []
        outs = None
        for _ in range(a.runs):
            STAGE["vis"].clear(); STAGE["merger"].clear(); GEN["pre"].clear()
            torch.cuda.synchronize(); t0 = time.perf_counter(); GEN["t0"] = t0
            outs = llm.generate(reqs, sp)
            torch.cuda.synchronize(); walls.append((time.perf_counter() - t0) * 1e3)
            if want_stage:
                pre_b.append(GEN["pre"][0] if GEN["pre"] else float("nan"))
                vis_b.append(sum(STAGE["vis"])); mrg_b.append(sum(STAGE["merger"]))
        return (statistics.median(walls),
                statistics.median(pre_b) if pre_b else 0,
                statistics.median(vis_b) if vis_b else 0,
                statistics.median(mrg_b) if mrg_b else 0, outs)

    ttft_w, pre_b, vis_b, mrg_b, out = run(sp1, True)
    full_w, *_ , _ = run(spD, False)
    n_vis = sum(1 for t in out[0].prompt_token_ids if t == vtid)
    # per-request (÷B)
    pre = pre_b / B; vis = vis_b / B; mrg = mrg_b / B
    enc = vis - mrg                                   # ViT blocks (encoding)
    ttft = ttft_w / B
    prefill = ttft - pre - vis                        # LLM prefill (remainder)
    tpot = (full_w - ttft_w) / a.decode_tokens        # batch wall decode / tokens; per-step latency
    segs = [("preprocessing\n(CPU pixel)", pre, "#9467bd"), ("encoding\n(ViT)", enc, "#4c72b0"),
            ("projector\n(merger)", mrg, "#2ca02c"), ("prefill\n(LLM)", prefill, "#dd8452")]
    print(f"\n[Qwen2.5 cold] {a.frames}f b{B} n_vis={n_vis}  ttft/req={ttft:.0f}ms")
    for nm, v, _ in segs:
        print(f"  {nm.split(chr(10))[0]:<16}{v:>8.1f} ms  ({v/ttft*100:>4.0f}% of TTFT)")
    print(f"  {'TPOT (decode)':<16}{tpot:>8.2f} ms/token")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.5, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    bottom = 0.0
    for nm, v, c in segs:
        axL.bar(0, v, bottom=bottom, color=c, width=0.6, label=nm.replace("\n", " "))
        if v > ttft * 0.03:
            axL.text(0, bottom + v / 2, f"{nm}\n{v:.0f}ms ({v/ttft*100:.0f}%)", ha="center", va="center", fontsize=8)
        bottom += v
    axL.set_xticks([]); axL.set_ylabel("TTFT breakdown (ms)")
    axL.set_title(f"Qwen2.5-VL cold TTFT\n{a.frames}f b{B}, n_vis={n_vis}  (total {ttft:.0f}ms)")
    axL.set_ylim(0, ttft * 1.08)
    axR.bar(0, tpot, color="#c44e52", width=0.5)
    axR.text(0, tpot, f"{tpot:.2f}\nms/tok", ha="center", va="bottom", fontsize=10)
    axR.set_xticks([]); axR.set_ylabel("TPOT (ms / output token)")
    axR.set_title(f"decoding\n(TPOT, {a.decode_tokens} tok)"); axR.set_ylim(0, tpot * 1.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
