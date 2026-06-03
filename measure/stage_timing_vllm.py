"""Layer 1 per-stage cost primitives via vLLM (ModServe SoCC'25 methodology).

COMPLEMENTS measure/stage_timing.py (transformers) — does NOT replace it. Same
primitives, measured through the engine the paper actually serves with.

Method: SINGLE request, batch=1, isolated (no batching, so stages don't overlap),
prefix caching OFF (baseline — else the 2nd identical request hits cache).

  T_ttft (encode+prefill) = wall-clock( generate(max_tokens=1) )      # time to first token
  T_full                  = wall-clock( generate(min=max=decode_tokens, ignore_eos) )
  T_decode                = T_full - T_ttft                            # the extra decode steps

vLLM FUSES vision encode into prefill (the ttft above). Two ways to split encode:
  - transformers path (stage_timing.py) measures encode standalone; but do NOT
    subtract it from vLLM's ttft — the engines have different per-stage speeds
    (vLLM's whole encode+prefill can be < transformers' encode alone), so the
    subtraction goes NEGATIVE. Keep the two engines' stage numbers side by side.
  - WITHIN vLLM (`--text-baseline`): time an equal-length TEXT-only prompt; its
    ttft is pure LLM prefill over N tokens, so T_encode ~= ttft_mm - ttft_text and
    T_prefill ~= ttft_text. Same engine -> valid.

We use wall-clock around blocking llm.generate() rather than vLLM's V1 metrics
object (not stably exposed in 0.22). At batch=1 the python/scheduler overhead is
small and constant; median over >=5 runs (warmups discarded) absorbs it.
detokenize=False so detok cost doesn't inflate decode. CUDA_VISIBLE_DEVICES=1.
peak_vram_gib is the engine's gpu_memory_utilization RESERVATION (vLLM pre-grabs
KV cache), not a working-set peak — use the transformers path for true peak VRAM.

Output: long-format CSV (one row per model,video,stage,run_idx) -> OUTPUT_DIR/
stage_timing_vllm.csv, so analysis can recompute median/IQR and join encode.

Usage:
  python -m measure.stage_timing_vllm --model qwen2.5-vl-7b --num-images 1 2 4
  python -m measure.stage_timing_vllm --model qwen2.5-vl-7b --videos-csv results/nextqa_sample.csv --frames 16
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402


def gpu_used_gib(phys_index: int = 1) -> float:
    """Process-wide GPU memory in use on the physical H100 (index 1), via nvidia-smi.
    vLLM's engine runs in a subprocess, so torch.cuda in this process can't see it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={phys_index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return round(int(out.stdout.strip()) / 1024, 3)
    except Exception:
        return float("nan")


def build_image_request(processor, n_images: int, resolution: int):
    import numpy as np
    from PIL import Image
    imgs = [Image.fromarray(np.uint8(np.random.rand(resolution, resolution, 3) * 255))
            for _ in range(n_images)]
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": "Describe in detail."})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt, "multi_modal_data": {"image": imgs}}, f"synthimg_R{resolution}_k{n_images}"


def build_video_request(processor, path: str, fps: float = 2.0, n_frames: int | None = None,
                        with_metadata: bool = False, query: str = "Describe in detail."):
    import decord
    import numpy as np
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path)
    vfps = float(vr.get_avg_fps())
    if n_frames is not None:                       # explicit frame count (for frame sweep)
        idx = np.linspace(0, len(vr) - 1, num=min(n_frames, len(vr))).round().astype(int)
    else:
        from measure.frames import qwen_frame_indices
        idx = qwen_frame_indices(len(vr), vfps, fps=fps)   # Qwen-native fps sampling
    frames = vr.get_batch(idx).asnumpy()           # (T, H, W, C) uint8
    video = frames
    if with_metadata:                              # Qwen3-VL needs metadata (we pre-sampled)
        video = (frames, {"total_num_frames": int(len(vr)), "fps": vfps,
                          "frames_indices": idx.tolist(), "do_sample_frames": False,
                          "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                          "duration": len(vr) / vfps, "video_backend": "decord"})
    content = [{"type": "video"}, {"type": "text", "text": query}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt, "multi_modal_data": {"video": video}}, f"{Path(path).stem}_{len(idx)}f"


def build_video_request_internvl(tokenizer, path: str, n_frames: int, query: str = "Describe in detail."):
    """InternVL via vLLM's native VIDEO path: a single `<video>` placeholder that
    vLLM expands to `<|video_pad|>` per patch (1 tile/frame for video). frames passed
    as a (T,H,W,C) array under multi_modal_data['video']."""
    import decord
    import numpy as np
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path)
    idx = np.linspace(0, len(vr) - 1, num=min(n_frames, len(vr))).round().astype(int)
    frames = vr.get_batch(idx).asnumpy()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<video>\n{query}"}],
        tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"video": frames}}, f"{Path(path).stem}_{len(idx)}f"


def mm_token_ids(repo_id: str, trust: bool) -> set[int]:
    """image/video placeholder token ids, to count n_vision_tokens in the prompt."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=trust)
    ids = set()
    for attr in ("image_token_id", "video_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int):
            ids.add(v)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-images", type=int, nargs="+", default=[1, 2, 4],
                    help="synthetic-image mode: image counts to sweep")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--videos-csv", help="real-video mode: nextqa_sample.csv (overrides image mode)")
    ap.add_argument("--fps", type=float, default=2.0, help="Qwen-native fps frame sampling (longer video -> more frames)")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="explicit frame-count SWEEP per video (overrides --fps); e.g. 8 16 32 64 128")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=40960,
                    help="must exceed largest n_vision_tokens (fps sampling on long video is large)")
    ap.add_argument("--cudagraph", action="store_true", help="allow CUDA graphs (default: enforce_eager)")
    ap.add_argument("--no-text-baseline", dest="text_baseline", action="store_false",
                    help="skip the equal-length text-only prompt. DEFAULT: ON — always record "
                         "the TTFT breakdown (T_encode ~= ttft_mm - ttft_text, T_prefill ~= ttft_text).")
    ap.set_defaults(text_baseline=True)
    a = ap.parse_args()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "must run on the H100 (GPU 1)"
    spec = load_models().models[a.model]

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    is_internvl = spec.key.startswith("internvl")
    if is_internvl:                                 # no HF processor; use tokenizer + manual <image> prompt
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        mm_ids = {tok.convert_tokens_to_ids("<|video_pad|>")}   # vLLM InternVL video context token
        filler_id = tok.encode("the")[-1]
        make_video = lambda path, nf, q: build_video_request_internvl(tok, path, nf, query=q)
    else:
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        mm_ids = mm_token_ids(spec.repo_id, spec.trust_remote_code)
        filler_id = processor.tokenizer.encode("the")[-1]  # any valid id; identical-token prefill fine for timing
        needs_meta = spec.key.startswith("qwen3")        # Qwen3-VL requires video metadata
        make_video = lambda path, nf, q: build_video_request(processor, path, n_frames=nf, with_metadata=needs_meta, query=q)

    # build the request list
    requests = []
    if a.videos_csv:
        assert a.frames, "video mode here uses --frames sweep"
        with open(os.path.expanduser(a.videos_csv)) as f:
            for row in csv.DictReader(f):
                dur = float(row.get("duration_s") or 0)
                q = (row.get("question") or "Describe in detail.").strip()   # REAL dataset query
                for nf in a.frames:                 # frame-count sweep: each N a separate request
                    requests.append(make_video(row["path"], nf, q) + (dur,))
    else:
        for k in a.num_images:
            requests.append(build_image_request(processor, k, a.resolution) + (None,))

    img_lim = max([8] + (a.frames or []) + (a.num_images or []))
    print(f"[vllm] loading {spec.repo_id} (prefix_caching=OFF, mm_cache=OFF, eager={not a.cudagraph}) ...")
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code,
              max_model_len=a.max_model_len, gpu_memory_utilization=0.85,
              enforce_eager=not a.cudagraph, enable_prefix_caching=False,
              mm_processor_cache_gb=0,              # disable mm cache: every request re-encodes (no contamination)
              limit_mm_per_prompt={"image": img_lim, "video": 1})

    sp_ttft = SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1,
                             ignore_eos=True, detokenize=False)
    sp_full = SamplingParams(temperature=0.0, min_tokens=a.decode_tokens,
                             max_tokens=a.decode_tokens, ignore_eos=True, detokenize=False)

    out_path = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "stage_timing_vllm.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fcsv = open(out_path, "a", newline="")
    fields = ["model", "video_id", "stage", "run_idx", "value_s", "n_vision_tokens",
              "prompt_tokens", "decode_tokens", "gen_tokens", "duration_s", "peak_vram_gib",
              "engine", "prefix_caching", "timestamp"]
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        writer.writeheader()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def timed(req, sp):
        t = time.perf_counter()
        o = llm.generate(req, sp, use_tqdm=False)[0]
        return time.perf_counter() - t, o

    for req, vid, dur in requests:
        # warmups (compile/alloc); also establishes prompt token count. A request whose
        # n_vis exceeds the model context overflows at the first generate here -> record
        # it as the single-H100 limit (caching-required point) and skip.
        last = None
        try:
            for _ in range(a.warmup):
                _, last = timed(req, sp_ttft)
        except Exception as e:
            writer.writerow({"model": spec.key, "video_id": vid, "stage": "overflow",
                             "run_idx": 0, "value_s": "", "n_vision_tokens": -1,
                             "prompt_tokens": -1, "decode_tokens": a.decode_tokens, "gen_tokens": "",
                             "duration_s": dur, "peak_vram_gib": "", "engine": "vllm",
                             "prefix_caching": False, "timestamp": ts})
            print(f"[vllm] {vid}: OVERFLOW/ERROR ({type(e).__name__}: {str(e)[:70]}) — recorded, skip")
            continue
        n_prompt = len(last.prompt_token_ids)
        n_vis = sum(1 for t in last.prompt_token_ids if t in mm_ids) if mm_ids else -1

        ttfts, fulls, gtoks = [], [], []
        for _ in range(a.runs):
            tt, _ = timed(req, sp_ttft)
            tf, of = timed(req, sp_full)
            ttfts.append(tt); fulls.append(tf)
            gtoks.append(len(of.outputs[0].token_ids))   # ACTUAL generated tokens (fair per-token)
        decs = [f - t for f, t in zip(fulls, ttfts)]
        vram = gpu_used_gib(1)

        def emit(stage, vals, gen):
            """gen: per-row generated-token count (list aligned to vals, or scalar)."""
            for i, v in enumerate(vals):
                writer.writerow({
                    "model": spec.key, "video_id": vid, "stage": stage, "run_idx": i,
                    "value_s": round(v, 6), "n_vision_tokens": n_vis, "prompt_tokens": n_prompt,
                    "decode_tokens": a.decode_tokens,
                    "gen_tokens": (gen[i] if isinstance(gen, list) else gen),
                    "duration_s": dur, "peak_vram_gib": vram,
                    "engine": "vllm", "prefix_caching": False, "timestamp": ts})
        emit("encode_prefill_ttft", ttfts, 1)   # encode+prefill (vLLM fuses encode); 1 token generated
        emit("decode", decs, [g - 1 for g in gtoks])   # decode steps = generated - the ttft token

        # WITHIN-vLLM encode/prefill split via an equal-length text-only prompt:
        #   ttft_text(N tokens) = pure LLM prefill over N tokens
        #   T_encode  ~= ttft_mm - ttft_text   ;   T_prefill ~= ttft_text
        # Valid because both stay in the same engine (unlike subtracting transformers' encode).
        enc_str = ""
        if a.text_baseline:
            text_req = {"prompt_token_ids": [filler_id] * n_prompt}
            for _ in range(a.warmup):
                timed(text_req, sp_ttft)
            tt_text = [timed(text_req, sp_ttft)[0] for _ in range(a.runs)]
            emit("prefill", tt_text, 1)
            emit("encode", [m - t for m, t in zip(ttfts, tt_text)], 1)
            enc_str = (f" prefill={statistics.median(tt_text)*1e3:.1f}ms "
                       f"encode={statistics.median([m-t for m,t in zip(ttfts,tt_text)])*1e3:.1f}ms")
        gen_med = statistics.median(gtoks)
        per_tok = statistics.median(decs) / max(gen_med - 1, 1)
        print(f"[vllm] {vid}: n_vis={n_vis} prompt={n_prompt} gen_tok={gen_med} "
              f"ttft={statistics.median(ttfts)*1e3:.1f}ms{enc_str} "
              f"decode_total={statistics.median(decs)*1e3:.1f}ms "
              f"({per_tok*1e3:.2f}ms/tok) vram={vram}GiB")

    fcsv.close()
    print(f"[vllm] wrote rows -> {out_path}")


if __name__ == "__main__":
    main()
