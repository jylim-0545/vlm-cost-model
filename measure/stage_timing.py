"""Layer 1 — per-stage cost primitives via transformers (CLAUDE.md Section 5).

Measures T_encode / T_prefill / T_decode + n_vision_tokens per model, batch=1,
each stage isolated with cuda events. NO DATASET NEEDED: prefill/decode time
depends on token COUNT, not content (Section 5), so we feed SYNTHETIC frames and
sweep the vision-token count. Decode length is forced fixed (min==max) so a
short/garbled output never pollutes timing.

Stage isolation (subtraction method — robust across model families):
  T_encode  = vision tower forward
  T_prefill = full first forward(use_cache=True) - T_encode
  T_decode  = generate(decode_tokens) - full first forward     (per-token = /decode_tokens)

Each model family supplies an Adapter exposing encode()/full_forward()/generate(),
because forward vs generate take different args (e.g. InternVL forward needs
image_flags but generate doesn't). Discards `--warmup` runs (first run pays
compile/alloc), reports MEDIAN over `--runs`, records peak VRAM. Writes one tidy
CSV row per (model, synth input); schema is what analyze/price_model.py consumes.

Token-count knob:
  Qwen      — `--num-images` images at `--resolutions` px (dynamic tokens).
  InternVL  — `--num-images` = number of 448px tiles (256 vision tokens each).

Usage:
  python -m measure.stage_timing --model qwen2.5-vl-7b --num-images 1 2 4
  python -m measure.stage_timing --model internvl3.5-8b --num-images 1 4 8
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402

DEV = "cuda:0"  # H100 after CUDA_VISIBLE_DEVICES=1 pin


def _cuda_time(fn):
    import torch
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); out = fn(); e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000.0, out


# --------------------------------------------------------------------------
# Per-family adapters. Each builds synthetic inputs and exposes the three stages
# as callables; the timing harness below is model-agnostic.
# --------------------------------------------------------------------------
class Adapter:
    family = "base"

    def __init__(self, spec):
        self.spec = spec

    def load(self):
        raise NotImplementedError

    def build_inputs(self, n_images: int, resolution: int, batch: int) -> tuple[int, int]:
        """Build + store a BATCH of `batch` identical synthetic inputs.
        Return (n_vision_tokens PER REQUEST, input_len)."""
        raise NotImplementedError

    def build_video_inputs(self, frames, batch: int) -> tuple[int, int]:
        """Build + store a BATCH of `batch` identical REAL-video inputs (frames =
        np array [T,H,W,C]). Return (n_vision_tokens PER REQUEST, input_len)."""
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError          # vision tower forward (T_encode)

    def full_forward(self):
        raise NotImplementedError          # encode + prefill (single forward)

    def generate(self, decode_tokens: int):
        raise NotImplementedError          # full forward + decode_tokens decode steps

    def effective_resolution(self, resolution: int) -> int:
        return resolution


class QwenVLAdapter(Adapter):
    """Qwen2.5-VL & Qwen3-VL: model.model.visual; keep mm_token_type_ids (Qwen3 M-RoPE)."""
    family = "qwen-vl"

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.spec.repo_id, trust_remote_code=self.spec.trust_remote_code)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.spec.repo_id, dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=self.spec.trust_remote_code).to(DEV).eval()
        self.visual = self.model.model.visual
        self.image_token_id = self.model.config.image_token_id
        self.video_token_id = getattr(self.model.config, "video_token_id", None)

    def build_inputs(self, n_images: int, resolution: int, batch: int) -> tuple[int, int]:
        import numpy as np
        from PIL import Image
        imgs = [Image.fromarray(np.uint8(np.random.rand(resolution, resolution, 3) * 255))
                for _ in range(n_images)]
        content = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": "Describe in detail."})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        # batch of `batch` identical samples; identical lengths -> no padding needed
        raw = self.processor(text=[text] * batch, images=imgs * batch,
                             return_tensors="pt", padding=True).to(DEV)
        self._fwd = dict(raw)              # incl mm_token_type_ids (Qwen3 needs it; Qwen2.5 tolerates)
        self._enc = (raw["pixel_values"], raw["image_grid_thw"])
        n_vis_total = int((raw["input_ids"] == self.image_token_id).sum())
        return n_vis_total // batch, int(raw["input_ids"].shape[1])

    def build_video_inputs(self, frames, batch: int) -> tuple[int, int]:
        content = [{"type": "video"}, {"type": "text", "text": "Describe in detail."}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        raw = self.processor(text=[text] * batch, videos=[frames] * batch,
                             return_tensors="pt", padding=True).to(DEV)
        self._fwd = dict(raw)
        self._enc = (raw["pixel_values_videos"], raw["video_grid_thw"])
        n_vis_total = int((raw["input_ids"] == self.video_token_id).sum())
        return n_vis_total // batch, int(raw["input_ids"].shape[1])

    def encode(self):
        return self.visual(self._enc[0], grid_thw=self._enc[1])

    def full_forward(self):
        return self.model(**self._fwd, use_cache=True)

    def generate(self, decode_tokens: int):
        return self.model.generate(**self._fwd, min_new_tokens=decode_tokens,
                                   max_new_tokens=decode_tokens, do_sample=False)


class InternVLAdapter(Adapter):
    """InternVL3.5 (OpenGVLab custom `internvl_chat`). Manual inputs: pixel_values are
    448px tiles, IMG_CONTEXT tokens (256/tile) get vision features injected. Needs a
    transformers-5.9 compat shim (custom code predates `all_tied_weights_keys`)."""
    family = "internvl"
    IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"

    def load(self):
        import torch
        import transformers
        # compat shim: transformers 5.9 sets all_tied_weights_keys in post_init(), which the
        # older custom modeling never calls. Empty default lets loading proceed — weight TYING
        # is irrelevant for timing (stage cost depends on shapes/counts, not values).
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            transformers.PreTrainedModel.all_tied_weights_keys = {}
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.spec.repo_id, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
        self.ctx_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CTX)
        self.model.img_context_token_id = self.ctx_id
        self.num_image_token = int(self.model.num_image_token)  # 256 for these configs
        self.image_size = int(self.model.config.force_image_size or
                              self.model.config.vision_config.image_size)

    def build_inputs(self, n_images: int, resolution: int, batch: int) -> tuple[int, int]:
        import torch
        if batch != 1:
            raise NotImplementedError(
                "InternVLAdapter batch>1 not implemented yet (custom forward needs batched "
                "image_flags/input_ids reshape) — measure InternVL at batch=1 for now.")
        P = n_images                       # number of 448px tiles
        S = self.image_size
        self._pixel_values = torch.rand(P, 3, S, S, dtype=torch.bfloat16, device=DEV)
        self._image_flags = torch.ones(P, 1, dtype=torch.long, device=DEV)
        img_block = self.IMG_START + self.IMG_CTX * (self.num_image_token * P) + self.IMG_END
        text = img_block + "\nDescribe in detail."
        enc = self.tokenizer(text, return_tensors="pt").to(DEV)
        self._input_ids = enc["input_ids"]
        self._attn = enc["attention_mask"]
        n_vis = int((self._input_ids == self.ctx_id).sum())
        assert n_vis == self.num_image_token * P, f"{n_vis} != {self.num_image_token * P}"
        return n_vis, int(self._input_ids.shape[1])

    def build_video_inputs(self, frames, batch: int) -> tuple[int, int]:
        # InternVL treats a video as num_frames 448px tiles. Pixel VALUES don't affect
        # timing (depends on shape/count), so reuse the tile path with P = num_frames.
        return self.build_inputs(len(frames), self.image_size, batch)

    def encode(self):
        return self.model.extract_feature(self._pixel_values)

    def full_forward(self):
        return self.model(pixel_values=self._pixel_values, input_ids=self._input_ids,
                          attention_mask=self._attn, image_flags=self._image_flags,
                          use_cache=True)

    def generate(self, decode_tokens: int):
        return self.model.generate(pixel_values=self._pixel_values, input_ids=self._input_ids,
                                   attention_mask=self._attn, min_new_tokens=decode_tokens,
                                   max_new_tokens=decode_tokens, do_sample=False)

    def effective_resolution(self, resolution: int) -> int:
        return self.image_size


ADAPTERS = {
    "qwen2.5-vl-7b": QwenVLAdapter,
    "qwen3-vl-8b": QwenVLAdapter,
    "internvl3.5-8b": InternVLAdapter,
    "internvl3.5-14b": InternVLAdapter,
}


@dataclass
class Row:
    model: str
    video_id: str
    n_vision_tokens: int
    t_encode_s: float
    t_prefill_s: float
    t_decode_s: float
    duration_s: float | None
    batch: int
    resolution: int
    num_images: int
    input_len: int
    decode_tokens: int
    n_runs: int
    t_encode_iqr_s: float
    t_prefill_iqr_s: float
    t_decode_iqr_s: float
    peak_vram_gib: float
    attn_impl: str
    transformers_version: str
    timestamp: str


def _iqr(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def _time_and_row(adapter: Adapter, *, video_id, n_vis, input_len, duration_s,
                  resolution, num_images, decode_tokens, warmup, runs, batch) -> Row:
    """Shared timing loop. Inputs must already be built on the adapter. Stages are
    timed for the whole BATCH; divide by `batch` for PER-REQUEST cost."""
    import torch
    torch.cuda.reset_peak_memory_stats()
    enc, pre, dec = [], [], []

    @torch.inference_mode()
    def one():
        t_enc, _ = _cuda_time(adapter.encode)
        t_ep, _ = _cuda_time(adapter.full_forward)
        t_gen, _ = _cuda_time(lambda: adapter.generate(decode_tokens))
        return t_enc / batch, (t_ep - t_enc) / batch, (t_gen - t_ep) / batch

    for i in range(warmup + runs):
        te, tp, td = one()
        if i >= warmup:
            enc.append(te); pre.append(tp); dec.append(td)

    return Row(
        model=adapter.spec.key, video_id=video_id, n_vision_tokens=n_vis,
        t_encode_s=statistics.median(enc), t_prefill_s=statistics.median(pre),
        t_decode_s=statistics.median(dec), duration_s=duration_s, batch=batch,
        resolution=resolution, num_images=num_images, input_len=input_len,
        decode_tokens=decode_tokens, n_runs=runs,
        t_encode_iqr_s=_iqr(enc), t_prefill_iqr_s=_iqr(pre), t_decode_iqr_s=_iqr(dec),
        peak_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        attn_impl="sdpa",
        transformers_version=__import__("transformers").__version__,
        timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )


def measure(adapter: Adapter, n_images: int, resolution: int,
            decode_tokens: int, warmup: int, runs: int, batch: int) -> Row:
    n_vis, input_len = adapter.build_inputs(n_images, resolution, batch)
    R = adapter.effective_resolution(resolution)
    return _time_and_row(adapter, video_id=f"synth_R{R}_k{n_images}", n_vis=n_vis,
                         input_len=input_len, duration_s=None, resolution=R,
                         num_images=n_images, decode_tokens=decode_tokens,
                         warmup=warmup, runs=runs, batch=batch)


def measure_video(adapter: Adapter, frames, video_id: str, duration_s, n_frames: int,
                  decode_tokens: int, warmup: int, runs: int, batch: int) -> Row:
    n_vis, input_len = adapter.build_video_inputs(frames, batch)
    R = adapter.effective_resolution(0)
    return _time_and_row(adapter, video_id=video_id, n_vis=n_vis, input_len=input_len,
                         duration_s=duration_s, resolution=R, num_images=n_frames,
                         decode_tokens=decode_tokens, warmup=warmup, runs=runs, batch=batch)


def _out_path() -> Path:
    d = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results")))
    d.mkdir(parents=True, exist_ok=True)
    return d / "stage_timing.csv"


def append_csv(rows: list[Row]) -> Path:
    path = _out_path()
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[448])
    ap.add_argument("--num-images", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--videos-csv", help="real-video mode: nextqa_sample.csv (overrides synthetic)")
    ap.add_argument("--fps", type=float, default=2.0, help="Qwen-native fps frame sampling (longer video -> more frames)")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1,
                    help="batch regime (per-request cost = batch time / batch). "
                         "Each batch is a SEPARATE run — not linear. Qwen only; InternVL batch=1.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "must run on the H100 (GPU 1)"
    spec = load_models().models[a.model]
    adapter = ADAPTERS[a.model](spec)
    print(f"[stage] loading {spec.repo_id} ...")
    adapter.load()
    print(f"[stage] loaded {type(adapter.model).__name__}")

    def show(row):
        print(f"[stage] {row.video_id} b={row.batch}: n_vis={row.n_vision_tokens} "
              f"enc={row.t_encode_s*1e3:.1f}ms pre={row.t_prefill_s*1e3:.1f}ms "
              f"dec={row.t_decode_s*1e3:.1f}ms ({row.t_decode_s/row.decode_tokens*1e3:.2f}ms/tok) "
              f"vram={row.peak_vram_gib}GiB")

    rows = []
    if a.videos_csv:                       # real-video mode
        import csv as _csv
        import decord
        decord.bridge.set_bridge("native")
        from measure.frames import qwen_frame_indices
        with open(os.path.expanduser(a.videos_csv)) as f:
            for d in _csv.DictReader(f):
                vr = decord.VideoReader(d["path"])
                idx = qwen_frame_indices(len(vr), vr.get_avg_fps(), fps=a.fps)
                frames = vr.get_batch(idx).asnumpy()
                vid = f"{Path(d['path']).stem}_{len(idx)}f"   # match stage_timing_vllm
                dur = float(d["duration_s"]) if d.get("duration_s") else None
                row = measure_video(adapter, frames, vid, dur, len(idx),
                                    a.decode_tokens, a.warmup, a.runs, a.batch)
                rows.append(row); show(row)
    else:                                  # synthetic-image mode
        for R in a.resolutions:
            for k in a.num_images:
                row = measure(adapter, k, R, a.decode_tokens, a.warmup, a.runs, a.batch)
                rows.append(row); show(row)
    path = append_csv(rows)
    print(f"[stage] wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
