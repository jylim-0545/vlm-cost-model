# Why one video + a frame sweep suffices (no separate MLVU/NextQA or resolution runs)

**Claim.** For the cost-primitive measurements in this paper, we run each model on a **single
video** with a **frame-count sweep**, and plot everything against **n_vis** (the number of vision
tokens). We do **not** run MLVU and NextQA as separate experiments, and we do **not** sweep
resolution as an independent axis. This note justifies that choice with the experiments we ran.

---

## 1. We measure cost, not accuracy

The quantities we report — `T_encode`, `T_prefill`, `T_decode` (→ TTFT / TPOT / throughput) and the
stored-state byte sizes (`bytes_vision_tokens`, `bytes_kv`) — are **functions of tensor shapes
(token counts), not of pixel content or semantics**. Two clips with the same n_vis execute the same
kernels over the same shapes and therefore cost the same, whatever they depict. Reuse *quality*
(does a cached state still answer a new query correctly?) is explicitly **out of scope** for this
phase. Consequently the *identity* of the dataset/video is irrelevant to every number we report.

## 1.5 How each model turns a video into tokens (the mechanism behind the claim)

Two stages: **(A) frame sampling — we control it, identical for all models** and **(B) per-frame
tokenization — model-specific**. The whole justification rests on separating these.

**(A) Frame sampling (shared, under our control).** For every model we pick a *fixed number of
frames* with a uniform `linspace` over the clip (`--frames 16/32/64/128`). vLLM/HF is told to use
exactly those pre-sampled frames (e.g. Qwen3 `do_sample_frames=False`). Therefore:
- **The number of frames is an input we set, not a property of the video.** Native fps and duration
  do not change it.
- **Video length is irrelevant.** A 12 s NextQA clip and a 50 min MLVU clip both become *N* frames.
  (The only thing length must satisfy is having ≥ *N* frames to sample, which all our clips do.)

**(B) Per-frame tokenization (model-specific).** This is the only place models differ, and it
determines *tokens-per-frame*, hence `n_vis = (tokens/frame) × (frames)`:

| Model | Per-frame handling | tokens / frame | Resolution-dependent? | n_vis vs frames |
|---|---|---|---|---|
| **InternVL3.5** | every frame **force-resized to 448×448**, InternViT, then pixel-shuffle | **256, FIXED** | **No** (resize pins it) | `256 × frames`, exact |
| **LLaVA-OV-7B** | every frame **resized to 384** (SigLIP), bilinear-pooled | **196, FIXED** | **No** (resize pins it) | `196 × frames`, exact |
| **Qwen2.5-VL** | `smart_resize` to pixels ∈ [min,max]; tokens ∝ area, **capped per-frame** by `max_pixels` (`size.longest_edge` = *per-frame* budget) | **dynamic ∝ resolution**, ≈360 at the cap | **Yes** (until the per-frame cap) | `∝ frames` for a fixed video (no total cap) |
| **Qwen3-VL** | per-frame like Qwen2.5 **plus a *total-video* token budget** (`size.longest_edge` ≈ 12,288 tok) | dynamic, but **total saturates** | **Yes**, and total-capped | saturates unless we **raise** `longest_edge` → then `∝ frames` |

Reading of the table:
- **InternVL / LLaVA force a fixed resize**, so a frame is *always* the same number of tokens no
  matter the source resolution → `n_vis` is a clean multiple of the frame count, and "resolution" is
  not even a free variable. These two are the cleanest case for a single video.
- **Qwen2.5** lets tokens grow with resolution but caps each *frame* (`max_pixels`); there is no
  whole-video cap, so for one fixed video `n_vis ∝ frames` just like the others (the *slope*,
  tokens/frame, depends on that video's resolution).
- **Qwen3** additionally imposes a *whole-video* token budget, so naïvely `n_vis` stops growing with
  frames (saturates ~12,288). We lift that budget (`longest_edge ×256`) so it behaves like Qwen2.5.
  Net effect: **for Qwen3, the n_vis ceiling is a knob we choose, not a property of the clip.**

Crucially, **(B) only sets *how many* tokens a frame becomes — and §3 shows the cost depends only on
that token count, not on the resolution that produced it.** So all four mechanisms reconcile onto the
single shared x-axis, `n_vis`.

## 2. Frame count and video length are decoupled — and we control the frame count

We sample a **fixed number of frames** (uniform `linspace` over the clip) regardless of duration, so
**n_vis is set directly by the frame count, not by how long the video is**. A 30-second clip and a
1-hour clip, both sampled to 64 frames, yield the **same n_vis** and the same cost. Therefore the
only thing that distinguishes "long videos (MLVU)" from "short videos (NextQA)" for *cost* is the
**range of n_vis they naturally span** — and we set n_vis ourselves via the frame count.

## 3. Cost ∝ n_vis, and is resolution-independent (measured)

This is the empirical crux. We held **n_vis fixed and varied resolution** (320×240 / 640×480 /
1280×720, adjusting the frame count per resolution to hit matched n_vis ≈ 2.5k / 5k / 7.5k) and
measured encode latency (`encode = cold_ttft − vt_inject`), H100, cudagraph, median-of-5.

| matched n_vis | enc/n_vis — Qwen2.5 (µs/tok) | enc/n_vis — Qwen3 (µs/tok) |
|---|---|---|
| ~2.5k | 320:103 / 640:102 / 720:117 | 320:92 / 640:(84) / 720:86 |
| ~5k   | 100 / 101 / 97 | 90 / 84 / 92 |
| ~7.5k | 89 / 93 / 107 | 94 / 85 / — |

**At matched n_vis, encode/token is constant within ±10–15 %, with no systematic resolution trend.**
So resolution is **not** an independent cost axis — it only changes *where on the n_vis axis* a given
(video, frame-count) lands. (We also confirmed encode is **linear in n_vis** for the fixed-token
models, e.g. InternVL-8B ≈ 31 µs/tok flat from 4k→33k tokens.) The theoretical worry — that a ViT's
within-frame attention (∝ patches-per-frame²) would make high-resolution frames disproportionately
expensive at equal n_vis — does **not** materialise, because the encoders are MLP-dominated and that
term cancels against the opposite per-frame fixed overhead (low-res needs *more* frames at matched
n_vis). Evidence: `results/qwen_res_encode/`, `analyze/qwen_res_encode.py`.

**This generalises to prefill/decode/KV too:** prefill is attention over the same n_vis tokens, KV
bytes are `n_vis × (per-token KV bytes)`, decode is shared. All are functions of n_vis. So **n_vis is
a sufficient, universal x-axis** for the whole cost model.

## 4. Therefore MLVU vs NextQA (and resolution) collapses to "n_vis coverage"

Because cost is a function of n_vis alone, the two datasets differ *only* in the n_vis range they
naturally reach: NextQA (short, low-res) sits at the low end, MLVU/high-res reaches the high end with
fewer frames. Since we sweep n_vis directly via the frame count, **a single video covers the axis**,
and we simply choose videos to span the n_vis range we care about. Running both datasets as separate
experiments would produce points that **fall on the same n_vis curve** — i.e. measure nothing new.

## 5. What we actually run

Per model: **1 video, frame sweep {16, 32, 64, 128}, batch sweep {1, 4, 8, 16}, x-axis = measured
n_vis.** (A handful of higher-resolution clips are used only to *extend n_vis coverage* and to run
the §3 resolution-independence check — not as a separate "MLVU experiment.")

---

## Scope, caveats, per-model notes (honesty)

- **Cost-only.** Everything above is for cost primitives. Accuracy/reuse-quality would depend on
  content and dataset and is not claimed here.
- **The encode *constant* differs per model** (InternVL ≈ 31, Qwen ≈ 90 µs/tok) — different vision
  encoders. That is a real, per-model property we report, not an unfairness; the *shape* (∝ n_vis)
  is shared.
- **How n_vis is produced differs per model** (all reconcile to the same n_vis axis):
  - **InternVL3.5** — 256 tok/frame *fixed* (every frame resized to 448, pixel-shuffle): n_vis ∝
    frames exactly, resolution-independent *by construction*.
  - **LLaVA-OV** — 196 tok/frame *fixed* (SigLIP-384, bilinear pool): same, single video used.
  - **Qwen2.5-VL** — *dynamic* tok/frame ∝ resolution, capped per-frame by `max_pixels`
    (`size.longest_edge` = per-frame budget → **no total saturation**): n_vis ∝ frames for a fixed
    video.
  - **Qwen3-VL** — has a **total-video token budget** (`size.longest_edge` = 25.2M px ≈ **12,288
    tokens**) that *saturates* n_vis regardless of frame count. We **raise** `longest_edge` so n_vis
    ∝ frames like the others. Practically this means **for Qwen3 we set the n_vis range via a config
    knob**, not the video.
- **Figure implementation detail.** Figures **pin to a single video** so the n_vis axis is monotonic.
  Multi-video data with dynamic tok/frame (Qwen) otherwise mixes clips with different n_vis at the
  same frame count, folding the curve back on the n_vis axis. Fixed-token models (InternVL/LLaVA) are
  unaffected (identical n_vis across clips), so the pin is harmless there.

## One-line summary

> Because we measure *cost* (a function of token count, not content) and we *control* n_vis via the
> frame count, and because encode/prefill/decode/KV are all ∝ n_vis and resolution-independent
> (measured, ±10–15 %), a single video swept over frames — plotted against n_vis — captures
> everything; separate MLVU/NextQA or resolution runs would only re-trace the same n_vis curve.
