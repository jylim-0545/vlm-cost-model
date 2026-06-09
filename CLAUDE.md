# CLAUDE.md — VLM Vision-Token Caching Cost Model

## 0. What this project is (read first)

We are writing a **systems / cost-modeling paper** (target: HotStorage, a short
workshop paper). The research question is:

> For VLM video analytics, **when is it cheaper to STORE intermediate inference
> state (vision tokens / KV cache) and reuse it, versus RE-COMPUTING it on every
> query?**

The output of THIS phase is **measurements + a price model**, not a full serving
system. We are NOT building a production cache, NOT solving reuse-accuracy, NOT
implementing eviction/tiering. We only need to MEASURE cost primitives and
compute break-even points.

We compare three "price models":
1. **baseline** — no reuse; every query recomputes encode + prefill + decode.
2. **KV reuse** — store the KV cache; reuse skips encode + prefill (decode only).
3. **vision-token reuse** — store vision tokens; reuse skips encode (prefill still runs).

> **⚠️ 2026-06-09 PIVOT — read §11.** The measurement now uses a **3-way, all REAL-measured
> in vLLM**: (1) **cold** = vLLM full recompute; (2) **kv_reuse** = **LMCache**-based (real KV
> offload+reload — the retrieval we used to COMPUTE is now MEASURED); (3) **vt_reuse** =
> **vLLM-based PRE-projector** (reuse the ENCODER/ViT output, skip ViT, re-run the cheap
> projector) — or **POST-projector via EC** where pre is impossible (Qwen3 DeepStack). §5's
> reuse_real (inject-based post-projector vt + warm-cache kv) is the PRIOR phase; §11 supersedes
> the mechanism. The economic question is unchanged.

The economic intuition we are testing:
- Storage is cheap and roughly constant over time; recompute is expensive and
  recurs on every access. So if a video is queried N times, the per-access
  recompute saving accumulates while storage cost stays ~flat.
- We expect **vision-token reuse to almost always pay off** (tokens are small),
  but **KV reuse to be risky for long videos** because KV is ~8-18x larger than
  vision tokens and the retrieval bandwidth from storage may exceed prefill cost.

Everything below exists to measure the inputs to that price model.

---

## 1. CRITICAL: division of labor (do not cross this line)

**The human has ALREADY set up everything before you start: conda env, PyTorch,
vLLM, transformers, all model downloads, and all datasets are done. You (Claude
Code) write code only.** The env is a conda environment, already activated.

DO NOT, under any circumstance, run or generate commands that:
- create/modify the conda env, or run `conda install` / `pip install`
  (everything is installed; if an import fails, REPORT it precisely — which
  package/version — do NOT try to install or "fix" it yourself)
- download model weights or datasets (no `huggingface-cli download`, no dataset
  pulls, no `wget`/`curl` — they are already on disk at the Section 4 paths)
- check or modify NVIDIA drivers / CUDA / system packages
- launch long-running jobs that consume GPU for more than a quick smoke test

If a task seems to require any of the above, **STOP and tell the human exactly
what is missing or failing**, then wait. Everything is assumed present at the
Section 4 paths. If a path is empty or an import errors, say so precisely and
stop — do not attempt to remediate.

You MAY: write Python/shell scripts, edit code, write small unit-test stubs, run
*fast* sanity checks (e.g. `python -c "import torch; print(torch.__version__)"`)
in the already-active env.

---

## 2. Hardware & environment (assume, do not set up)

- This server has **TWO GPUs**:
  - GPU 0: NVIDIA RTX PRO 6000 Blackwell (97 GB) — **DO NOT USE**
  - GPU 1: NVIDIA **H100 PCIe** (81 GB) — **USE THIS ONE ONLY**
- **ALWAYS pin to the H100** with `CUDA_VISIBLE_DEVICES=1` on every run (and set
  it in scripts via env, not hardcoded device indices). All cost numbers are
  normalized to H100; never let a job touch GPU 0 or it silently pollutes
  measurements. After `CUDA_VISIBLE_DEVICES=1`, the H100 appears as `cuda:0`
  inside the process — that's expected.
- **GPU 1 is SHARED with another user (`ljh`).** Before a timing run, check it's
  idle: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  (concurrent use contaminates timing — likely caused earlier decode-ms instability).
  **Killing a vLLM run does NOT kill its `VLLM::EngineCore` subprocess** — it orphans
  and holds ~all GPU memory. After stopping a vLLM job, find the lingering
  `VLLM::EngineCore` PID via the same query and `kill -9` it (verify it's ours with
  `ps -p <pid> -o cmd`, NOT the other user's).
- Driver 595.71.05, CUDA 13.2. CONFIRMED WORKING: torch **2.11** (cu13 build, pulled
  in by vLLM) sees the H100 fine (`cuda.is_available()` True, device name
  "NVIDIA H100 PCIe"). Do not try to upgrade torch — vLLM is compiled against
  2.11 and changing it breaks vLLM kernels. Do not touch CUDA/torch yourself.
- Conda env: **`vlmcost`** (Python 3.11), activated by the human before any run.
- **Primary inference engine: vLLM 0.22.0** (CONFIRMED installed). It's what
  comparable MLLM serving papers use (ElasticMM, EPD), so **write code against
  vLLM by default** — throughput/batching, reuse via prefix caching, and the
  break-even numbers all come from vLLM. **transformers 5.9.0** (CONFIRMED) is a
  **fallback ONLY** for primitives vLLM structurally cannot expose — chiefly
  splitting `T_encode` out of prefill (vLLM fuses stages via continuous batching
  + chunked prefill). Do not reach for transformers when vLLM can already give
  the number. See Section 5 for the exact split.
- **⚠️ vLLM IS LOCALLY PATCHED (2026-06-05):** `model_executor/models/llava_onevision.py`
  has an ADDED `video_embeds` injection path (for LLaVA-OV vt_reuse — see Section 3 LLaVA
  quirks). Backup: `llava_onevision.py.bak_videoembeds`; reproducible diff:
  `patches/llava_onevision_videoembeds.patch`. **If the env/vLLM is ever reinstalled, RE-APPLY
  this patch** or LLaVA vt_reuse breaks. Other models/files are stock vLLM. This is the only
  source-level vLLM edit; don't touch CUDA/torch/kernels (Section 2 rules still hold).
- vLLM multimodal prefix caching is enabled by default in V1 and is hash-keyed on
  image content, so the same (video+query) hits cache on the 2nd request —
  REMEMBER to disable it (`enable_prefix_caching=False`) when measuring the
  no-reuse baseline.

### Disk layout (THIS MATTERS for measurement validity)
Local disks are tight (`/home` ~55 GB free, `/` ~311 GB free); a large NFS mount
`/mnt/nas` (~12 TB free) holds the bulk.
- **Model weights**: may live on `/mnt/nas` (loaded once into VRAM, then NFS is
  irrelevant — does NOT pollute inference timing). Slow first load is the only cost.
- **Video datasets**: live on `/mnt/nas` (size). BUT when timing CPU
  decode/preprocess, **copy the video to a LOCAL disk first** (or record that the
  number includes NFS read latency) — otherwise C_preprocess is polluted by NFS
  bandwidth, not real decode cost.
- **Stored vision-tokens / KV**: we do NOT measure retrieval time, so their disk
  location does not affect any reported number — retrieval is computed from
  `read_bytes` + config bandwidth/egress per storage tier (Sections 5 & 7). If you
  run the OPTIONAL disk->GPU load sanity check, do it on LOCAL NVMe (not NFS) so the
  check reflects a real fast tier; but it feeds nothing in the price model.
- **Results (CSV/JSON)**: anywhere local (small).

---

## 3. Target models (decoder-only VLMs, ~7-14B, sizes matched)

**6 target models** spanning backbone scale (4B–14B) and KV/token byte ratio (8×–29×).
(LLaVA-OneVision was dropped 2026-06-02, then RE-ADDED 2026-06-03 once confirmed vLLM
0.22 supports `LlavaOnevisionForConditionalGeneration` — see LLaVA quirks below.)

| Model | LLM backbone | hidden | layers | kv_heads | head_dim | vision encoder | KV/token ratio |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B   | Qwen2.5-7B | 3584 | 28 | 4 | 128 | custom ViT ~670M (window attn) | 8× |
| Qwen3-VL-8B     | Qwen3-8B   | 4096 | 36 | 8 | 128 | SigLIP2-SO-400M (+DeepStack)   | 18× |
| InternVL3.5-4B  | Qwen3-4B   | 2560 | 36 | 8 | 128 | InternViT-300M (pixel-shuffle) | **29×** |
| InternVL3.5-8B  | Qwen3-8B   | 4096 | 36 | 8 | 128 | InternViT-300M (pixel-shuffle) | 18× |
| InternVL3.5-14B | Qwen3-14B  | 5120 | 40 | 8 | 128 | InternViT-300M (pixel-shuffle) | 16× |
| LLaVA-OV-7B     | Qwen2-7B   | 3584 | 28 | 4 | 128 | SigLIP-384                     | 8× |

(All dims confirmed from `config.json` on disk. InternVL-4B has the HIGHEST ratio:
small LLM hidden (2560→5KB vision token) but same KV structure as 8B (144KB) → 29×.
LLaVA-OV-7B shares Qwen2-7B dims with Qwen2.5-VL-7B → identical bytes/ratio (8×); it
differs only in vision encoder (SigLIP) and video tokenization, isolating the
encode/token-count axis at fixed KV structure.)

**HF repo ids (downloaded by human, under HF_HOME=/mnt/nas/VLM/hf), all in
config/models.yaml — case matters (HF cache dir is case-sensitive):**
- `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`
- `OpenGVLab/InternVL3_5-4B` — **CAPITALIZED** (the complete cache dir), trust_remote_code
- `openGVLab/InternVL3_5-8B` — **lowercase** (capitalized 8B dir is incomplete), trust_remote_code
- `openGVLab/InternVL3_5-14B` — **lowercase**, trust_remote_code
- `llava-hf/llava-onevision-qwen2-7b-ov-hf` — NO trust_remote_code (native vLLM/transformers)

**30B MoE added on disk (2026-06-09, LONG-TERM target — see §11):** `OpenGVLab/InternVL3_5-30B-A3B-Instruct`
and `Qwen/Qwen3-VL-30B-A3B-Instruct` (A3B = 30B total / 3B active). Fit H100 (~60GB weights, smaller 96KB
KV than 14B); compute cheap (active 3B) → a stress-test of "store-vs-recompute" in the compute-cheap regime
(cheap compute → reuse pays off LESS, esp. kv_reuse whose advantage is skipping the now-cheap prefill).
NOT yet measured.

NOTE: Qwen3-VL (Sept 2025) and InternVL3.5 (Aug 2025) are recent; they require a
new enough transformers (confirmed: transformers 5.9.0 in env). If a model fails
to load with an "unknown architecture" or similar error, REPORT the exact
transformers version and error — do not try to upgrade/patch. The human resolves
version issues.

**vLLM video input — per-model quirks (PRIMARY engine; see reuse_real.py /
stage_timing_vllm.py):**
- **Qwen2.5-VL**: `multi_modal_data={"video": frames}` (np (T,H,W,C)) — frames as-is.
  Tokens/frame are **DYNAMIC (∝ resolution)**: 640×360→149, 720p→598, 4K→3952 tok/frame
  → high-res MLVU overflows context. **Cap with VIDEO_MAX/MIN_PIXELS** (768/128 ·28·28,
  Qwen-native) via `mm_processor_kwargs` on the engine AND `max_pixels=` on the HF
  processor (must match) → ~360 tok/frame, uniform across resolutions. For Qwen2.5 the
  video processor's `size.longest_edge` = the max_pixels we pass (PER-FRAME cap), so n_vis
  ∝ frames with NO total-video saturation (unlike Qwen3 — see longest_edge note below).
- **Qwen3-VL**: REQUIRES video metadata → `{"video": (frames, {total_num_frames, fps,
  frames_indices, do_sample_frames:False, width, height, duration, video_backend})}`.
  cold/kv/**vt_reuse all RESOLVED 2026-06-07 — NO vLLM patch needed** (unlike LLaVA). The
  vt n_vis "mismatch" (cold 1760 vs vt 440 @16f) was a `reuse_real.py` request-construction
  bug, NOT vLLM: **1760 is CORRECT** (grid_t=8=16/temporal_patch2; grid[8,22,40] prod/4=1760).
  Three vt-inject fixes: (1) pass `video_metadata=[{...}]`+`do_sample_frames=False` to the HF
  `proc()` (frames-only re-sampled to grid_t=2 → 440); (2) embeds dim = `out_hidden_size*(1+
  len(deepstack_visual_indexes))` = 4096·4 = **16384** (DeepStack; inject skips the tower so
  must match its output dim), not HID; (3) timestamps **NESTED** `[[0..t-1]]` (it's
  `MultiModalFieldConfig.batched("video")` → outer list = per-video batch; flat → "Cannot merge
  batch sizes"). Validated H100: vt<cold at 16f (152<204) & 64f (524<1059), n_vis aligned
  (1760/7040), encode saving super-linear — matches InternVL/LLaVA/Qwen2.5.
- **⚠️ Qwen3 n_vis SATURATES — `size.longest_edge` is a TOTAL-video token budget (we set it):**
  Unlike Qwen2.5 (per-frame cap), Qwen3's video processor `size.longest_edge=25,165,824 px`
  is a TOTAL budget → n_vis SATURATES at ~12,288 tok (=25165824/16²/4/2) regardless of frames
  (more frames → processor shrinks per-frame res). So **for Qwen3, WE set max n_vis** via
  longest_edge, NOT the video. To sweep n_vis like the other models (up to ~32768), **RAISE
  `proc.video_processor.size.longest_edge`** (e.g. ×256) → n_vis ∝ frames again (1280×720:
  7040/14080/28160/56320 @16/32/64/128f). Caveat: raised budget breaks resolution-uniformity
  (1280×720→440 vs 4K→580 tok/frame) and per-frame max_pixels still caps each frame — but that's
  FINE: n_vis is the x-axis (see encode∝n_vis note). Must set longest_edge on BOTH the HF proc
  (reuse_real) AND the vLLM engine (mm_processor_kwargs) so cold & vt agree. 128f×1280×720=56320
  overflows ctx 40960 → cap frames so n_vis ≤ ~32768.
- **✅ encode ∝ n_vis, RESOLUTION-INDEPENDENT (verified 2026-06-07, Qwen2.5+Qwen3):** at matched
  n_vis, varying resolution (320×240/640×480/1280×720, frames adjusted) gave enc/n_vis ≈ 90-103
  (qwen2.5) / 84-94 µs/tok (qwen3), ±10-15% scatter, NO resolution trend. (Theoretical within-frame-
  attention worry didn't materialize — ViT is MLP-dominated.) → **n_vis is a sufficient x-axis for
  ALL components (encode/prefill/decode/KV); resolution only sets where a (video,frame) lands on it.
  So 1 video suffices for the cost sweep for EVERY model** (Qwen included, with raised longest_edge).
  Artifacts: results/qwen_res_encode/, analyze/qwen_res_encode.py. Real video tok/frame (max_pixels
  768·28·28, BELOW saturation): 320×240→qwen2.5 70/qwen3 40; 640×480→195/150; 1280×720&4K capped→~360.
- **InternVL3.5**: native `<video>` placeholder + `{"video": frames}`; ctx token
  `<|video_pad|>` (count for n_vis). No `max_dynamic_patch`; native **256 tok/frame
  FIXED** (resolution-independent → clean n_vis ∝ n_frame). No HF processor → tokenizer
  chat template. **vt_reuse via IMAGE-embeds** `{"image": (1,n_vis,H)}` (video-embeds
  path is an unfinished vLLM TODO; image path works, verified). vLLM reimplements
  InternVL so **no timm / no shim needed**.
- **LLaVA-OV-7B**: native `<video>` via HF processor (same `build_video_request` path as
  Qwen, `with_metadata=False`); ctx token `video_token_index=151647`. **196 tok/frame FIXED**
  (bilinear-pooled SigLIP, resolution-independent → clean n_vis ∝ n_frame, like InternVL).
  NO pixel cap (max/min_pixels are Qwen-only — would error on the LLaVA processor).
  `max_model_len=32768` (Qwen2 ceiling; 128f=25088 tok fits).
  **vt_reuse via VIDEO-embeds (PATCHED vLLM, RESOLVED 2026-06-05):** the original IMAGE-embeds
  workaround was WRONG — LLaVA's image path expands placeholders into a 2D spatial grid with a
  per-row `image_newline`, so injecting a flat video as one image DOUBLED the prefill sequence
  (12557→25101 @64f) → fake "super-linear inject overhead", vt LOSES. Root cause confirmed via
  profiler (gemm 2×, attn 4× = seq doubled) + prompt_len. FIX: vLLM's LLaVA *video* path had no
  video_embeds branch, so we ADDED one to `llava_onevision.py` (schema `LlavaOnevisionVideoEmbeddingInputs`
  + `_parse_and_validate_video_input` + `_get_mm_fields_config` + `embed_multimodal`; ~20 lines;
  backup `*.bak_videoembeds`; jylim-only env, no other-model/user impact). Inject = RAW prompt
  (single `<video>` placeholder) + `{"video": (1,n_vis,H)}`; vLLM expands to n_vis FLAT tokens
  (video path adds only 1 newline, not per-row). Result: **vt < cold at ALL frames**, encode
  saving ∝ n_vis (H100: 16f −122, 32f −255, 64f −444, 128f −988ms) — now matches InternVL/Qwen.
  **128f works** with `--max-num-batched-tokens 32768` (encoder-cache budget = batched-tokens;
  default 16384 < 25088 vision tokens → ValueError; raise to 32768). So LLaVA runs FULL-range
  16–128f like InternVL. kv_reuse was always fine (warm 26–92ms).
- **Cache discipline differs by script**: `reuse_real.py` runs `enable_prefix_caching=True`
  + `mm_processor_cache_gb=8` (needed for kv_reuse warm) and `reset()`s before every cold
  gen; `stage_timing_vllm.py` runs both caches OFF (batch=1 isolated, no contamination).
  `max_model_len=40960` (InternVL ceiling; LLaVA uses 32768); overflow caught & skipped per-config.

**InternVL via TRANSFORMERS (secondary, stage_timing.py only — Layer-1 cross-check /
true VRAM):** loads `AutoModel(trust_remote_code=True)` (model_type=internvl_chat),
needs `timm` (installed 1.0.27) + 1-line shim
`transformers.PreTrainedModel.all_tied_weights_keys = {}` (custom code predates
transformers 5.9 post_init tying; safe — tying doesn't affect timing). Manual inputs
(`<img>`+`<IMG_CONTEXT>`*256/tile+`</img>`); forward needs `image_flags`, generate
doesn't → adapters expose encode/full_forward/generate separately.

**Precomputed byte sizes (BF16, per token):**
- Vision token = hidden_size x 2 bytes:
  InternVL3.5-4B = 5.0 KB; Qwen2.5-VL / LLaVA-OV-7B = 7.0 KB; Qwen3-VL / InternVL3.5-8B = 8.0 KB;
  InternVL3.5-14B = 10.0 KB
- KV per token = 2 x layers x kv_heads x head_dim x 2 bytes:
  Qwen2.5-VL / LLaVA-OV-7B = 56 KB; InternVL3.5-4B/8B & Qwen3-VL = 144 KB; InternVL3.5-14B = 160 KB
- => KV/token ratio: Qwen2.5-VL / LLaVA-OV-7B ~8x; Qwen3-VL/InternVL-8B ~18x; InternVL-14B ~16x;
  **InternVL-4B ~29x** (small 2560 hidden but 144KB KV) — the byte-ratio extreme.

---

## 4. Paths (human confirms exact values; do not download)

Base dir on NFS is `/mnt/nas/VLM`. CONFIRMED values (2026-06-02):
```
HF_HOME=/mnt/nas/VLM/hf             # models + datasets cache live under hf/hub/
DATA_DIR=/mnt/nas/VLM/datasets      # nextqa (short) + MLVU (long)
LOCAL_SCRATCH=~/VLM/scratch         # LOCAL NVMe (/home, ~317G free): retrieval-timing artifacts + video decode copies
OUTPUT_DIR=~/VLM/results            # results, local, append-only
```
Models are loaded by HF repo id (Section 3) resolved through `HF_HOME`, so there
is no separate `MODELS_DIR` — point `HF_HOME` at `/mnt/nas/VLM/hf` and use repo
ids directly.

Conda: env `vlmcost` is NOT auto-activated in non-interactive shells. Every GPU
run must first:
`source ~/miniforge3/etc/profile.d/conda.sh && conda activate vlmcost`

Datasets (**DONE** as of 2026-06-03; samples prepared on LOCAL_SCRATCH + metadata CSV):
- **NExT-QA** (short, 11–90s) — `lmms-lab/NExTQA`. 16-video sample (size-stratified,
  length spread 12–90s) → `results/nextqa_sample.csv`. Low-res (~640×360) so Qwen
  tokens/frame stay small (no overflow).
- **MLVU** (long, 4–57min) — `MLVU/MVLU`. 10-video sample → `results/mlvu_sample.csv`,
  720p/1080p/4K mix. **Qwen overflows on high-res MLVU unless the VIDEO_MAX_PIXELS cap
  is applied** (Section 3); InternVL (256 tok/frame fixed) is fine.
- (Video-MME `lmms-lab/Video-MME` optional, overlaps MLVU's long range.)
- `data/prepare_nextqa.py` / `prepare_mlvu.py` (re)build the samples with `--n`.

Reminder (Section 2): retrieval-timing artifacts and any "decode this video"
copies go on `LOCAL_SCRATCH` (local NVMe), never on `/mnt/nas`.
If any path is unset/empty, or no .mp4 files are found under DATA_DIR, STOP and
ask the human — do not download. (Both datasets are mid-download now; the human
will signal when complete.)

---

## 5. Measurement design (this is the spec — follow it)

vLLM is the primary engine (Section 2). ONE script — **`measure/reuse_real.py`** —
does almost everything: it loads a model ONCE and measures all three price models
on vLLM's ACTUAL paths (not analytical skips), across a BATCH sweep, with CUDA
graphs ON (real-serving). The earlier two-layer split (stage_timing + throughput)
is SUBSUMED by reuse_real. **`measure/throughput.py` is DEPRECATED** — it measured
only baseline (recompute) throughput; reuse_real now covers throughput for ALL 3
variants. We do NOT need a separate encode/prefill stage-split: TTFT, TPOT, and
throughput are sufficient (the encode-skip prefill comes from vt_reuse directly).

### measure/reuse_real.py — the integrated measurement (PRIMARY)
One model load measures, per (model, video, frames, BATCH), three variants by
controlling vLLM's TWO reuse caches:

| vLLM cache | skips | turned on by |
|---|---|---|
| prefix cache | LLM **prefill** | `enable_prefix_caching=True` |
| mm processor cache | vision **encode** | `mm_processor_cache_gb=8` |

- **cold (baseline)** = `reset()` BOTH caches before every gen → always recompute.
  Records `ttft`(=encode+prefill), `full`(+decode), and a text-only equal-length
  prompt → `prefill_textbase` (prefill alone).
- **kv_reuse** = both caches WARM (populate once, then repeat → cache HIT) → REAL
  warm latency, including the ~25–120ms cache-hit overhead that analytical models
  wrongly treat as 0. (mm cache MUST be on, else warm re-runs the encoder ~215ms.)
- **vt_reuse** = inject precomputed embeds (`enable_mm_embeds=True`) → encoder
  SKIPPED, real encode-skip prefill. random embeds (values irrelevant for latency;
  only shape / n_vis matters). Per-family injection:
  - Qwen2.5: `{"video": {"video_embeds": (n_vis,HID), "video_grid_thw": grid}}` — frames-only
    proc() (cold is also frames-only → grids agree; no metadata/deepstack).
  - Qwen3 (RESOLVED 2026-06-07, no vLLM patch): `{"video": {"video_embeds": (n_vis, **16384**),
    "video_grid_thw": grid, "timestamps": [[0..t-1]]}}`. Build the grid via `proc(...,
    video_metadata=[{...}], do_sample_frames=False)` (NOT frames-only — that re-samples to
    grid_t=2 → n_vis 4× too small). embeds dim = `out_hidden_size*(1+len(deepstack_visual_indexes))`
    = 16384 (DeepStack); timestamps NESTED (per-video batch). See Section 3 Qwen3 quirks.
  - InternVL: `{"image": (1, n_vis, H)}` — vLLM's InternVL **video-embeds** path is
    an unfinished TODO, but video == single-tile 256-tok image sequence, so inject
    ALL n_vis as ONE image item (verified n_vis matches the video path; a single
    item also has ZERO per-item scatter overhead — measured N-item adds 0.3% @16f,
    4.5% @64f). Build vt_reuse AFTER cold/kv so any construction failure loses ONLY
    vt_reuse, not baseline+kv.
  - LLaVA-OV: VIDEO-embeds via PATCHED vLLM (see Section 3 LLaVA quirks). RAW prompt (single
    `<video>`) + `{"video": (1, n_vis, H)}` → vLLM expands to n_vis FLAT tokens. (The earlier
    image-embeds workaround doubled the seq via per-row image_newline → reverted.) Run with
    `--max-num-batched-tokens 32768` for 128f. vt < cold at all frames; matches InternVL/Qwen.
- **DRAM→GPU H2D** (`h2d_tok`, `h2d_kv`) = the real retrieval hop, measured with
  cuda events. storage→DRAM is COMPUTED per tier (Section 7), not measured.

Derived (no stage-split needed): `TTFT`=ttft; `TPOT`=(full−ttft)/decode_tokens;
`throughput`=batch·decode_tokens/wall(full−ttft). break-even uses the TTFT
differences (decode CANCELS, Section 7): token saving = cold_ttft − tok_inject
(= encode); kv saving = cold_ttft − kv_warm (= encode+prefill).

### BATCH sweep (first-class axis, NEVER extrapolated)
`--batches 1 4 8 16`. batch=B submits B requests together; vLLM continuous-batches
them; we record PER-REQUEST cost = whole-batch wall / B. **cold uses B DISTINCT
videos** so prefix caching can't share KV across the batch (InternVL gives
identical n_vis per frame-count regardless of video → distinct videos are a fair
batch). Max batch is capped by KV-cache ÷ per-request-tokens (InternVL-8B 128f
≈ 11), so large batch × large frame OOMs → caught per-config and skipped.

### CUDA graphs ON (`--cudagraph`, real-serving throughput)
`enforce_eager=False`. CUDA graphs remove per-step kernel-launch overhead → decode
(hence throughput) matches PRODUCTION serving; ttft (prefill-bound) is ~unchanged
vs eager. vt_reuse embeds-inject works under cudagraph (verified). Cost: ~16s
torch.compile per frame-shape (one-time) + small capture memory; `warmup≥1` per
(frame,batch) absorbs the first-gen capture so timed runs are clean.

### Timing method (verified correct)
`llm.generate()` is SYNCHRONOUS (returns only when all tokens are done + GPU
synced), so wall-clock via `time.perf_counter()` around it IS the real-serving
latency — cuda events are neither needed nor appropriate here (we want wall-clock,
not pure-kernel time). `detokenize=False`, warmup discarded, median of ≥5 runs.
Only the H2D transfer (truly async) uses cuda Event + synchronize. (The separate
transformers path, `stage_timing.py`, DOES use cuda events — it isolates GPU stages
in-engine; never mix the two engines' numbers, Section below.)

### Frame sampling & per-model n_vis (fairness)
decord, uniform `linspace` of `--frames` indices (frame-COUNT sweep, NOT fps — so
n_vis is controlled directly as the x-axis). **InternVL = 256 tok/frame FIXED**
(n_vis ∝ n_frame exactly → clean fairness, no encode-vs-n_frame ambiguity).
**Qwen = DYNAMIC** (n_vis ∝ resolution) → cap frame resolution with
`VIDEO_MAX/MIN_PIXELS` (`--video-max-patches 768` = Qwen-native default) applied to
BOTH the HF processor (for embeds_req's grid) AND the vLLM engine
(`mm_processor_kwargs`) so they agree — else high-res MLVU (4K = 3952 tok/frame)
overflows 40960 context. NExT-QA is low-res so the cap is a no-op there.
**FAIRNESS RESOLVED (2026-06-07): n_vis is a sufficient x-axis for ALL models** — encode (and
prefill/decode/KV) is ∝ n_vis and RESOLUTION-INDEPENDENT at matched n_vis (verified, Section 3
Qwen notes). So the "encode-vs-n_frame ambiguity" for dynamic Qwen is moot: just use measured
n_vis as x. **1 video + frame sweep suffices for every model** (Qwen3 needs raised longest_edge
to keep n_vis ∝ frames instead of saturating at 12288 — Section 3).

### Secondary / cross-check (NOT the main path)
- `stage_timing_vllm.py`: batch=1 ttft/decode + `--text-baseline` encode/prefill
  split. Subsumed by reuse_real; kept for an explicit stage split if needed.
- `stage_timing.py` (transformers): true peak VRAM + cuda-event stage timing.
  Cross-check only. **Do NOT subtract one engine's encode from the other's ttft** —
  vLLM's fused tower is ~4× faster so the subtraction goes negative.
- `measure/throughput.py`: **DEPRECATED** (baseline-only; reuse_real covers it).

### Retrieval / network — 2-hop (storage→DRAM COMPUTED, DRAM→GPU MEASURED)
`retrieval_per_query = network_cost(storage→DRAM) + h2d·resource_price`. The
storage→DRAM hop is COMPUTED from `read_bytes` + tier (bandwidth/egress); the
DRAM→GPU hop (`h2d_tok`/`h2d_kv`) is MEASURED in reuse_real via cuda events.
`read_bytes` is COMPUTED from config + measured `n_vision_tokens`. bandwidth,
egress price, and `resource_price` are config parameters SWEPT per storage tier
(Section 7) — they depend on tier+cloud and can't be measured on one local server.
(No "local NVMe lower bound".)

- `bytes_vision_tokens` = n_vision_tokens x (hidden x 2)
  — vision-token reuse reads this (small).
- `bytes_kv` = n_vision_tokens x (2 x layers x kv_heads x head_dim x 2)
  — KV reuse reads this (8-18x larger). This byte gap drives the break-even.
These feed both `C_store` and the `read_bytes` term of the network cost (Section 7).

Reuse cost is now MEASURED DIRECTLY by reuse_real (cold/kv/token), NOT composed
analytically — this captures vLLM's real cache-hit (kv_warm ~25–120ms) and
embeds-inject overhead that the analytical "skip = 0" ignores. KV reuse = kv_warm +
decode + network(bytes_kv); token reuse = tok_inject (real encode-skip prefill) +
decode + network(bytes_vision_tokens). `read_bytes` feeds `C_store` + the
storage→DRAM network term.

---

## 6. Datasets (assume present; roles)

- **Video-MME** (or MLVU) — long videos (minutes to ~1hr). Long end of the
  length axis; where KV cost / prefill cost is dominant.
- **NExT-QA** — short videos (tens of seconds), multiple QAs per video. Short
  end; where token reuse is clearly enough.
- For cost-primitive measurement, query CONTENT does not matter — only video
  length / resolution / frame count. So sample videos to cover a wide
  length & resolution spread; metadata (orig resolution, duration, fps) must be
  logged alongside every measurement row.
- **Arrival pattern (N, time) is synthetic**: Zipf popularity over videos +
  Poisson/trace arrival. No QA dataset has timestamps. Keep this in a separate
  module from the cost-primitive measurement.

---

## 7. Price model (the final computation)

**N is a query RATE: queries per MONTH** (matches the per-month storage rent). Over a
retention of R months (R = retention_days/30) a video gets N·R total accesses; the
one-time store cost (encode[+prefill]) is paid ONCE, storage rent is per-month × R.

Total $ over the retention window, per model variant:
- baseline:        N·R x (T_encode + T_prefill + T_decode)
- KV reuse:        once(T_encode + T_prefill) + N·R x (T_decode + network_cost(bytes_kv))
                   + C_store_per_month(bytes_kv) x R
- token reuse:     once(T_encode) + N·R x (T_prefill + T_decode + network_cost(bytes_vision_tokens))
                   + C_store_per_month(bytes_vision_tokens) x R

Break-even query rate: N* (per month) = (F + storage_total) / (R·(b − r)), where
F = one-time store cost, b = per-query baseline, r = per-query reuse. As R→∞,
N* → storage_per_month / (b − r) (steady state: monthly rent vs per-query saving).

**T_decode CANCELS in the break-even** (baseline and both reuse variants all decode):
b − r = (T_encode + T_prefill) for KV reuse, = T_encode for token reuse (minus network).
So decode length does NOT move the crossover — it only scales absolute cost and the
cost-share plot. What sets break-even is the encode(+prefill) compute SAVED vs the
storage+network rent. Empirically (vLLM, fast encode/prefill) the per-query saving is
small, so break-even is HIGH (tens–hundreds of queries/month) — reuse pays off only
for popular videos, and storage rent dominates so retention barely shifts N*.

**Network / retrieval — 2-hop; storage→DRAM COMPUTED, DRAM→GPU MEASURED:**
```
read_bytes      = bytes_kv (KV) | bytes_vision_tokens (token)         # COMPUTED
storage_to_dram = read_bytes / bandwidth                              # per-tier
network_cost    = read_bytes x egress_price + storage_to_dram x resource_price
retrieval_total = network_cost + h2d x resource_price                 # + DRAM->GPU (MEASURED)
```
- `read_bytes` COMPUTED (config + measured n_vis). `bandwidth`, `egress_price` are
  per-tier config (`config/storage_tiers.yaml`). **`resource_price` is NOT a tier knob** —
  it is $/s of the resource that STALLS during retrieval (the H100), read from
  `prices.yaml` (`gpu_h100_usd_per_hour`). `latency_fixed` & `get_price` REMOVED (negligible).
- `read_bytes` is THE lever: KV is 8–29× the vision-token bytes → KV reuse moves that much
  more data → drives the break-even between the two reuse types.
- `T_decode` cancels in break-even but keep it for cost-share / TPOT / throughput plots.
- **Sweeps → a FAMILY of surfaces, x-axis = measured n_vis, per model:** (1) **storage
  tier** ×2 (`config/storage_tiers.yaml`: `local_nvme` 5GB/s egress 0; `s3_same_region`
  1GB/s $0.023/GB-mo egress 0) — `ebs_gp3` and `object_internet` were dropped to keep
  the figure to the two extremes (fast-local vs slow-object); (2) **GPU-stall on/off**
  (`--no-gpu-stall` → resource_price=0 = retrieval overlapped with compute) — this DECIDES
  whether KV reuse lives; (3) **egress on/off**; (4) **batch**. This is the paper's core figure.
- analyze: **`analyze/breakeven_reuse.py`** is PRIMARY — uses reuse_real's MEASURED
  cold/kv/token TTFTs directly (keeps real warm/inject overhead). `analyze/price_model.py`
  is the analytical (stage-split) variant, kept for cross-check.
- **Key findings (InternVL 4B/8B/14B, batch=1, GPU-stall ON):** (a) N* DROPS as n_vis
  grows (encode+prefill super-linear) — long video favors reuse; (b) **token reuse
  ~always wins** (object_same_region 5–21/mo), smaller model → lower N* (smaller token
  bytes); (c) **KV reuse = `never` on ≤1GB/s tiers** — KV's huge bytes make the retrieval
  stall exceed the compute saving; only local_nvme (5GB/s) survives (146–487/mo);
  (d) byte-ratio decides — InternVL-4B (29×) is worst for KV reuse; (e) **GPU-stall is the
  KV switch** — stall OFF revives KV reuse on all same-region tiers (~50/mo); (f) egress
  kills object_internet, but even egress=0 won't save KV on slow tiers (bandwidth does).

### 7.1 Workload-level TCO (2026-06-08, `tco_report.py` + `total_vpm.csv` → `Report.md`)
Beyond per-(model,n_vis) break-even, we compute **workload TCO saving** over a REAL popularity dist
(`total_vpm.csv`: 88,217 videos, views/month + age_months). Per-video optimal: cache iff `N·R·saving_per_q
> F + storage·R` (= N≥N*), else recompute. Saving% = Σ_cached max(0,gain) / Σ_all baseline_TCO (decode
included → % of TOTAL TCO). **Op point: frame=128, batch=8, R=median age (~50mo), x-axis = measured n_vis,
vt_reuse MAIN + kv_reuse compare; 4 models (IVL-8B, LLaVA, Qwen2.5, Qwen3 — 4B/14B dropped from report).**
Key results & mechanisms:
- **vt_reuse saves 13–37% (local) / 8–31% (s3) of total TCO and is tier-INSENSITIVE** (vt bytes tiny →
  retrieval cheap). **kv_reuse: high on local (40–65%) but DIES on s3** (InternVL/Qwen3 N*=`never`) — it's
  **retrieval (bandwidth), not storage rent**, that kills KV (verified by --no-retrieval: KV revives 68–76%).
- **Break-even coverage (heavy-tail):** on s3 only **21–38% of videos** clear vt's N* (2.6–12/mo), yet those
  cover **~99.7% of all views** — cache the popular minority, recompute the long-tail (≈0 view volume).
- **vt inject overhead ≈ FIXED ~53ms** (batch=8; in-engine embeds scatter/merge, NOT the h2d transfer).
  It's baked into measured vt_ttft → TCO already reflects it. It suppresses apparent encode-saving at SMALL
  n_vis (back it out → encode/n_vis ≈ constant ~19µs/tok, i.e. the encoder IS linear).
- **decode is batch-dependent:** batch=1 decode is WEIGHT-bandwidth-bound → ~n_vis-independent (fixed FFN
  floor); batch↑ amortizes weights → KV-bound → decode ∝ n_vis (measured: n_vis 8× → decode b1 1.2×, b16 3.85×).
  encode ALSO collapses with batch (encoder parallelizes: b1→b8 encode 1004→550ms). So batch↑ → decode/encode
  collapse → kv saving% UP; vt% roughly flat.
- **saving% vs n_vis is NON-monotonic (rise→peak→fall):** the FIXED decode FFN floor (~312ms@b8, n-independent)
  dominates the denominator at small n_vis → suppresses saving% (rising region); prefill (super-linear, vt
  can't skip it) dominates at large n_vis → falling region. **TTFT% (denom=encode+prefill, NO decode floor)
  peaks EARLY (~32f) and is ~monotone down without the inject-overhead artifact; TCO% (denom +decode floor)
  peaks LATE (~n_vis 40–65k ≈ ctx limit), so within feasible n_vis it looks rise→plateau, decline only just
  beyond ctx.** prefill is only mildly super-linear (~n^1.2) here (FFN+flash-attn), not n², so decline is slow.

---

## 8. Code conventions

- **Every script that touches the GPU must enforce `CUDA_VISIBLE_DEVICES=1`**
  (the H100). Read it from env and assert exactly one visible device, or set it
  at the top of the entry point. Never hardcode `cuda:1` (after pinning, the
  H100 is `cuda:0`). A run that lands on GPU 0 is invalid and must be discarded.
  ESCAPE HATCH: `reuse_real.py` honors `ALLOW_GPU0=1` to bypass the H100 assert for
  FUNCTIONAL validation on GPU0 (Blackwell) — those timings are NOT H100-normalized
  and must be written to an ISOLATED CSV (e.g. `results/nextqa_blackwell/`), never the
  real `results/{dataset}/reuse_real.csv`. Final price-model numbers come from H100 only.
- Python 3, type hints, small composable functions. One concern per file.
- No magic numbers — model dims, unit prices, dtype sizes live in a `config/`
  module or YAML, not inline.
- Every measurement script writes a tidy CSV/JSON row per (model, video,
  stage, batch, run_idx) with all metadata; never overwrite, append + timestamp.
- Timing: always `torch.cuda.synchronize()` before/after; warmup excluded;
  report median + IQR. Separate CPU (decode/resize) time from GPU (encode/
  prefill/decode) time — they get different unit prices.
- Make measurement reproducible: fix seeds, log model + vLLM versions, GPU clock
  if pinned.
- Prefer dataclasses for results; a single `analyze.py` reads the CSVs and
  produces the break-even plots — keep measurement and analysis separate.

## 9. Suggested file layout

```
config/
  models.yaml         # 6 model dims + dtype (single source for byte math)
  prices.yaml         # compute $/hr (GPU, CPU) + run defaults (decode_tokens, retention)
  storage_tiers.yaml  # 2 tiers: local_nvme (5GB/s), s3_same_region (1GB/s, $0.023/GB-mo)
  __init__.py         # typed loaders: load_models / load_prices / load_storage_tiers
measure/
  byte_sizes.py        # computes read_bytes (token + KV) from config (no GPU) — feeds network_cost
  frames.py            # frame sampling helpers (linspace frame-count; Qwen-native fps)
  reuse_real.py        # ***PRIMARY*** integrated: cold/kv_reuse/vt_reuse x BATCH x cudagraph;
                       #   TTFT/TPOT/throughput + H2D; vLLM cache control; Qwen max_pixels cap.
                       #   --vt-mode {inject(default)|mmhit}; --max-num-batched-tokens (LLaVA 128f=32768);
                       #   LLaVA vt=video-embeds(raw <video>), Qwen vt=video_embeds+grid(+ts for qwen3)
  stage_timing_vllm.py # SECONDARY (vLLM): batch=1 ttft/decode + --text-baseline split (cross-check)
  stage_timing.py      # SECONDARY (transformers): true VRAM + cuda-event stages (cross-check)
  throughput.py        # DEPRECATED (baseline-only batch sweep; subsumed by reuse_real)
  preprocess_timing.py # C_preprocess: CPU video decode/resize per video (no GPU)
data/
  prepare_nextqa.py    # extract NExT-QA sample (zip) -> LOCAL_SCRATCH + metadata CSV
  prepare_mlvu.py      # sample MLVU (mp4) -> LOCAL_SCRATCH + metadata + REAL query from json
workload/
  arrival.py           # Zipf popularity + Poisson/trace arrival synthesis
analyze/
  breakeven_reuse.py   # ***PRIMARY*** break-even from reuse_real's MEASURED cold/kv/token TTFTs;
                       #   tier x gpu-stall x egress sweep; --models filter; (model,n_vis) aggregate
  price_model.py       # analytical (stage-split) break-even — cross-check variant
  fig_internvl8b.py    # ***FIGURES*** --model/--frame/--dataset parameterized. fig1 TTFT-vs-n_vis,
                       #   fig2 throughput, fig3 break-even, fig5 TCO-saving% grid(batch×tier),
                       #   fig5_1 = LOCAL saving-vs-N @batch=8 (L:retr-EXCL | R:retr-INCL panels),
                       #   fig5_2 = S3 saving-vs-N @batch=8 (same 2 panels; shows kv dies w/ retrieval)
                       #   (2026-06-08: old fig5_0 R=1000mo & fig5_2 retr-grid REMOVED; per-tier 2-panel now),
                       #   fig6 TPOT, fig7 tput-by-frame, fig8 TTFT-breakdown (compute/sto→DRAM/H2D).
                       #   fig3/5 use s3_same_region only; fig8 shows both tiers.
                       #   PINS to ONE video (stem w/ most rows = the batch-sweep video) so the n_vis
                       #   x-axis is monotonic — multi-video Qwen data has varying n_vis/frame and would
                       #   fold back. InternVL/LLaVA unaffected (fixed tok/frame). Skips if no batch-1 data.
  plots.py             # primitives CSV + config -> break_even/cost_share figs (x=n_vis)
  tco_workload.py      # ***WORKLOAD TCO*** per-video-optimal caching over total_vpm.csv (real
                       #   views/month + age dist); --frame/--batch/--tier/--no-retrieval/--retention-months
  tco_report.py        # ***REPORT FIGS/TABLES*** -> results/report/fig1..7 + Report.md tables.
                       #   default op point: frame=128, batch=8, R=median age. vt_reuse MAIN, kv compare.
total_vpm.csv          # 88,217 real videos: views_per_month (N, heavy-tail median≈1.1) + age_months (R, median≈50)
Report.md              # ***DELIVERABLE*** (Korean) workload-TCO report; figs in results/report/
scripts/run_full.sh    # orchestrator (per-(model,dataset) process, freeze watchdog, EngineCore reap)
        run_qwen25_2pass.sh / run_intern_4b_14b.sh / run_llava_ve.sh  # per-model launchers
                       #   (run_llava_ve.sh = LLaVA video-embeds, 1 video, 16-128f, batched-tokens 32768)
results/{dataset}/{model_tag}/   # CSV per dataset (reuse_real.csv) + per-model figures (append-only)
        nextqa_blackwell/         # ISOLATED GPU0/Blackwell prelim — never mixed with H100 data
```

## 10. Current state & how to run (2026-06-03)

Env + models + datasets all READY; byte_sizes sanity-check done. Pipeline:
1. Samples prepared (`data/prepare_nextqa.py` / `prepare_mlvu.py`).
2. **MEASURE (PRIMARY): `measure/reuse_real.py`** — one model load → cold/kv/token ×
   batch × cudagraph. e.g.:
   ```
   python -m measure.reuse_real --model internvl3.5-8b \
     --videos-csv results/nextqa_sample.csv --frames 16 32 64 128 \
     --batches 1 4 8 16 --runs 5 --warmup 2 --cudagraph
   ```
3. **ANALYZE (PRIMARY): `analyze/breakeven_reuse.py`** — N* over tier × gpu-stall × egress
   (e.g. `--models internvl3.5-8b --no-gpu-stall`).

STATUS: **InternVL** (4B/8B/14B) validated (NExT-QA + MLVU; 256 tok/frame fixed). **Qwen2.5**
works (MLVU needs max_pixels cap). **LLaVA-OV-7B** RESOLVED on H100 (2026-06-05): vt_reuse via
**video-embeds (patched vLLM)** — full-range 16/32/64/**128**f, vt < cold at all frames
(−122/−255/−444/−988ms), matches InternVL/Qwen. Single video sufficient (encode latency is
video-length-independent at fixed frame count). All 5 models now live in ONE CSV
`results/nextqa/reuse_real.csv` (LLaVA = 540 rows video-embeds; old image-embeds artifact
purged). Figures in `results/nextqa/{model}/`. **Qwen3-VL: vt_reuse RESOLVED 2026-06-07 — NO
vLLM patch** (it was a `reuse_real.py` construction bug, not vLLM). The cold/vt n_vis "mismatch"
was: 1760 is CORRECT (grid_t=8), vt's 440 was wrong (metadata-less proc re-sampled to grid_t=2).
3 fixes in the vt-inject branch: `video_metadata`+`do_sample_frames=False` on proc(), embeds dim
16384 (DeepStack `out_hidden_size*(1+3)`), NESTED timestamps `[[...]]` (batched("video") field).
Validated H100: vt<cold @16f (152<204, n_vis 1760) & @64f (524<1059, n_vis 7040). **Qwen3 NOT yet
in production CSV** (clean slate, no purge) — run the full sweep like the other 5 models.

**DONE (2026-06-07, all 6 models complete):** cost sweep unified to **x-axis = measured n_vis** for
ALL 6 models (justified: encode/prefill/decode/KV all ∝ n_vis, resolution-independent — verified).
- **Qwen3-VL-8B sweep DONE** → results/nextqa/reuse_real.csv (3375 rows; cold/kv_reuse/vt_reuse;
  frames 16/32/64/128 × batch 1/4/8/16; cold==vt n_vis aligned; 128f=19200 confirms longest_edge
  override worked in production). Run via `scripts/run_qwen3_2pass.sh` (`--qwen3-longest-edge` default
  154M lifts the 12288 saturation; `--max-num-batched-tokens 32768`). Backup: reuse_real.csv.bak_pre_qwen3.
- **All 6 models now in results/nextqa/reuse_real.csv**: internvl 4b/8b/14b, llava-ov-7b, qwen2.5-vl-7b,
  qwen3-vl-8b. **Figures (11 each) in results/nextqa/{TAG}/** where TAG = internvl4b/8b/14b, llavaov,
  qwen25, qwen3vl8b (NOTE: TAG ≠ model-key; see fig_internvl8b.py TAG map).
- **Feasibility: 5 models done** (results/feasibility.csv): internvl 4b/8b/14b + llava + **qwen2.5-vl-7b**
  (b1/4/8/16/32; all ~112f context-bound at 40960, small 56KB KV like LLaVA → fits high batch). **Qwen3
  feasibility SKIPPED (moot — n_vis is our longest_edge knob, never token-OOMs).** `feasibility.py` fixed:
  real n_vis count, ctx capped MML_CAP=40960, VID=vid_1280x720.csv. (Qwen3 stray junk rows purged;
  capped 1280x720 SATURATES Qwen3 n_vis ~12k so it's only valid for Qwen2.5 there.)
- **Break-even** (`analyze/breakeven_reuse.py --csv results/nextqa/reuse_real.csv`): all 6 models,
  tiers local_nvme + s3_same_region, gpu-stall on/off. On these egress-free tiers N* is low (vt~1.0,
  kv~1.3-1.9/mo) → reuse ~always wins. Outputs: /tmp/breakeven_{all,nostall}.txt (regenerate as needed).

vLLM PATCH (jylim env only, backup kept): `llava_onevision.py` gained a video_embeds path.
Final numbers come from this patched vLLM; revert via `*.bak_videoembeds` if needed.

Re-run LLaVA: `CUDA_VISIBLE_DEVICES=1 scripts/run_llava_ve.sh` (1 video, 16–128f, video-embeds,
--max-num-batched-tokens 32768) → `fig_internvl8b.py --model llava-ov-7b --dataset nextqa --frame 128`.

Before any GPU run: confirm H100 (GPU 1) idle (shared w/ `ljh`,`chani227`), `CUDA_VISIBLE_DEVICES=1`,
`conda activate vlmcost`. After a vLLM job: reap the orphan `VLLM::EngineCore` (Section 2).

---

## 11. LMCache validation + pre-projector vt_reuse — CURRENT DIRECTION (2026-06-09)

The measurement pivoted to a **3-way comparison, ALL real-measured in vLLM TTFT** (no analytical skips):
1. **cold** = vLLM full recompute (encode + prefill + decode).
2. **kv_reuse** = **LMCache**-based (real KV offload to a tier + reload; the retrieval §7 used to COMPUTE is now MEASURED).
3. **vt_reuse** = **vLLM-based, PRE-projector** (reuse the vision ENCODER/ViT output → skip ViT, RE-RUN the cheap projector) — or **POST-projector via EC** where pre is impossible (Qwen3 DeepStack).

### Per-model vt_reuse assignment
- **PRE-projector** (skip ViT, run projector): InternVL-4B/8B/14B, LLaVA-OV-7B, Qwen2.5-VL-7B.
- **POST-projector via EC** (skip whole tower): Qwen3-VL-8B — DeepStack taps mid-ViT layers so pre-projector is ill-defined.

### Environments (version friction — READ before running)
- **Use `vlmcost`** (vLLM 0.22 / torch 2.11 / transformers 5.9): **lmcache 0.4.6 installed here and its c_ops loads clean**. KV + EC + pre-projector all run here = SAME engine as reuse_real → **zero version confound**.
- A separate `lmcache` conda env (vLLM 0.18 / torch 2.10 / lmcache 0.4.4) exists from the first KV test but is now SECONDARY — upgrading it to 0.4.6 BREAKS (torch-2.10 c_ops ABI `undefined symbol`; lmcache 0.4.6 needs transformers≥5.4 vs vLLM-0.18's 4.5x). Prefer vlmcost.
- **GPU0/Blackwell is BLOCKED for LMCache** — prebuilt c_ops has no sm_120 kernel (`cudaErrorNoKernelImageForDevice`). LMCache work (KV & EC) is **H100-only**. (Pre-projector preproj_vllm.py is plain vLLM → GPU0 would work, but keep on H100 for consistency.)

### kv_reuse = LMCache — `measure/reuse_lmcache.py`
`--mode {lmcache(KV)|ours|ec}`, `--tier {dram|disk}`, isolated → `results/lmcache/`. KV connector `LMCacheConnectorV1` via `kv_transfer_config`. **`enable_prefix_caching=False` forces a REAL tier load** (else KV is GPU-resident → log shows `need to load: 0` = GPU hit, not tier retrieval).
- **VALIDATION RESULT:** our COMPUTED-then-measured retrieval term `h2d_kv` (reuse_real 0.22: 3.2/6.5/12.9/25.9ms @16/32/64/128f) ≈ LMCache's REAL DRAM KV load (3.8/7.4/14.9/29.6ms), both ~45–52 GB/s. → **our kv_reuse retrieval model is validated by a real system.** (`Report_lmcache.md` + `analyze/lmcache_compare.py` fig.)
- disk(NVMe) tier HANGS with GDS+forced-spill → retry `LMCACHE_USE_GDS=False`. DRAM tier works.

### vt_reuse EC (encoder cache) — what vLLM vs LMCache provide
vLLM ships the EC FRAMEWORK (`vllm.distributed.ec_transfer`: ECConnectorBase + ECExampleConnector[disk toy] + factory) in 0.22/main. LMCache adds the production EC backend (`lmcache...vllm_ec_adapter.LMCacheECConnectorImpl` + `ECCacheEngine`, tiered). The vLLM-side bridge `LMCacheECConnector` is in **UNMERGED vLLM PR #38668** → vendored into OUR repo at `measure/lmcache_ec_connector.py`, pointed to via `ec_connector_module_path="measure.lmcache_ec_connector"` (factory falls back to it for unregistered names → **NO vLLM source edit**; needs PYTHONPATH=repo for the worker). EC caches the POST-projector tower output keyed by vLLM **mm_hash**. **CRITICAL: do NOT set `mm_processor_cache_gb=0` for EC** — that yields positional mm_hash (`renderer0-mm-N`) → never hits; default mm cache gives a content hash → warm hits & skips the tower. Qwen3 EC verified: warm≪cold (16/32/64f: 77/149/318 vs 210/508/1097ms), `EC put` stores n_vis×16384(DeepStack)×2 bytes.

### vt_reuse PRE-projector — `measure/preproj_vllm.py` (REAL vLLM, NO source edit)
Run engine **IN-PROCESS** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so a main-process monkeypatch reaches the model, + **`mm_processor_cache_gb=0`** so vision REALLY re-runs each generate (opposite of EC! else the repeat hits the in-engine mm cache → all modes look identical). Per-model monkeypatch on the encoder→projector split, branched by env `VLM_REUSE_MODE`:
- **cold** = original (ViT + projector); **pre** = random ViT-output → run projector → prefill (skip ViT only); **post** = random post-projector embeds → prefill (skip ViT + projector).
- Patchers (PATCHERS registry): InternVL `extract_feature` (vision_model→pixel_shuffle→mlp1); LLaVA `_video_pixels_to_features` (vision_tower→multi_modal_projector→apply_pooling); Qwen2.5 `Qwen2_5_VisionTransformer.forward` (skip blocks loop, keep merger + reverse_indices; ctx=`merger.ln_q.weight.shape[0]`, d_model=`merger.mlp[-1].output_size` — both vLLM RMSNorm/ParallelLinear, NOT nn.Linear). Qwen3 NOT patchable (DeepStack) → EC.

### KEY FINDINGS (REAL vLLM TTFT, verified InternVL-8B / LLaVA / Qwen2.5)
- **pre ≈ post at all n_vis** (projector/merger is NEGLIGIBLE): cold−pre = ViT(encoder) cost (InternVL 51–376 / LLaVA 40–145 / Qwen2.5 63–229 ms over 16–128f); pre−post ≈ 0 (±7ms). → **switching vt_reuse post→pre does NOT change cost/break-even.**
- The ONLY reason to prefer PRE-projector is **cross-model sharing**: InternVL-4B/8B/14B share the SAME InternViT → ViT (pre-projector) features are reusable ACROSS models; post-projector embeds are LLM-specific (not shareable). This is the long-term angle (esp. with the two 30B MoEs).
- EC ≈ our old inject (LLaVA EC 119/247/522/1199 vs inject 123/249/596/1357ms) → inject was a faithful proxy; prior reuse_real vt numbers stand.

### TCO changes (2026-06-09, going forward)
- **Reflect the encoder-output vs projector-output BYTE difference**: vt_reuse stores the **ENCODER output (pre-projector)** for InternVL/LLaVA/Qwen2.5 — different bytes than post-projector (e.g. InternVL pre = 1025 tok × 1024-dim/tile vs post = 256 × llm_hidden). Qwen3 vt = post-projector (EC) → projector bytes (16384-dim DeepStack).
- **GPU-stall cost EXCLUDED** (`--no-gpu-stall`, resource_price=0 = retrieval overlapped with compute).
- **S3 object storage ONLY** for TCO requests for now (drop local_nvme).

### Files (this phase — all isolated under results/lmcache/)
- `measure/preproj_vllm.py` — ***PRIMARY vt_reuse*** (cold/pre/post real vLLM TTFT; InternVL/LLaVA/Qwen2.5 patchers).
- `measure/lmcache_ec_connector.py` — vendored vLLM PR #38668 EC connector (repo-local; no vLLM edit).
- `measure/reuse_lmcache.py` — ***kv_reuse*** (LMCache KV) + `--mode ec/ours`.
- `measure/smoke_lmcache.py` (KV smoke), `measure/smoke_ec_qwen3.py` (Qwen3 EC smoke), `measure/preproj_internvl.py` (transformers stage-split, SUPERSEDED by preproj_vllm).
- `analyze/lmcache_compare.py` + `Report_lmcache.md` — kv_reuse validation figure + report.
- `results/lmcache/` — reuse_lmcache.csv (KV/ours), reuse_ec.csv (EC), preproj_vllm.csv (cold/pre/post), lmcache_retrieve_dram.csv, fig_lmcache_compare.png.

### TODO (next)
- InternVL-4B/14B pre-projector quick verify (same internvl patcher; deferred 2026-06-09).
- Full cold/vt/kv sweep for the production 5(+Qwen3) models → TCO with the encoder-byte / no-stall / S3-only changes.
- 30B MoE (InternVL/Qwen) measurement (long-term).

---

## 12. FINAL paper experiment — harness READY (2026-06-09)

The production 3-way run. Orchestrator: **`scripts/run_final.sh`** (background; reaps jylim
EngineCore between vLLM runs; warns on other-user GPU use; per-config OOM/overflow auto-skip).

**Matrix:** 4 models {InternVL-8B, LLaVA-OV-7B, Qwen2.5-VL-7B, Qwen3-VL-8B} × frames {16,32,64,128}
× batch {1,4,8,16} × 6 videos (`final_videos.csv`: NExT-QA 360p ×3 + MLVU game_33=720 / movie101_87≈1080
/ xiaoliyu_9=4K). All eager (in-process monkeypatch needs it); engine-mismatch across variants accepted.

**Who measures what (all REAL `generate()` TTFT, per-request = whole-batch wall / B):**
- **cold + vt** → `measure/preproj_vllm.py` (in-process, `mm_processor_cache_gb=0` so vision re-runs):
  cold = orig extract_feature (full recompute); vt_pre/vt_post via the encoder→projector monkeypatch.
  **Qwen3 = COLD-ONLY here** (no pre-projector patcher → DeepStack). Records cold ttft+**full(decode=256)**
  + vt_pre/vt_post ttft. n_vis counted from real prompt tokens. `--csv results/final/preproj_vllm.csv`.
- **kv_reuse (all 4)** → `reuse_lmcache.py --mode lmcache --tier dram` (real KV LOAD; prefix OFF).
  `results/final/reuse_lmcache.csv`.
- **vt_reuse (Qwen3)** → `reuse_lmcache.py --mode ec --tier dram` (EC post-projector; mm cache ON for
  content-hash). `results/final/reuse_ec.csv`.

**cold source = preproj_vllm** (NOT reuse_real) — real full recompute, validated (InternVL 308 / Qwen2.5
670@720 ms @16f), same engine as vt for the 3 patchable models → clean cold−vt saving.

**Measurement tier = DRAM** (real DRAM→GPU load, ~42–52 GB/s measured). storage→DRAM (S3 / local) is
COMPUTED in TCO (§7). **TCO knobs (per §11): encoder-output bytes for vt, GPU-stall OFF, S3-only.**

**⚠️ vt H2D consistency (2026-06-09):** kv (LMCache) and Qwen3-EC vt INCLUDE the real DRAM→GPU load in
their TTFT (LMCache loads from CPU-DRAM). To match, preproj `_h2d()` now loads the "cached" vision
tensor from a **pinned CPU buffer → `.to(GPU)` (blocking)** inside the timed generate (pre = ViT/encoder
output bytes; post = post-projector bytes), instead of synthesizing on-GPU. So all 3 variants' TTFT
include real DRAM→GPU retrieval. **Impact MEASURED ≈ negligible / within run-to-run noise** (InternVL b1
vt: 16f 258→262, 128f 3143→3164 ms) — vision-token bytes are small (128f InternVL ≈268MB → ~5ms @50GB/s),
buried in prefill-bound TTFT. So rigor is preserved; conclusions (vt≈prefill, break-even) unchanged.
**decode_tokens = 256** (preproj cold `full`; decode is variant-independent so vt/kv full = their ttft +
(cold_full − cold_ttft)). Long-format CSVs keyed by (model,dataset,video_id,res_label,frames,batch,n_vis,
variant,metric).

Launch: `nohup bash scripts/run_final.sh > results/final/run.log 2>&1 &`. Prior results archived to
`results/_archive/pre_final_20260609/`.