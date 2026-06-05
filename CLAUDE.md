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
  processor (must match) → ~360 tok/frame, uniform across resolutions.
- **Qwen3-VL**: REQUIRES video metadata → `{"video": (frames, {total_num_frames, fps,
  frames_indices, do_sample_frames:False, width, height, duration, video_backend})}`.
  cold/kv work; **vt_reuse (video-embeds inject) still raises `KeyError:'timestamps'`
  in vLLM 0.22 → UNRESOLVED** (build vt_reuse after cold/kv so only it is lost).
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
  - Qwen: `{"video": {"video_embeds": (n_vis,H), "video_grid_thw": grid}}`
  - InternVL: `{"image": (1, n_vis, H)}` — vLLM's InternVL **video-embeds** path is
    an unfinished TODO, but video == single-tile 256-tok image sequence, so inject
    ALL n_vis as ONE image item (verified n_vis matches the video path; a single
    item also has ZERO per-item scatter overhead — measured N-item adds 0.3% @16f,
    4.5% @64f). Build vt_reuse AFTER cold/kv so a construction failure (Qwen3's
    vLLM video-embeds path still raises KeyError:'timestamps') loses ONLY
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
                       #   fig2 throughput, fig3 break-even, fig5/5_1/5_2/5_3 TCO-saving%,
                       #   fig6 TPOT, fig7 tput-by-frame, fig8 TTFT-breakdown (compute/sto→DRAM/H2D).
                       #   fig3/5 use s3_same_region only; fig8 shows both tiers.
  plots.py             # primitives CSV + config -> break_even/cost_share figs (x=n_vis)
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
purged). Figures in `results/nextqa/{model}/`. **Qwen3-VL: video-embeds inject now RUNS** with
timestamps(nested `[[...]]`, len==grid_thw[0]) + DeepStack embeds dim `HID*(1+3)=16384`
(deepstack_visual_indexes=[8,16,24]) — but cold/vt n_vis MISMATCH (Qwen3 spatial-merges video:
cold counts raw 1760 vs vt merged 440) so fair comparison needs n_vis reconciliation → DEFERRED.

vLLM PATCH (jylim env only, backup kept): `llava_onevision.py` gained a video_embeds path.
Final numbers come from this patched vLLM; revert via `*.bak_videoembeds` if needed.

Re-run LLaVA: `CUDA_VISIBLE_DEVICES=1 scripts/run_llava_ve.sh` (1 video, 16–128f, video-embeds,
--max-num-batched-tokens 32768) → `fig_internvl8b.py --model llava-ov-7b --dataset nextqa --frame 128`.

Before any GPU run: confirm H100 (GPU 1) idle (shared w/ `ljh`,`chani227`), `CUDA_VISIBLE_DEVICES=1`,
`conda activate vlmcost`. After a vLLM job: reap the orphan `VLLM::EngineCore` (Section 2).