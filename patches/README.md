# vLLM patches (jylim env, vLLM 0.22.0)

## llava_onevision_videoembeds.patch
Adds a `video_embeds` injection path to vLLM's `LlavaOnevisionForConditionalGeneration`
(it only had `pixel_values_videos`). Enables encode-skip vt_reuse for LLaVA-OV via the
VIDEO path (flat, 1 newline) instead of the IMAGE path (per-row newline → seq doubling).

Apply:  cd $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/model_executor/models
        patch -p0 < <repo>/patches/llava_onevision_videoembeds.patch    # or copy 4 edits manually
Edits:  schema LlavaOnevisionVideoEmbeddingInputs + _parse_and_validate_video_input
        + _get_mm_fields_config(video_embeds) + embed_multimodal(video_embeds branch)
Run:    reuse_real.py --model llava-ov-7b ... --max-num-batched-tokens 32768  (128f)
