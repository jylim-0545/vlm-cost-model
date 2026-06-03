"""vLLM smoke test: confirm a target VLM loads on the H100 and generates from a
synthetic image. No dataset needed — content is irrelevant for a load/run check.

Pins CUDA_VISIBLE_DEVICES=1 (H100) and asserts exactly one visible device
(CLAUDE.md Section 8). Quick smoke test only — enforce_eager skips CUDA-graph compile.

Usage:
  python -m measure.smoke_vllm                          # qwen2.5-vl-7b
  python -m measure.smoke_vllm --model internvl3.5-8b
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "must run on the H100 (GPU 1)"

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    spec = load_models().models[a.model]
    print(f"[smoke] model={spec.repo_id} trust_remote_code={spec.trust_remote_code}")

    proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    img = Image.new("RGB", (a.image_size, a.image_size), (120, 120, 120))
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is the dominant color of this image? Answer in one word."},
    ]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(
        model=spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,                 # skip compile for a fast smoke test
        limit_mm_per_prompt={"image": 1},
    )
    out = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": img}},
        SamplingParams(max_tokens=a.max_tokens, temperature=0.0),
    )
    text = out[0].outputs[0].text
    n_prompt = len(out[0].prompt_token_ids)
    print(f"[smoke] prompt_tokens={n_prompt}  gen_tokens={len(out[0].outputs[0].token_ids)}")
    print(f"[smoke] OUTPUT: {text!r}")
    print("[smoke] OK — model loaded and generated on the H100.")


if __name__ == "__main__":
    main()
