"""Vision-token pruning methods, ported faithfully from our pruning study
`scripts/qaware_vqa.py` (commit e2b8a3a). Two query-dependent selection signals:

  salience   — query-AGNOSTIC, FastV-style. LLM attention RECEIVED by each visual
               token under a neutral prompt ("Describe the image."), summed over
               heads + all query rows, over the LOWER decoder layers (range(2,8)).
               In deployment the score is precomputed once at INGEST and stored, so
               serve-time pruning is just a stored top-k.
  sparsevlm  — query-AWARE (the SparseVLM regime). Same attention readout but under
               the ACTUAL question, query rows restricted to TEXT tokens
               (ids != image), over the UPPER layers (range(12,24)) — i.e. how much
               the answer-generating text attends to each visual token.

Both rank visual tokens by score, keep the top-k (re-sorted by position so temporal
order survives), and apply the cut by SPLICE (drop dropped tokens from the embedding
sequence; InternVL / LLaVA) or MASK (zero the dropped columns in the attention mask;
Qwen3, whose DeepStack + mrope forbid an embeds splice).

ENV: this module patches `transformers ... eager_attention_forward` and loads a real
VLM on GPU. Verified on transformers 4.57.6 AND 5.9, so it runs in the cost repo's own
`vlmcost`/vLLM env — no separate env needed. Scoring uses the HF eager-attention path
(the vLLM *engine* itself can't expose per-token attention, but HF transformers lives
in the same env). torch/transformers are imported lazily (inside Pruner / the scorers)
so `import pruning.methods` is cheap; only constructing a Pruner needs them.
See pruning/README.md.
"""
from __future__ import annotations

# --- constants (qaware_vqa.py:43-49) -----------------------------------------
SAL_LAYERS = list(range(2, 8))      # salience: lower decoder layers, neutral prompt
REL_LAYERS = list(range(12, 24))    # sparsevlm: upper layers, text->visual attention
KEEPS = [0.50, 0.333, 0.222, 0.111, 0.056]   # 192/128/64/32/16 of 576 (SparseVLM grid)
PROMPT = "\nAnswer the question using a single word or phrase."

# which (repo_id, LM attention module to patch, image resize) per model (qaware_vqa.py:29-41)
_MODELS = {
    "llava15":  ("llava-hf/llava-1.5-7b-hf",      "transformers.models.llama.modeling_llama",        None),
    "internvl": ("OpenGVLab/InternVL3_5-8B-HF",   "transformers.models.qwen3.modeling_qwen3",        (448, 448)),
    "qwen":     ("Qwen/Qwen3-VL-8B-Instruct",     "transformers.models.qwen3_vl.modeling_qwen3_vl",  (448, 448)),
}
_SPLICE = {"llava15", "internvl"}   # the rest (qwen) use MASK mode


def select_topk(scores, k: int):
    """Top-k visual tokens by score, re-sorted by original index (qaware_vqa.py:298).
    Pure tensor op — model-agnostic, unit-testable without a model."""
    import torch
    k = max(1, min(int(k), scores.numel()))
    return torch.argsort(scores, descending=True)[:k].sort().values


def splice_keep_mask(ids, img_id: int, k: int):
    """Boolean keep-mask over the full token sequence that drops all but the first k
    image-placeholder positions (qaware_vqa.py:146-148). Factored out so the splice
    bookkeeping is unit-testable without a model. `ids` is 1-D input_ids."""
    import torch
    pad = (ids == img_id).nonzero(as_tuple=True)[0]
    mask = torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)
    mask[pad] = False
    mask[pad[:k]] = True
    return mask


class Pruner:
    """Holds a loaded VLM + processor and exposes the two scorers and the two apply
    modes. Construction loads the model (GPU, transformers 4.57.x)."""

    def __init__(self, which: str = "internvl", device: str = "cuda"):
        if which not in _MODELS:
            raise ValueError(f"unknown model '{which}'; choose from {list(_MODELS)}")
        import importlib
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.which = which
        self.device = device
        repo_id, lm_mod_path, self.resize = _MODELS[which]
        self.splice = which in _SPLICE
        self.lm_mod = importlib.import_module(lm_mod_path)   # the module we monkeypatch
        print(f"[pruning] load {repo_id}  mode={'splice' if self.splice else 'mask'}", flush=True)
        self.proc = AutoProcessor.from_pretrained(repo_id)
        # .to(device) instead of device_map= so we don't require `accelerate`
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").eval().to(device)
        m = self.model
        lm = m.model.language_model if hasattr(m.model, "language_model") else m.language_model
        self.lm_cfg = lm.config
        self.img_id = getattr(m.config, "image_token_id", None) or m.config.image_token_index
        self.emb = m.get_input_embeddings()
        self.hid = m.config.get_text_config().hidden_size

    # ---- input construction (qaware_vqa.py:81-95) ----------------------------
    def build(self, img, prompt: str):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        txt = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        text = txt if self.which == "internvl" else [txt]
        return self.proc(text=text, images=[img], return_tensors="pt").to(self.device)

    def feats_of(self, inputs):
        """Post-projector vision tokens [n_vis, hidden] (qaware_vqa.py:86-95)."""
        import torch
        with torch.no_grad():
            if self.which == "qwen":
                emb, _deep = self.model.get_image_features(
                    pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
                emb = torch.cat(list(emb), 0) if isinstance(emb, (tuple, list)) else emb
                return emb.reshape(-1, self.hid)
            out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            return self._extract_feats(out)

    def _extract_feats(self, out):
        """Pull the post-projector token tensor [*, hidden] out of get_image_features's
        return, robustly across transformers versions: 4.57 returns a tensor/tuple;
        5.9 returns an output object whose post-projector embeds live in `.pooler_output`
        (`.last_hidden_state` is the pre-projector ViT output). Pick the tensor whose last
        dim == LLM hidden."""
        import torch
        if torch.is_tensor(out):
            cands = [out]
        elif isinstance(out, (tuple, list)):
            cands = list(out)
        else:  # ModelOutput-like
            cands = [getattr(out, a, None) for a in ("pooler_output", "image_embeds",
                                                     "last_hidden_state")]
        for t in cands:                                   # prefer the post-projector tensor
            if torch.is_tensor(t) and t.shape[-1] == self.hid:
                return t.reshape(-1, self.hid)
        for t in cands:                                   # fallback: first tensor
            if torch.is_tensor(t):
                return t.reshape(-1, self.hid)
        raise TypeError(f"get_image_features returned {type(out).__name__}; "
                        "cannot find a post-projector token tensor")

    # ---- the shared attention readout (qaware_vqa.py:96-112) ------------------
    def _attn_to_visual(self, inputs, layers, query_rows=None):
        import torch
        ids = inputs["input_ids"][0]
        vis = (ids == self.img_id).nonzero(as_tuple=True)[0]
        rows = query_rows if query_rows is not None else torch.ones(
            ids.numel(), dtype=torch.bool, device=ids.device)
        st = {"layer": 0, "recv": torch.zeros(vis.numel(), device=self.device, dtype=torch.float32),
              "vis": vis, "rows": rows}
        orig = self.lm_mod.eager_attention_forward

        def patched(module, *a, **kw):
            out, aw = orig(module, *a, **kw)
            li = st["layer"]; st["layer"] += 1
            if li in layers and aw is not None and aw.dim() == 4:
                A = aw[0].float().sum(0)                       # sum heads -> [seq, seq]
                st["recv"] += A[st["rows"]][:, st["vis"]].sum(0)   # text rows -> per visual token
            return out, aw

        self.lm_mod.eager_attention_forward = patched
        self.lm_cfg._attn_implementation = "eager"
        try:
            with torch.no_grad():
                o = self.model(**inputs, use_cache=False); del o
        finally:
            self.lm_mod.eager_attention_forward = orig
            self.lm_cfg._attn_implementation = "sdpa"
        torch.cuda.empty_cache()
        return st["recv"] / len(layers)

    def score_salience(self, neutral_inputs):
        """Query-agnostic salience for `build(img, "Describe the image.")` inputs."""
        return self._attn_to_visual(neutral_inputs, SAL_LAYERS)

    def score_sparsevlm(self, question_inputs):
        """Query-aware SparseVLM signal for `build(img, question + PROMPT)` inputs."""
        ids = question_inputs["input_ids"][0]
        return self._attn_to_visual(question_inputs, REL_LAYERS, query_rows=(ids != self.img_id))

    # ---- apply the cut + answer (qaware_vqa.py:143-182) ----------------------
    def answer_splice(self, inp, kept_feats):
        import torch
        with torch.no_grad():
            ids = inp["input_ids"][0]; k = kept_feats.shape[0]
            mask = splice_keep_mask(ids, self.img_id, k)
            nid = ids[mask]
            emb = self.emb(nid).clone()
            emb[(nid == self.img_id)] = kept_feats.to(emb.dtype)
            out = self.model.generate(
                inputs_embeds=emb.unsqueeze(0),
                attention_mask=torch.ones(1, nid.numel(), device=ids.device),
                max_new_tokens=8, do_sample=False)
            return self.proc.tokenizer.decode(out[0], skip_special_tokens=True)

    def answer_mask(self, inp, keep_idx):
        """MASK mode (Qwen mrope+deepstack): drop visual tokens by blocking their
        attention COLUMNS, keeping the diagonal so rope sees every position."""
        import torch
        with torch.no_grad():
            ids = inp["input_ids"]; L = ids.shape[1]; dev = ids.device
            vis = (ids[0] == self.img_id).nonzero(as_tuple=True)[0]
            keepbool = torch.zeros(vis.numel(), dtype=torch.bool, device=dev); keepbool[keep_idx] = True
            drop = vis[~keepbool]
            NEG = torch.finfo(self.model.dtype).min
            m = torch.zeros(L, L, device=dev, dtype=self.model.dtype)
            m.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1), NEG)
            m[:, drop] = NEG; m[drop, drop] = 0.0
            gi = {k: v for k, v in inp.items() if k != "attention_mask"}
            out = self.model(**gi, attention_mask=m[None, None], use_cache=True)
            past = out.past_key_values; logits = out.logits[0, -1]
            eos = self.proc.tokenizer.eos_token_id
            keepcol = torch.ones(L, dtype=torch.long, device=dev); keepcol[drop] = 0
            toks = []
            for _ in range(8):
                nxt = int(logits.argmax())
                if nxt == eos:
                    break
                toks.append(nxt)
                keepcol = torch.cat([keepcol, torch.ones(1, dtype=torch.long, device=dev)])
                out = self.model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past,
                                 attention_mask=keepcol[None], use_cache=True)
                past = out.past_key_values; logits = out.logits[0, -1]
            return self.proc.tokenizer.decode(toks, skip_special_tokens=True)

    def prune(self, feats, scores, keep: float):
        """Prune ONLY — no generation. Select the top-k vision tokens by `scores` and
        return (kept_feats [k, hid], keep_idx [k]). Use this when you want the pruned
        tokens themselves (to store / hand to another pipeline); pair with
        answer_splice/answer_mask (or your own model) to actually decode."""
        k = max(1, int(round(feats.shape[0] * keep)))
        idx = select_topk(scores, k)
        return feats[idx], idx

    def prune_and_answer(self, inp, feats, scores, keep: float):
        """Convenience: prune() then apply via splice/mask and return the decoded answer."""
        kept, idx = self.prune(feats, scores, keep)
        return self.answer_splice(inp, kept) if self.splice else self.answer_mask(inp, idx)

    # ---- latency measurement (cuda events) -----------------------------------
    def _cuda_time_ms(self, fn, runs: int = 7, warmup: int = 3) -> float:
        """Median GPU ms of `fn` over `runs` (warmup discarded). Real wall on the
        device via cuda Events + synchronize."""
        import torch
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2]

    def encode_latency(self, inputs, runs: int = 7, warmup: int = 3) -> float:
        """ms for the vision tower (ViT + projector) — constant w.r.t. pruning, since
        you encode the full frames once and prune the OUTPUT."""
        return self._cuda_time_ms(lambda: self.feats_of(inputs), runs, warmup)

    def prefill_latency(self, inp, kept_feats, runs: int = 7, warmup: int = 3) -> tuple[float, int]:
        """ms for a single prefill forward (to first-token logits) over the spliced
        sequence of `kept_feats` vision tokens + text. Returns (ms, seq_len).
        SPLICE models only — mask mode keeps the sequence length (no prefill saving)."""
        import torch
        if not self.splice:
            raise ValueError("prefill_latency measures the splice-mode saving; mask-mode "
                             "(qwen) keeps full sequence length, so prefill does not shrink")
        ids = inp["input_ids"][0]; k = kept_feats.shape[0]
        mask = splice_keep_mask(ids, self.img_id, k)
        nid = ids[mask]
        emb = self.emb(nid).clone()
        emb[(nid == self.img_id)] = kept_feats.to(emb.dtype)
        am = torch.ones(1, nid.numel(), device=ids.device)

        def fwd():
            with torch.no_grad():
                self.model(inputs_embeds=emb.unsqueeze(0), attention_mask=am, use_cache=True)

        return self._cuda_time_ms(fwd, runs, warmup), int(nid.numel())

    def prefill_latency_nvis(self, n_vis: int, runs: int = 7, warmup: int = 3,
                             prompt: str = "Answer:") -> tuple[float, int]:
        """Real LLM prefill ms over `n_vis` (synthetic) vision tokens + a short text
        prompt. Latency is content-independent, so random vision embeddings are fine.
        This reaches the VIDEO-scale regime the cost model targets (n_vis up to tens of
        thousands), where prefill is super-linear in n_vis — unlike a single 256-token
        image, where prefill is weight-bandwidth bound and barely moves. Returns
        (ms, seq_len)."""
        import torch
        tok = self.proc.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            txt = self.emb(tok)[0]                                  # [t, hid]
        vis = torch.randn(int(n_vis), self.hid, device=self.device, dtype=txt.dtype) * 0.02
        emb = torch.cat([txt, vis, txt[-1:]], 0).unsqueeze(0)       # [1, t+n_vis+1, hid]
        am = torch.ones(1, emb.shape[1], device=self.device)

        def fwd():
            with torch.no_grad():
                self.model(inputs_embeds=emb, attention_mask=am, use_cache=True)

        return self._cuda_time_ms(fwd, runs, warmup), int(emb.shape[1])
