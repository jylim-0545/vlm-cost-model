"""Hub encoder + backbone injection for vision-token sharing, ported faithfully from our
token-sharing study (`EfficientVLM/scripts/{d6_adapteronly,d12_holistic,e1_stagea}.py`).

A `HubShare` holds a loaded VLM **backbone** (reader, FROZEN) + the shared SigLIP **hub**
encoder, and exposes the two things the trainer/evaluator need:

  - hub_on_pixels / hub_encode  : run the hub on an image (or the backbone's own tiles) ->
                                  [*, 1152] hub tokens.
  - inject(adapter)             : install a forward-patch so the backbone reads
                                  `adapter(hub_tokens)` in place of its native vision tokens.
  - native_vt_on_pixels         : the backbone's OWN vision-tower output (the recon target
                                  for ridge/MLP fitting, and the native baseline).
  - forced_choice / accuracy    : multiple-choice letter scoring (paired native vs adapter).

Injection modes (per backbone):
  vtpatch  (LLaVA-OV, same-family): hub runs on OV's own pixel tiles; adapter maps
           hub-VT-out(1152) -> OV-VT-out(1152); OV's native projector+anyres run after.
           This is the path with ALL of ridge/mlp_recon/mlp_e2e + multi-task (d6/d12).
  embed    (InternVL, cross-family): hub tokens resampled to the backbone grid; adapter maps
           1152 -> post-projector HID; spliced into input-embeds at image positions (e1).
           Used for the cross-encoder GENERALIZATION story (ridge; see FINDINGS §family).

ENV: loads a real VLM + SigLIP on GPU (verified on the study's transformers 4.57 box,
4×RTX 4090). torch/transformers/PIL are imported LAZILY (inside the constructor / methods)
so `import sharing.methods` is cheap (only constructing a HubShare loads the models).
See sharing/README.md for which model/data paths are expected.
"""
from __future__ import annotations

import glob
import os

# ---- model/data path resolution (defaults point at the EfficientVLM checkout this repo
#      is nested under; override via the SHARE_* env vars) -----------------------------
_HF_ROOTS = [
    os.environ.get("SHARE_HF_ROOT", ""),
    "/home/yhlee/EfficientVLM/hf_cache/hub",
    "/mnt/nas/VLM/hf/hub",
    "/mnt/nas/yhlee/hf_cache/hub",
    os.path.expanduser("~/.cache/huggingface/hub"),
]


def _snap(p: str) -> str:
    for s in glob.glob(p + "/snapshots/*"):
        if glob.glob(s + "/*.safetensors") or glob.glob(s + "/*.bin"):
            return s
    return p


def _find_hf(name: str) -> str | None:
    """Resolve a HF cache dir `models--<name>` across the known roots -> snapshot path."""
    for r in _HF_ROOTS:
        if not r:
            continue
        hits = glob.glob(f"{r}/models--{name}")
        if hits:
            return _snap(hits[0])
    return None


HUB_NAME = "google--siglip-so400m-patch14-384"
HUB_DIM = 1152
HUB_TOKENS = 729

# backbone registry: (repo resolver, injection mode, hub-input resize, trust_remote_code)
_BACKBONES = {
    "llavaov":   dict(name="llava-hf--llava-onevision-qwen2-7b-ov-hf", mode="vtpatch", res=384, trust=False),
    "internvl8": dict(name="OpenGVLab--InternVL3_5-8B-HF", mode="embed", res=448, trust=True),
    "internvl4": dict(name="OpenGVLab--InternVL3_5-4B-HF", mode="embed", res=448, trust=True),
}
LETTERS5 = ["A", "B", "C", "D", "E"]


def _resolve_backbone_repo(name: str) -> str:
    p = _find_hf(name)
    if p:
        return p
    # fall back to the hub id (last path component, de-mangled)
    return name.replace("--", "/", 1)


class HubShare:
    """Loaded backbone (frozen) + SigLIP hub + injection patch. Construct once (loads both
    models on `device`); set an adapter with `inject(adapter)`, then call the model normally
    — the patch substitutes `adapter(hub_tokens)` for the native vision tokens while enabled.
    `inject(None)` / `with hubshare.native():` restores the native path (the baseline)."""

    def __init__(self, backbone: str = "llavaov", device: str = "cuda"):
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone '{backbone}'; choose from {list(_BACKBONES)}")
        import torch
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  SiglipVisionModel)
        self.torch = torch
        cfg = _BACKBONES[backbone]
        self.backbone = backbone
        self.mode = cfg["mode"]
        self.res = cfg["res"]
        self.device = device
        repo = _resolve_backbone_repo(cfg["name"])
        print(f"[sharing] load backbone {backbone}: {repo}  (mode={self.mode})", flush=True)
        self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=cfg["trust"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo, dtype=torch.bfloat16, device_map=device,
            trust_remote_code=cfg["trust"], attn_implementation="sdpa").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)                       # reader FULLY frozen
        self.tok = self.proc.tokenizer if hasattr(self.proc, "tokenizer") else self.proc
        self.hid = self.model.config.get_text_config().hidden_size
        self.img_id = (getattr(self.model.config, "image_token_id", None)
                       or getattr(self.model.config, "image_token_index", None))
        self.emb = self.model.get_input_embeddings()

        hub_repo = _find_hf(HUB_NAME) or "google/siglip-so400m-patch14-384"
        print(f"[sharing] load hub SigLIP: {hub_repo}  (HID={self.hid}, vt_dim={self.vt_dim})",
              flush=True)
        self.hub = SiglipVisionModel.from_pretrained(hub_repo, dtype=torch.bfloat16).to(device).eval()
        for p in self.hub.parameters():
            p.requires_grad_(False)

        self._adapter = None
        self._enable = False
        self._install_patch()

    # ---- the dimension the adapter must OUTPUT --------------------------------
    @property
    def vt_dim(self) -> int:
        """Adapter output width: OV reads SigLIP-VT tokens (1152, projector runs after);
        embed-splice backbones read post-projector tokens (HID)."""
        return HUB_DIM if self.mode == "vtpatch" else self.hid

    # ---- hub encoding ---------------------------------------------------------
    def _hub_pixels(self, img, size: int = 384):
        """SigLIP preprocessing: resize to `size`, scale to [-1,1] (mean=std=0.5)."""
        import numpy as np
        torch = self.torch
        a = torch.from_numpy(np.asarray(img.convert("RGB").resize((size, size)),
                                        dtype=np.float32) / 255.0)
        return ((a - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.bfloat16)

    def hub_encode(self, img):
        """Single image -> [729, 1152] hub tokens (float, on device)."""
        torch = self.torch
        with torch.no_grad():
            return self.hub(pixel_values=self._hub_pixels(img)).last_hidden_state[0].float()

    def hub_on_pixels(self, pv):
        """Run the hub on a backbone pixel_values tensor (OV tiles) -> [T*729, 1152] float.
        Reshapes (T, C, H, W) tiles through SigLIP. Used by the vtpatch path so hub and OV
        see the SAME tiles (per-token aligned, no resample)."""
        torch = self.torch
        with torch.no_grad():
            pv = pv.reshape(-1, *pv.shape[-3:]).to(self.device, torch.bfloat16)
            return self.hub(pixel_values=pv).last_hidden_state.float()    # [T,729,1152]

    def native_vt_on_pixels(self, pv):
        """The backbone's OWN vision-tower output on its tiles -> [T,729,1152] float
        (the recon target for ridge/MLP fitting). vtpatch backbones only."""
        torch = self.torch
        with torch.no_grad():
            return self._orig_vt(pixel_values=pv.reshape(-1, *pv.shape[-3:])).last_hidden_state.float()

    # ---- input construction ---------------------------------------------------
    def build(self, images, prompt: str):
        """Chat-template a single- or multi-image prompt -> processor inputs on device."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        ims = [im.convert("RGB").resize((self.res, self.res)) for im in images]
        content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": prompt}]
        conv = [{"role": "user", "content": content}]
        txt = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        return self.proc(text=[txt], images=ims, return_tensors="pt").to(self.device)

    # ---- adapter injection -----------------------------------------------------
    def inject(self, adapter):
        """Set the adapter to substitute for native vision tokens (and enable injection).
        Pass None to clear. The adapter's params keep their requires_grad (trainer trains
        them; hub features are detached so only the adapter learns)."""
        self._adapter = adapter
        self._enable = adapter is not None

    def enable(self, on: bool = True):
        self._enable = on and self._adapter is not None

    def native(self):
        """Context manager: temporarily disable injection (score the native baseline)."""
        return _Native(self)

    def _install_patch(self):
        """Monkeypatch the vision tower so, while enabled, its output is replaced by
        adapter(hub_tokens). vtpatch only (the documented training path). embed-mode does
        its injection at scoring time via splice() instead."""
        if self.mode != "vtpatch":
            self._orig_vt = None
            return
        VT = (self.model.model.vision_tower if hasattr(self.model.model, "vision_tower")
              else self.model.model.visual)
        self._orig_vt = VT.forward
        orig_vt = self._orig_vt
        share = self

        def patched_vt(*a, **kw):
            pv = kw.get("pixel_values", a[0] if a else None)
            out = orig_vt(pixel_values=pv, **{k: v for k, v in kw.items() if k != "pixel_values"})
            if share._enable and share._adapter is not None:
                new = share._adapter(share.hub_on_pixels(pv)).to(out.last_hidden_state.dtype)
                out.last_hidden_state = new
                if getattr(out, "hidden_states", None) is not None:
                    hs = list(out.hidden_states); hs[-1] = new; out.hidden_states = tuple(hs)
            return out
        VT.forward = patched_vt

    # ---- multiple-choice scoring (forced choice over answer letters) ----------
    def _letter_ids(self, letters):
        lid = {}
        for L in letters:
            s = set()
            for t in (L, " " + L):
                e = self.tok.encode(t, add_special_tokens=False)
                if e:
                    s.add(e[0])
            lid[L] = list(s)
        return lid

    def forced_choice(self, inp, letters=("A", "B", "C", "D")):
        """Return the argmax answer letter from the last-position logits."""
        torch = self.torch
        lid = self._letter_ids(letters)
        with torch.no_grad():
            lg = self.model(**inp, use_cache=False).logits[0, -1]
        best, bl = None, -1e9
        for L in letters:
            if not lid[L]:
                continue
            v = max(float(lg[i]) for i in lid[L])
            if v > bl:
                bl, best = v, L
        return best

    def target_token(self, letter: str) -> int:
        """Token id of ' <letter>' — the CE target for VQA-CE fine-tuning."""
        return self.tok.encode(" " + letter, add_special_tokens=False)[0]


class _Native:
    def __init__(self, share: HubShare):
        self.share = share

    def __enter__(self):
        self._prev = self.share._enable
        self.share._enable = False
        return self.share

    def __exit__(self, *exc):
        self.share._enable = self._prev
        return False
