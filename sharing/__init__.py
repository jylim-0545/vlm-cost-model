"""Vision-token SHARING, ported from our token-sharing study, packaged for the cost model.

One shared SigLIP **hub** encodes an image once; a light per-backbone **adapter** maps the
hub tokens into each frozen VLM backbone's vision-token space (hub-and-spoke). This module
(1) PRICES that — "encode once, serve N" compute + canonical-TokenStore storage — as the
cost model's E4 accounting, and (2) reproduces the adapter TRAINING (ridge / mlp_recon /
mlp_e2e, multi-task) and the recovery-vs-native measurement.

Layers, decoupled by dependency (see sharing/README.md):

- `sharing.cost`     — encode/storage/break-even projection. PURE arithmetic (config byte
                       math + tier costs). NO torch/transformers/GPU. Safe anywhere.
- `sharing.adapters` — adapter modules + closed-form ridge fit. Pure `torch` (no model,
                       no GPU needed); unit-testable on its own.
- `sharing.methods`  — hub encoder + backbone injection (embed-splice / qwen-patch /
                       vtpatch). Loads real VLMs on GPU; imports torch/transformers LAZILY.
- `sharing.train`    — AdapterTrainer: ridge / mlp_recon / mlp_e2e (+ recon-anchor, cosine,
                       multi-task) + recovery-vs-native eval. GPU.

Importing this package (or `sharing.cost`) imports neither torch nor transformers.
"""
