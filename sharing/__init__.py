"""Vision-token SHARING — reproduces our token-sharing experiment inside this repo.

We cache vision tokens to skip re-encoding. But each VLM uses its own vision tokens, so a
cache helps only one model. This module reproduces the study that asks: can ONE shared
encoding (a SigLIP "hub") drive MANY frozen VLM backbones through a small per-backbone
adapter? If so, one cached token set could serve several models. The deliverable here is
the ACCURACY side — how well a shared+adapted token recovers each backbone's native
accuracy (the cost side is handled separately).

Layers, decoupled by dependency (see sharing/README.md):

- `sharing.adapters` — adapter modules (`RidgeAffine`, `ZScoreMLP`) + closed-form ridge fit
                       + a loader for the study's pre-trained adapters. Pure `torch`
                       (no model, no GPU); unit-testable on its own.
- `sharing.methods`  — `HubShare`: hub encoder + backbone load + injection patch. Loads
                       real VLMs on GPU; imports torch/transformers LAZILY.
- `sharing.train`    — `AdapterTrainer`: ridge / mlp_recon / mlp_e2e (+ recon-anchor,
                       cosine, multi-task) + recovery-vs-native and forgetting eval. GPU.

Importing this package does not import torch or transformers.
"""
