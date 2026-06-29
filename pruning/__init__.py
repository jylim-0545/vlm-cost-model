"""Vision-token pruning ported from our pruning study, packaged for the cost model.

Two layers, deliberately decoupled by their dependencies (see pruning/README.md):

- `pruning.cost`    — keep-ratio -> storage/break-even projection. PURE arithmetic
                      (config byte math + tier costs). NO torch/transformers/GPU.
                      Safe to import in the `vlmcost` env.
- `pruning.methods` — the real salience / SparseVLM scorers + splice/mask apply.
                      Needs transformers 4.57.x (patches eager_attention_forward) and
                      a GPU; imports torch/transformers LAZILY (only when a Pruner is
                      constructed), so importing the package never drags them in.

Importing this package (or `pruning.cost`) imports neither torch nor transformers.
"""
