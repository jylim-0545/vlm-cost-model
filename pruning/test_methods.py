"""GPU-free unit checks for the model-agnostic pieces of pruning.methods.

Validates the selection + splice bookkeeping on synthetic tensors — no model, no
transformers. Needs only torch (present in `vlmcost`). Run:
    python -m pruning.test_methods
"""
from __future__ import annotations

import torch

from pruning.methods import select_topk, splice_keep_mask


def test_select_topk_picks_highest_and_keeps_order():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.05])
    idx = select_topk(scores, 3)                       # top-3 are positions 1,3,2
    assert idx.tolist() == [1, 2, 3], idx.tolist()     # returned sorted by index (order preserved)


def test_select_topk_clamps_k():
    scores = torch.rand(10)
    assert select_topk(scores, 0).numel() == 1         # k>=1
    assert select_topk(scores, 99).numel() == 10       # k<=N
    # no duplicate indices
    idx = select_topk(scores, 5)
    assert idx.unique().numel() == idx.numel()


def test_splice_keep_mask_drops_all_but_first_k_image_tokens():
    IMG = 999
    ids = torch.tensor([5, IMG, IMG, IMG, IMG, 7, 8])  # 4 image placeholders
    mask = splice_keep_mask(ids, IMG, k=2)
    # text tokens always kept; first 2 image positions kept; last 2 dropped
    assert mask.tolist() == [True, True, True, False, False, True, True], mask.tolist()
    # exactly k image tokens survive
    assert int((ids[mask] == IMG).sum()) == 2


def test_splice_keep_mask_k_equals_all():
    IMG = 1
    ids = torch.tensor([0, IMG, IMG, 2])
    mask = splice_keep_mask(ids, IMG, k=2)
    assert mask.all()


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
