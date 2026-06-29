"""Accuracy sanity check for the ported pruning (REAL model, GPU).

For a handful of image-questions it computes the salience (query-agnostic) and
SparseVLM (query-aware) scores, prunes to a few keep-ratios, answers with the pruned
tokens, and prints accuracy per (method, keep). This validates that the PORT is
faithful — that keeping the top-k tokens still answers, and that the query-aware
signal holds up better at low keep (the query-aware finding).

  # GPU env (transformers 4.57 or 5.9, e.g. the repo's vlmcost):
  python -m pruning.demo_accuracy --which internvl --tsv /path/to/textvqa.tsv --bench textvqa --n 20

It is image VQA (single image, fast) — only an algorithm-faithfulness check, NOT the
video cost story. Reuses the same TSV format as qaware_vqa.py
(columns: index, image[base64], question, answer).

Runs on transformers 4.57 and 5.9 (incl. the vlmcost/vLLM env); needs a GPU. See pruning/README.md.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import re
from collections import defaultdict

# --- benchmark scoring, ported from qaware_vqa.py:51-66 -----------------------
_ARTICLES = {"a", "an", "the"}


def vqa_norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.strip().lower())
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def score_ans(pred: str, gt: str, bench: str) -> float:
    p = vqa_norm(pred)
    if bench == "textvqa":
        try:
            anns = [vqa_norm(a) for a in ast.literal_eval(gt)]
        except Exception:
            anns = [vqa_norm(gt)]
        p1 = p.split()[0] if p.split() else p
        cand = {p, p1}
        m = max(sum(1 for a in anns if a == c) for c in cand)
        return min(1.0, m / 3.0)
    g = vqa_norm(gt); toks = p.split()
    return float(g == p or g in toks)


def main() -> None:
    import csv
    import sys
    from PIL import Image
    from pruning.methods import Pruner, PROMPT

    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))   # base64 image cells are huge

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="internvl", choices=["internvl", "llava15", "qwen"])
    ap.add_argument("--tsv", required=True, help="benchmark TSV (qaware_vqa format)")
    ap.add_argument("--bench", default="textvqa", choices=["textvqa", "gqa"])
    ap.add_argument("--n", type=int, default=20, help="number of image-questions")
    ap.add_argument("--keeps", default="1.0,0.5,0.25", help="keep-ratios to test")
    a = ap.parse_args()
    keeps = [float(x) for x in a.keeps.split(",")]

    pr = Pruner(a.which)

    def b64img(s):
        img = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        return img.resize(pr.resize) if pr.resize else img

    with open(a.tsv, newline="") as f:
        rows = []
        for d in csv.DictReader(f, delimiter="\t"):
            rows.append(d)
            if len(rows) >= a.n:
                break
    # resolve base64 (some rows reference another row's image by index)
    real = {str(d["index"]): d["image"] for d in rows if len(str(d["image"])) > 100}
    for d in rows:
        d["_b64"] = d["image"] if len(str(d["image"])) > 100 else real.get(str(d["image"]), d["image"])

    corr = defaultdict(float)   # (keep, method) -> sum score
    full = 0.0
    n = 0
    for r in rows:
        img = b64img(r["_b64"]); q = str(r["question"]); gt = str(r["answer"])
        neu = pr.build(img, "Describe the image.")
        feats = pr.feats_of(neu)
        sal = pr.score_salience(neu)
        qi = pr.build(img, q + PROMPT)
        rattn = pr.score_sparsevlm(qi)
        inp = pr.build(img, q + PROMPT)
        scores = {"salience": sal, "sparsevlm": rattn}
        full += score_ans(pr.prune_and_answer(inp, feats, sal, 1.0), gt, a.bench)
        for keep in keeps:
            for meth, sc in scores.items():
                ans = pr.prune_and_answer(inp, feats, sc, keep)
                corr[(keep, meth)] += score_ans(ans, gt, a.bench)
        n += 1
        print(f"  q{n}/{len(rows)} done", flush=True)

    print(f"\naccuracy sanity — {a.which} / {a.bench}  (n={n}, n_vis={feats.shape[0]})")
    print(f"  full (no prune): {100*full/n:.1f}%")
    print(f"  {'keep':>6} {'salience':>10} {'sparsevlm':>10}")
    for keep in keeps:
        print(f"  {keep:>6.3f} {100*corr[(keep,'salience')]/n:>9.1f}% "
              f"{100*corr[(keep,'sparsevlm')]/n:>9.1f}%")
    print("\nexpected: full≈ceiling; salience holds near keep=0.5 then "
          "degrades; sparsevlm ≥ salience at low keep (query-aware).")


if __name__ == "__main__":
    main()
