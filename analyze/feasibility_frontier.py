"""Qwen feasibility frontier: max_n_vis per (model, batch) from results/feasibility.csv,
then max_frames per RESOLUTION via max_frames = max_n_vis / tok_per_frame(res).
(n_vis ceiling is resolution-independent — verified; resolution only sets tok/frame.)"""
import csv
from collections import defaultdict

# measured real video tok/frame (max_pixels=768*28*28); capped (1280x720 & 4K) ~360
TPF = {
    "qwen2.5-vl-7b": {"320x240": 70, "640x480": 195, "1280x720(capped)": 360},
    "qwen3-vl-8b":   {"320x240": 40, "640x480": 150, "1280x720(capped)": 360},
}

rows = [r for r in csv.DictReader(open("results/feasibility.csv")) if r["model"].startswith("qwen")]
# max OK n_vis per (model, batch); also record the FAIL point
maxnv = defaultdict(int); maxfr = defaultdict(int); failat = {}
for r in rows:
    k = (r["model"], int(r["batch"]))
    if r["status"] == "OK" and int(r["n_vis"]) > maxnv[k]:
        maxnv[k] = int(r["n_vis"]); maxfr[k] = int(r["frame"])
    if r["status"] == "FAIL":
        failat[k] = (int(r["frame"]), r["detail"][:30])

for model in ["qwen3-vl-8b", "qwen2.5-vl-7b"]:
    batches = sorted({b for (m, b) in maxnv if m == model})
    if not batches:
        continue
    print(f"\n==== {model}: feasibility frontier ====")
    print(f'  {"batch":>5} {"max_n_vis":>9} {"@frame(capped)":>14} {"FAIL@":>16}', end="")
    for res in TPF[model]:
        print(f' {("maxF_"+res.split("(")[0]):>16}', end="")
    print()
    for b in batches:
        k = (model, b); nv = maxnv[k]
        fa = f"{failat[k][0]}f {failat[k][1]}" if k in failat else "STOPPED(lower-bd)"
        print(f'  {b:>5} {nv:>9} {maxfr[k]:>14} {fa:>16}', end="")
        for res, tpf in TPF[model].items():
            print(f' {nv // tpf:>16}', end="")
        print()
    print("  (maxF_<res> = max frames at that resolution = max_n_vis / tok_per_frame; capped=1280x720/4K)")
