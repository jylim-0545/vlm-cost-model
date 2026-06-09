"""Does encode latency depend on RESOLUTION at matched n_vis? (Qwen only.)
Reads results/qwen_res_encode/reuse_real.csv; encode = median(cold ttft) - median(vt_reuse ttft_inject).
Prints encode-vs-n_vis per resolution; if curves overlay -> n_vis sufficient, if the capped
(1280x720) curve sits above at matched n_vis -> resolution is an independent axis."""
import csv, statistics, sys
from collections import defaultdict

CSV = "results/qwen_res_encode/reuse_real.csv"
RES = {"3402648543": "320x240", "4561357748": "640x480", "game_33": "1280x720"}

rows = list(csv.DictReader(open(CSV)))
# (model, video_stem, frames) -> {metric: [values]}
agg = defaultdict(lambda: defaultdict(list))
nvis_of = {}
for r in rows:
    vid = r["video_id"].split("_b")[0]
    key = (r["model"], vid, int(r["frames"]))
    if r["variant"] == "cold" and r["metric"] == "ttft":
        agg[key]["cold"].append(float(r["value_ms"]))
    elif r["variant"] == "vt_reuse" and r["metric"] == "ttft_inject":
        agg[key]["vt"].append(float(r["value_ms"]))
    nvis_of[key] = int(r["n_vis"])

by_model = defaultdict(list)
for (model, vid, fr), m in agg.items():
    if "cold" in m and "vt" in m:
        cold = statistics.median(m["cold"]); vt = statistics.median(m["vt"])
        res = RES.get(vid, vid)
        by_model[model].append((nvis_of[(model, vid, fr)], res, fr, cold, vt, cold - vt))

for model in sorted(by_model):
    print(f"\n==== {model}: encode = cold_ttft - vt_inject  (sorted by n_vis) ====")
    print(f'  {"n_vis":>6} {"res":>10} {"frames":>6} {"cold":>8} {"vt":>8} {"encode":>8} {"enc/n_vis(us)":>14}')
    for nvis, res, fr, cold, vt, enc in sorted(by_model[model]):
        print(f'  {nvis:>6} {res:>10} {fr:>6} {cold:>8.1f} {vt:>8.1f} {enc:>8.1f} {enc/nvis*1000:>14.2f}')
    # matched-n_vis comparison: bucket by nearest 2500
    print(f"  -- matched-n_vis encode (us/token) by resolution --")
    buckets = defaultdict(dict)
    for nvis, res, fr, cold, vt, enc in by_model[model]:
        b = round(nvis / 2500) * 2500
        buckets[b][res] = enc / nvis * 1000
    for b in sorted(buckets):
        cells = " ".join(f"{r}={buckets[b].get(r, float('nan')):.1f}" for r in ["320x240", "640x480", "1280x720"] if r in buckets[b])
        print(f"  n_vis~{b}: {cells}")
