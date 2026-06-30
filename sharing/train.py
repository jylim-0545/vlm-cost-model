"""AdapterTrainer — fit a hub->backbone adapter and measure recovery vs native.

Unifies the study's two trainers into one library, for the LLaVA-OV (vtpatch) path that
carries the headline numbers (sharing/FINDINGS.md):
  - `EfficientVLM/scripts/d6_adapteronly.py`  : fine-grained, MMStar image MCQ.
  - `EfficientVLM/scripts/d12_holistic.py`    : holistic, NExT-QA video MCQ.

Adapter modes (the "raw -> recon -> E2E" ladder; reader ALWAYS frozen):
  raw        z-scored hub tokens injected directly (RidgeAffine identity). NO training.
  ridge      closed-form z-affine token-matching to native VT (label-free Stage-A). Seconds.
  mlp_recon  ZScoreMLP pretrained by MSE to mimic native VT tokens (label-free).
  mlp_e2e    mlp_recon, THEN VQA gold-letter CE fine-tune, with optional recon-anchor
             (loss += lambda * ||adapter(hub) - nativeVT||^2, keeps generality / cuts
             forgetting) and cosine+warmup schedule. Best adapter-only result (FINDINGS).

Extras: multi-task (mix a second MCQ task into E2E -> trade-off, halves forgetting),
optional forgetting eval on a held-out OTHER task. All knobs live in `ShareTrainConfig`
(no env-var soup). Recovery = adapter_acc / native_acc (the study's stable metric).

GPU + transformers (the study's 4.57 box / 4×4090). Heavy imports are lazy via
`sharing.methods`. See sharing/README.md to run; sharing/sweep.py for variant×seed sweeps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ShareTrainConfig:
    mode: str = "ridge"              # raw | ridge | mlp_recon | mlp_e2e
    hidden: int = 2048               # ZScoreMLP width (D6_MLP_H)
    ridge_lam: float = 1.0
    # recon stats/pretrain corpus
    pre_samples: int = 400           # train items used for z-score stats + recon pairs
    recon_subsample: int | None = 256  # tokens kept per sample for recon (None = all)
    pre_steps: int = 2000            # MLP recon-pretrain steps
    pre_lr: float = 1e-3
    # E2E VQA-CE fine-tune
    ft_steps: int = 600
    ft_lr: float = 3e-5
    accum: int = 8
    recon_lambda: float = 0.0        # recon-anchor weight (d6/d12: 8 for holistic)
    sched: str = "cosine"            # const | cosine
    warmup: int = 200
    grad_clip: float = 1.0
    # eval / data
    n_eval: int = 400
    seed: int = 0


@dataclass
class Task:
    """A multiple-choice task: lists of {images, prompt, gold} dicts + the letter set."""
    name: str
    letters: list
    train: list = field(default_factory=list)
    eval: list = field(default_factory=list)


# ----------------------------- datasets -------------------------------------
def mmstar_task(tsv: str, n_eval: int = 400, seed: int = 0, n_train: int | None = None) -> Task:
    """MMStar fine-grained image MCQ (4-choice). Deterministic eval split by `seed`."""
    import base64, io
    import pandas as pd
    from PIL import Image
    L = ["A", "B", "C", "D"]
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["answer"].isin(L)].reset_index(drop=True)
    eval_idx = set(int(x) for x in df.sample(n=min(n_eval, len(df)), random_state=seed)["index"])
    tr = df[~df["index"].isin(eval_idx)].sample(frac=1.0, random_state=seed + 1)
    ev = df[df["index"].isin(eval_idx)]

    def rows(frame, lim):
        out = []
        for _, r in frame.iterrows():
            if lim is not None and len(out) >= lim:
                break
            try:
                img = Image.open(io.BytesIO(base64.b64decode(str(r["image"])))).convert("RGB")
            except Exception:
                continue
            out.append({"images": [img],
                        "prompt": str(r["question"]) + "\nAnswer with the letter only.",
                        "gold": str(r["answer"])})
        return out
    return Task("mmstar", L, rows(tr, n_train), rows(ev, n_eval))


def nextqa_task(csv: str, video_dir: str, n_frames: int = 4, res: int = 384,
                n_eval: int = 200, seed: int = 0, n_train: int | None = None) -> Task:
    """NExT-QA holistic video MCQ (5-choice). Frames sampled by linspace."""
    import csv as csvmod
    import os
    import random
    import numpy as np
    from PIL import Image
    from decord import VideoReader, cpu
    L = ["A", "B", "C", "D", "E"]

    def sample(path):
        vr = VideoReader(path, ctx=cpu(0), num_threads=4)
        idx = np.linspace(0, len(vr) - 1, n_frames).astype(int)
        return [Image.fromarray(f).convert("RGB").resize((res, res))
                for f in vr.get_batch(idx).asnumpy()]

    def mc(q, opts):
        return ("Question: " + q + "\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts))
                + "\nAnswer with the letter only.")

    allq = []
    with open(csv) as f:
        for r in csvmod.DictReader(f):
            p = f"{video_dir}/{r['video']}.mp4"
            if not os.path.exists(p):
                continue
            allq.append({"path": p, "q": r["question"],
                         "opts": [r[f"a{i}"] for i in range(5)], "gold": L[int(r["answer"])]})
    random.seed(seed); random.shuffle(allq)
    ev_raw, tr_raw = allq[:n_eval], allq[n_eval:]
    if n_train is not None:
        tr_raw = tr_raw[:n_train]

    def materialize(raw):
        out = []
        for it in raw:
            try:
                frames = sample(it["path"])
            except Exception:
                continue
            out.append({"images": frames, "prompt": mc(it["q"], it["opts"]), "gold": it["gold"]})
        return out
    # video decode is the bottleneck — materialize lazily by storing the sampler
    t = Task("nextqa", L, [], [])
    t._raw_train, t._raw_eval, t._sample, t._mc = tr_raw, ev_raw, sample, mc  # type: ignore
    return t


def aokvqa_task(parquet_glob: str, n_eval: int = 300, seed: int = 0,
                n_train: int | None = 1100) -> Task:
    """A-OKVQA image MCQ (4-choice) — the study's "other task" for multi-task training and
    forgetting probes (d6_adapteronly.py). `parquet_glob` matches the HF dataset parquets."""
    import glob, io
    import pandas as pd
    import torch
    from PIL import Image
    L = ["A", "B", "C", "D"]
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"no A-OKVQA parquet at {parquet_glob}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    g = torch.Generator().manual_seed(seed + 2)
    order = torch.randperm(len(df), generator=g).tolist()
    ev_idx, tr_idx = order[:n_eval], order[n_eval:]
    if n_train is not None:
        tr_idx = tr_idx[:n_train]

    def prompt(r):
        o = list(r["choices"])
        return (str(r["question"]) + "\n" + ", ".join(f"{L[i]}: {x}" for i, x in enumerate(o))
                + "\nAnswer with the letter only.")

    def rows(idxs):
        out = []
        for i in idxs:
            r = df.iloc[i]
            if len(list(r["choices"])) != 4:
                continue
            try:
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            except Exception:
                continue
            out.append({"images": [img], "prompt": prompt(r), "gold": L[int(r["correct_choice_idx"])]})
        return out
    return Task("aokvqa", L, rows(tr_idx), rows(ev_idx))


# --------------------------- the trainer ------------------------------------
class AdapterTrainer:
    """Fits one adapter on a HubShare (LLaVA-OV vtpatch) for a Task, then reports recovery.

        share = HubShare("llavaov")
        task  = mmstar_task(MMSTAR_TSV, n_eval=cfg.n_eval, seed=cfg.seed)
        tr    = AdapterTrainer(share, ShareTrainConfig(mode="mlp_e2e", recon_lambda=8))
        res   = tr.run(task, multitask=other_task)   # {adapter, native, recovery, ...}
    """

    def __init__(self, share, cfg: ShareTrainConfig):
        if share.mode != "vtpatch":
            raise NotImplementedError(
                f"AdapterTrainer supports the vtpatch backbone (llavaov); got "
                f"'{share.backbone}' (mode={share.mode}). Cross-encoder embed-splice training "
                "is the ridge GENERALIZATION study (e1_stagea.py); see sharing/FINDINGS.md.")
        self.share = share
        self.cfg = cfg
        self.torch = share.torch

    # -- collect z-score stats + (hub, native) recon pairs from train samples --
    def _collect(self, task: Task, need_pairs: bool):
        torch = self.torch
        Xs, Ys = [], []
        samples = self._train_iter(task)
        n = self.cfg.pre_samples if need_pairs else max(60, self.cfg.pre_samples // 3)
        seen = 0
        for s in samples:
            if seen >= n:
                break
            inp = self.share.build(s["images"], "x")
            pv = inp["pixel_values"]
            x = self.share.hub_on_pixels(pv).reshape(-1, 1152)   # hub is always 1152-d
            if need_pairs:
                y = self.share.native_vt_on_pixels(pv).reshape(-1, self.share.vt_dim)
                if x.shape[0] != y.shape[0]:
                    continue
            k = self.cfg.recon_subsample
            if k:
                sel = torch.randperm(x.shape[0])[:k]
                x = x[sel]
                if need_pairs:
                    y = y[sel]
            Xs.append(x.cpu())
            if need_pairs:
                Ys.append(y.cpu())
            seen += 1
        X = torch.cat(Xs)
        Y = torch.cat(Ys) if need_pairs else None
        print(f"[collect] {seen} samples -> X {tuple(X.shape)}"
              + (f", Y {tuple(Y.shape)}" if need_pairs else ""), flush=True)
        return X, Y

    # -- build + fit the adapter per mode --------------------------------------
    def _build_adapter(self, task: Task):
        from sharing import adapters
        torch = self.torch
        dev = self.share.device
        cfg = self.cfg
        need_pairs = cfg.mode in ("ridge", "mlp_recon", "mlp_e2e")
        X, Y = self._collect(task, need_pairs)
        mean, std = adapters.zscore_stats(X)

        if cfg.mode == "raw":
            ad = adapters.RidgeAffine.identity(1152, mean, std)
        elif cfg.mode == "ridge":
            ad = adapters.RidgeAffine.fit(X, Y, lam=cfg.ridge_lam)
        else:  # mlp_recon / mlp_e2e
            ad = adapters.ZScoreMLP(1152, cfg.hidden, self.share.vt_dim)
            ad.set_stats(mean, std)
            ad = ad.to(dev).float()
            self._pretrain_recon(ad, X, Y)
            return ad.to(dev).float()
        return ad.to(dev).float()

    def _pretrain_recon(self, ad, X, Y):
        torch = self.torch
        dev = self.share.device
        cfg = self.cfg
        Zc = ((X - ad.mean.cpu()) / ad.std.cpu()).float()
        Yc = Y.float()
        opt = torch.optim.Adam(ad.parameters(), lr=cfg.pre_lr)
        for st in range(cfg.pre_steps):
            idx = torch.randperm(Zc.shape[0])[:8192]
            zb = Zc[idx].to(dev); yb = Yc[idx].to(dev)
            opt.zero_grad()
            loss = ((ad.net(zb) - yb) ** 2).mean()
            loss.backward(); opt.step()
            if (st + 1) % 500 == 0:
                print(f"  [pretrain] step {st+1}/{cfg.pre_steps} mse={float(loss):.4f}", flush=True)
        with torch.no_grad():
            sr = st_ = 0.0
            ym = Yc.mean(0, keepdim=True)
            for i in range(0, len(Zc), 4096):
                zb = Zc[i:i+4096].to(dev); yb = Yc[i:i+4096].to(dev)
                sr += float(((yb - ad.net(zb)) ** 2).sum())
                st_ += float(((yb - ym.to(dev)) ** 2).sum())
        print(f"  [pretrain] done, train recon R2={1 - sr/(st_+1e-9):.3f}", flush=True)

    # -- iterate train samples (materialize video lazily) ----------------------
    def _train_iter(self, task: Task):
        if hasattr(task, "_raw_train"):       # nextqa: decode on demand
            for it in task._raw_train:
                try:
                    frames = task._sample(it["path"])  # type: ignore
                except Exception:
                    continue
                yield {"images": frames, "prompt": task._mc(it["q"], it["opts"]),  # type: ignore
                       "gold": it["gold"]}
        else:
            for s in task.train:
                yield s

    def _eval_iter(self, task: Task):
        if hasattr(task, "_raw_eval"):
            for it in task._raw_eval[:self.cfg.n_eval]:
                try:
                    frames = task._sample(it["path"])  # type: ignore
                except Exception:
                    continue
                yield {"images": frames, "prompt": task._mc(it["q"], it["opts"]),  # type: ignore
                       "gold": it["gold"]}
        else:
            for s in task.eval:
                yield s

    # -- VQA-CE fine-tune (mlp_e2e / trainable affine) -------------------------
    def _finetune(self, ad, task: Task, multitask: "Task | None"):
        torch = self.torch
        import torch.nn.functional as F
        dev = self.share.device
        cfg = self.cfg
        m = self.share.model
        m.config.use_cache = False
        try:
            m.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
        m.train()                              # GC needs train mode; reader params still frozen
        self.share.inject(ad)
        opt = torch.optim.AdamW(ad.parameters(), lr=cfg.ft_lr)
        total_opt = max(1, cfg.ft_steps // cfg.accum)

        def lr_at(o):
            if cfg.sched != "cosine":
                return cfg.ft_lr
            if o < cfg.warmup:
                return cfg.ft_lr * (o + 1) / cfg.warmup
            p = min(1.0, (o - cfg.warmup) / max(1, total_opt - cfg.warmup))
            return cfg.ft_lr * 0.5 * (1 + math.cos(math.pi * p))

        def stream():
            a = list(self._train_iter(task))
            b = list(self._train_iter(multitask)) if multitask else []
            rows = a + b
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
            for j in order:
                yield rows[j]

        done = 0; ostep = 0; run = 0.0; opt.zero_grad()
        while done < cfg.ft_steps:
            progressed = False
            for s in stream():
                if done >= cfg.ft_steps:
                    break
                progressed = True
                inp = self.share.build(s["images"], s["prompt"])
                pv = inp["pixel_values"]
                try:
                    lg = m(**inp, use_cache=False).logits[0, -1]
                    tgt = self.share.target_token(s["gold"])
                    ce = F.cross_entropy(lg.float().unsqueeze(0), torch.tensor([tgt], device=dev))
                    loss = ce
                    if cfg.recon_lambda > 0:
                        yv = self.share.native_vt_on_pixels(pv)
                        recon = ((ad(self.share.hub_on_pixels(pv)) - yv) ** 2).mean()
                        loss = ce + cfg.recon_lambda * recon
                    (loss / cfg.accum).backward()
                    run += float(loss); done += 1
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(); torch.cuda.empty_cache(); continue
                if done % cfg.accum == 0:
                    torch.nn.utils.clip_grad_norm_(ad.parameters(), cfg.grad_clip)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_at(ostep)
                    opt.step(); opt.zero_grad(); ostep += 1
                if done % 100 == 0:
                    print(f"  [ft] step {done}/{cfg.ft_steps} loss={run/100:.3f}", flush=True)
                    run = 0.0
            if not progressed:
                break
        m.config.use_cache = True
        m.eval()

    # -- accuracy (adapter vs native) ------------------------------------------
    def _accuracy(self, ad, task: Task):
        self.share.model.eval()
        self.share.inject(ad)
        with self.torch.no_grad():
            acc_a, n = self._acc_pass(task)
        with self.share.native():
            with self.torch.no_grad():
                acc_n, _ = self._acc_pass(task)
        return acc_a, acc_n, n

    def _acc_pass(self, task: Task):
        cor = tot = 0
        for s in self._eval_iter(task):
            inp = self.share.build(s["images"], s["prompt"])
            pred = self.share.forced_choice(inp, task.letters)
            tot += 1; cor += int(pred == s["gold"])
        return (100.0 * cor / tot if tot else 0.0), tot

    # -- orchestrate -----------------------------------------------------------
    def run(self, task: Task, multitask: "Task | None" = None) -> dict:
        cfg = self.cfg
        self.torch.manual_seed(cfg.seed)
        print(f"[train] backbone={self.share.backbone} mode={cfg.mode} task={task.name} "
              f"seed={cfg.seed} (reader FROZEN)", flush=True)
        ad = self._build_adapter(task)
        if cfg.mode == "mlp_e2e" and cfg.ft_steps > 0:
            self._finetune(ad, task, multitask)
        acc_a, acc_n, n = self._accuracy(ad, task)
        rec = acc_a / acc_n if acc_n else float("nan")
        print(f"\n=== {cfg.mode} / {task.name} (n={n}) ===")
        print(f"  adapter = {acc_a:.1f}%   native = {acc_n:.1f}%   recovery = {rec:.3f}", flush=True)
        out = {"mode": cfg.mode, "backbone": self.share.backbone, "task": task.name,
               "adapter_acc": acc_a, "native_acc": acc_n, "recovery": rec, "n": n,
               "seed": cfg.seed, "recon_lambda": cfg.recon_lambda, "ft_steps": cfg.ft_steps,
               "multitask": multitask.name if multitask else None}
        self._adapter = ad
        return out

    def forgetting(self, other: Task) -> dict:
        """Eval the LAST-trained adapter on a held-out OTHER task vs native (forgetting)."""
        acc_a, acc_n, n = self._accuracy(self._adapter, other)
        return {"task": other.name, "adapter_acc": acc_a, "native_acc": acc_n,
                "forgetting": acc_n - acc_a, "n": n}
