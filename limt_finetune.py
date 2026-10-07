"""Fine-tune the LiTS Swin-2.5D model on LiMT with patient-wise 5-fold CV.

Each fold compares the zero-shot and fine-tuned model on the same held-out
LiMT patients.    # one split instead of 5-fold
"""
import pandas as pd  # before torch/sklearn (import-order segfault guard)

import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix

from run_all_experiments import (  # noqa: E402
    build_model, eval_probs, slice_metrics, patient_metrics, _base_tf,
    BATCH, DEVICE,
)
from limt_eval_transfer import (  # noqa: E402
    LimtTumor25D, cluster_bootstrap_auc, LIMT_IMG_DIR, LIMT_META_CSV,
)
import config

BTC_DIR = config.WORK_DIR
INIT_CKPT = BTC_DIR / "2.5d_swin_best.pth"
OUT_DIR = BTC_DIR / "limt_finetune"
SPLIT_JSON = OUT_DIR / "limt_split.json"
FT_CKPT = OUT_DIR / "limt_swin25d_ft.pth"
SEED = 42


# -- train dataset: same 2.5D construction as eval, but with augmentation on ----
class LimtTumor25DTrain(Dataset):
    def __init__(self, df, maxslice):
        self.df = df.reset_index(drop=True)
        self.maxslice = maxslice
        self.tf = _base_tf(True)          # train=True -> flips / rotation / jitter

    def __len__(self):
        return len(self.df)

    def _liverchan(self, v, idx):
        idx = int(min(max(idx, 0), self.maxslice[v]))
        return np.array(Image.open(LIMT_IMG_DIR / f"{v}_slice_{idx:03d}.jpg").convert("RGB"))[:, :, 0]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v, z = r["volume"], int(r["slice_idx"])
        stack = np.stack([self._liverchan(v, z - 1),
                          self._liverchan(v, z),
                          self._liverchan(v, z + 1)], axis=-1)
        return self.tf(Image.fromarray(stack)), torch.tensor(float(r["tumor_present"]))


def make_split(df, seed=SEED):
    """Patient-wise train/val/test, stratified by case disease status. Persisted
    and reused so zero-shot and fine-tuned share an identical test set."""
    if SPLIT_JSON.exists():
        s = json.load(open(SPLIT_JSON))
        print(f"reusing split: train={len(s['train'])} val={len(s['val'])} "
              f"test={len(s['test'])} cases", flush=True)
        return s["train"], s["val"], s["test"]

    case_lab = df.groupby("volume")["tumor_present"].max()
    cases = case_lab.index.to_numpy()
    labels = case_lab.to_numpy()
    tr, tmp, ytr, ytmp = train_test_split(cases, labels, test_size=0.30,
                                          stratify=labels, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=ytmp, random_state=seed)
    tr, va, te = sorted(tr.tolist()), sorted(va.tolist()), sorted(te.tolist())

    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"train": tr, "val": va, "test": te, "seed": seed}, open(SPLIT_JSON, "w"), indent=2)

    def desc(name, ids):
        n_dis = int(case_lab.loc[ids].sum())
        print(f"  {name:5s}: {len(ids):3d} cases ({n_dis} diseased / {len(ids) - n_dis} normal)",
              flush=True)
    print("patient-wise LiMT split (disjoint, disease-stratified):", flush=True)
    desc("train", tr); desc("val", va); desc("test", te)
    return tr, va, te


def loaders_for(df, maxslice, train_ids, val_ids, test_ids):
    tr_df = df[df.volume.isin(train_ids)].reset_index(drop=True)
    va_df = df[df.volume.isin(val_ids)].reset_index(drop=True)
    te_df = df[df.volume.isin(test_ids)].reset_index(drop=True)

    lab = tr_df["tumor_present"].values.astype(int)
    cw = 1.0 / np.bincount(lab)
    sampler = WeightedRandomSampler(cw[lab], num_samples=len(lab), replacement=True)
    nw = 0 if os.name == "nt" else 4    # keep Windows single-proc for stability

    tr_loader = DataLoader(LimtTumor25DTrain(tr_df, maxslice), batch_size=BATCH,
                           sampler=sampler, num_workers=nw, pin_memory=True)
    va_loader = DataLoader(LimtTumor25D(va_df, maxslice), batch_size=BATCH,
                           shuffle=False, num_workers=nw, pin_memory=True)
    te_loader = DataLoader(LimtTumor25D(te_df, maxslice), batch_size=BATCH,
                           shuffle=False, num_workers=nw, pin_memory=True)
    return tr_df, va_df, te_df, tr_loader, va_loader, te_loader


def val_f1(model, loader):
    y, p = eval_probs(model, loader)
    return f1_score(y, (p >= 0.5).astype(int), zero_division=0)


def finetune(model, tr_loader, va_loader, tr_df, epochs, lr, weight_decay, patience,
             smoke=False, tag=""):
    """Fine-tune `model` in place from its current (LiTS) weights on one LiMT
    training split; keep the best-val-F1 state. Returns best val-F1."""
    pos = float((tr_df.tumor_present == 1).sum()); neg = float((tr_df.tumor_present == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ep = 1 if smoke else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

    best_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(ep):
        model.train(); running = 0.0
        for imgs, lb in tr_loader:
            imgs, lb = imgs.to(DEVICE), lb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lb)
            loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)
            if smoke:
                break
        scheduler.step()
        vf1 = val_f1(model, va_loader)
        print(f"  {tag}epoch {epoch+1}/{ep} loss={running/max(1,len(tr_loader.dataset)):.4f} "
              f"val_f1={vf1:.4f}", flush=True)
        if vf1 > best_f1 + 1e-6:
            best_f1, no_improve = vf1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  {tag}early stop at epoch {epoch+1} (best val_f1={best_f1:.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1


def case_scores(model, df_sub, loader):
    """Per-case (top-3 mean) score + true label, via patient_metrics' aggregation
    at a fixed 0.5 (threshold irrelevant for scores). Returns a {volume: (y,score)}."""
    y, p = eval_probs(model, loader)
    _, pat = patient_metrics(df_sub, p, 0.5)
    return {r.volume: (int(r.y_true), float(r.score)) for r in pat.itertuples()}


def slice_and_case_scores(model, df_sub, loader):
    """Both per-slice probabilities (aligned to df_sub row order -- LimtTumor25D
    preserves it) and per-case top-3-mean scores, from a single inference pass.
    The per-slice output feeds the per-lesion-type detection breakdown."""
    y, p = eval_probs(model, loader)
    sub = df_sub.reset_index(drop=True).copy()
    sub["prob"] = p
    _, pat = patient_metrics(sub, p, 0.5)
    case = {r.volume: (int(r.y_true), float(r.score)) for r in pat.itertuples()}
    return sub[["volume", "slice_idx", "tumor_present", "prob"]], case


def pooled_patient_metrics(case2ys, thr):
    vols = sorted(case2ys)
    yt = np.array([case2ys[v][0] for v in vols])
    ys = np.array([case2ys[v][1] for v in vols])
    yp = (ys >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"n": len(vols), "auc": float(roc_auc_score(yt, ys)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "threshold": float(thr)}, vols, yt, ys


def score_split(model, df_sub, loader, n_boot):
    """AUC/sens/spec at the F1-optimal threshold (refit on this split's own PR
    curve), plus case-level bootstrap CIs. Used identically for zero-shot init and
    fine-tuned model on the shared test set."""
    y, p = eval_probs(model, loader)
    prec, rec, thrs = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    thr = float(thrs[int(np.nanargmax(f1s))]) if len(thrs) else 0.5
    sm = slice_metrics(y, p, thr)
    pm, pat = patient_metrics(df_sub, p, thr)
    sl_ci = cluster_bootstrap_auc(df_sub["volume"].values, y, p, n_boot)
    pt_ci = cluster_bootstrap_auc(pat["volume"].values, pat["y_true"].values,
                                  pat["score"].values, n_boot)
    return {"threshold": thr, "slice": sm, "patient": pm,
            "slice_auc_ci": sl_ci, "patient_auc_ci": pt_ci,
            "_pat": pat[["volume", "y_true", "score"]].to_dict("records")}


def paired_auc_bootstrap(vols, yt, s0, s1, n_boot, seed=SEED):
    """Paired case-level bootstrap of AUC(s1) - AUC(s0) over the same cases."""
    yt, s0, s1 = np.asarray(yt), np.asarray(s0), np.asarray(s1)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vols), size=len(vols))
        if len(np.unique(yt[idx])) < 2:
            continue
        diffs.append(roc_auc_score(yt[idx], s1[idx]) - roc_auc_score(yt[idx], s0[idx]))
    diffs = np.asarray(diffs)
    d = float(roc_auc_score(yt, s1) - roc_auc_score(yt, s0))
    return d, [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], \
        float((diffs > 0).mean())


def run_kfold(df, maxslice, args):
    """5-fold pooled fine-tuning. Every case is held out for test exactly once, so
    the fine-tuned predictions pool to all 150 cases -- matching the coverage of the
    zero-shot CV arm and giving a paired zero-shot-vs-fine-tuned comparison over the
    full cohort rather than a single 23-case split."""
    case_lab = df.groupby("volume")["tumor_present"].max()
    cases = case_lab.index.to_numpy()
    labels = case_lab.to_numpy()

   
    print("=== zero-shot (LiTS init) over all 150 LiMT cases ===", flush=True)
    init_state = torch.load(INIT_CKPT, map_location=DEVICE, weights_only=False)
    model = build_model("swin"); model.load_state_dict(init_state)
    all_loader = DataLoader(LimtTumor25D(df, maxslice), batch_size=BATCH,
                            shuffle=False, num_workers=0, pin_memory=True)
    zs_slice, zs_case = slice_and_case_scores(model, df, all_loader)
    zs_slice = zs_slice.rename(columns={"prob": "zs_prob"})
    print(f"  zero-shot pooled patient AUC="
          f"{roc_auc_score([zs_case[v][0] for v in zs_case], [zs_case[v][1] for v in zs_case]):.4f}",
          flush=True)

    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "_fold_cache"
    cache_dir.mkdir(exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    ft_case, fold_val_f1, ft_slice_parts = {}, [], []
    for k, (trv_idx, te_idx) in enumerate(sgkf.split(cases, labels, groups=cases)):
        test_ids = sorted(cases[te_idx].tolist())
        trv_ids = cases[trv_idx]
        trv_lab = labels[trv_idx]
        tr_ids, va_ids = train_test_split(trv_ids, test_size=0.15, stratify=trv_lab,
                                          random_state=SEED + k)
        tr_ids, va_ids = sorted(tr_ids.tolist()), sorted(va_ids.tolist())
        n_dis_te = int(case_lab.loc[test_ids].sum())

        slice_cache = cache_dir / f"fold{k}_slice.csv"
        case_cache = cache_dir / f"fold{k}_case.json"
        if slice_cache.exists() and case_cache.exists() and not args.smoke:
            meta = json.load(open(case_cache))
            fold_val_f1.append(meta["val_f1"])
            ft_case.update({v: tuple(ys) for v, ys in meta["case"].items()})
            ft_slice_parts.append(pd.read_csv(slice_cache))
            print(f"\n----- fold {k}: cached (val_f1={meta['val_f1']:.4f}) -> skip -----",
                  flush=True)
            continue

        print(f"\n----- fold {k}: train={len(tr_ids)} val={len(va_ids)} "
              f"test={len(test_ids)} ({n_dis_te} dis / {len(test_ids)-n_dis_te} norm) -----",
              flush=True)

        tr_df, va_df, te_df, tr_loader, va_loader, te_loader = loaders_for(
            df, maxslice, tr_ids, va_ids, test_ids)

        model = build_model("swin"); model.load_state_dict(init_state)   # fresh LiTS init each fold
        vf1 = finetune(model, tr_loader, va_loader, tr_df, args.epochs, args.lr,
                       args.weight_decay, args.patience, smoke=args.smoke, tag=f"f{k} ")
        fold_val_f1.append(vf1)
        ft_slice_k, ft_case_k = slice_and_case_scores(model, te_df, te_loader)
        ft_case.update(ft_case_k)
        ft_slice_k = ft_slice_k.rename(columns={"prob": "ft_prob"})
        ft_slice_parts.append(ft_slice_k)
        if not args.smoke:
            ft_slice_k.to_csv(slice_cache, index=False)
            json.dump({"val_f1": vf1, "case": {v: list(ys) for v, ys in ft_case_k.items()}},
                      open(case_cache, "w"))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -- per-slice out-of-fold predictions (+ lesion-type flags) for the breakdown -
    ft_slice = pd.concat(ft_slice_parts, ignore_index=True)
    lesion_cols = [c for c in df.columns if c.startswith("lesion_")]
    oof = (df[["volume", "slice_idx", "tumor_present", "liver_present"] + lesion_cols]
           .merge(zs_slice[["volume", "slice_idx", "zs_prob"]], on=["volume", "slice_idx"])
           .merge(ft_slice[["volume", "slice_idx", "ft_prob"]], on=["volume", "slice_idx"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUT_DIR / "limt_oof_slice_preds.csv", index=False)
    print(f"\n  wrote per-slice OOF predictions ({len(oof)} slices) -> "
          f"{OUT_DIR / 'limt_oof_slice_preds.csv'}", flush=True)

    # pooled, paired over all cases present in both dicts
    vols = sorted(set(zs_case) & set(ft_case))
    yt = np.array([zs_case[v][0] for v in vols])
    s0 = np.array([zs_case[v][1] for v in vols])
    s1 = np.array([ft_case[v][1] for v in vols])

    # F1-optimal pooled thresholds (each computed on its own pooled scores)
    def f1opt(y, s):
        prec, rec, thr = precision_recall_curve(y, s)
        f = 2 * prec * rec / (prec + rec + 1e-8)
        return float(thr[int(np.nanargmax(f))]) if len(thr) else 0.5
    zs_pm, _, _, _ = pooled_patient_metrics({v: zs_case[v] for v in vols}, f1opt(yt, s0))
    ft_pm, _, _, _ = pooled_patient_metrics({v: ft_case[v] for v in vols}, f1opt(yt, s1))
    zs_pm["auc_ci"] = cluster_bootstrap_auc(vols, yt, s0, args.n_boot)
    ft_pm["auc_ci"] = cluster_bootstrap_auc(vols, yt, s1, args.n_boot)

    d_auc, d_ci, p_gt0 = paired_auc_bootstrap(vols, yt, s0, s1, args.n_boot)

    print(f"\n=== POOLED over {len(vols)} cases (5-fold) ===", flush=True)
    print(f"  zero-shot  patient AUC={zs_pm['auc']:.4f} "
          f"[{zs_pm['auc_ci'][0]:.3f}, {zs_pm['auc_ci'][1]:.3f}]  "
          f"sens={zs_pm['sensitivity']:.3f} spec={zs_pm['specificity']:.3f}", flush=True)
    print(f"  fine-tuned patient AUC={ft_pm['auc']:.4f} "
          f"[{ft_pm['auc_ci'][0]:.3f}, {ft_pm['auc_ci'][1]:.3f}]  "
          f"sens={ft_pm['sensitivity']:.3f} spec={ft_pm['specificity']:.3f}", flush=True)
    print(f"  delta AUC (fine-tuned - zero-shot) = {d_auc:+.4f} "
          f"[{d_ci[0]:+.3f}, {d_ci[1]:+.3f}]  P(>0)={p_gt0:.3f}", flush=True)
    print(f"  per-fold val-F1: {[round(x,3) for x in fold_val_f1]}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"mode": "kfold", "config": vars(args),
           "zero_shot": zs_pm, "fine_tuned": ft_pm,
           "delta_patient_auc": d_auc, "delta_patient_auc_ci": d_ci,
           "prob_improvement_gt0": p_gt0, "per_fold_val_f1": fold_val_f1,
           "per_case": {v: {"y": int(zs_case[v][0]),
                            "zero_shot": zs_case[v][1], "fine_tuned": ft_case[v][1]}
                        for v in vols}}
    json.dump(out, open(OUT_DIR / "limt_finetune_kfold.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {OUT_DIR / 'limt_finetune_kfold.json'}", flush=True)
    print("LIMT FINE-TUNE (KFOLD) DONE", flush=True)


def run_single(df, maxslice, args):
    """Original single patient-wise split. Faster, but the ~23-case test set gives
    wide CIs; kept for a quick check. Prefer run_kfold for the reported result."""
    train_ids, val_ids, test_ids = make_split(df)
    tr_df, va_df, te_df, tr_loader, va_loader, te_loader = loaders_for(
        df, maxslice, train_ids, val_ids, test_ids)

    print("\n=== zero-shot (LiTS init) on LiMT test split ===", flush=True)
    init_state = torch.load(INIT_CKPT, map_location=DEVICE, weights_only=False)
    model = build_model("swin"); model.load_state_dict(init_state)
    zs = score_split(model, te_df, te_loader, args.n_boot)
    print(f"  patient AUC={zs['patient']['ROC-AUC']:.4f} "
          f"[{zs['patient_auc_ci'][0]:.3f}, {zs['patient_auc_ci'][1]:.3f}]", flush=True)

    vf1 = finetune(model, tr_loader, va_loader, tr_df, args.epochs, args.lr,
                   args.weight_decay, args.patience, smoke=args.smoke)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "backbone": "swin", "rep": "2.5d",
                "init": str(INIT_CKPT), "best_val_f1": vf1}, FT_CKPT)

    print("\n=== fine-tuned on LiMT test split ===", flush=True)
    ft = score_split(model, te_df, te_loader, args.n_boot)
    print(f"  patient AUC={ft['patient']['ROC-AUC']:.4f} "
          f"[{ft['patient_auc_ci'][0]:.3f}, {ft['patient_auc_ci'][1]:.3f}]", flush=True)

    zs_pat = {d["volume"]: d for d in zs["_pat"]}
    ft_pat = {d["volume"]: d for d in ft["_pat"]}
    vols = sorted(zs_pat)
    yt = np.array([zs_pat[v]["y_true"] for v in vols])
    s0 = np.array([zs_pat[v]["score"] for v in vols])
    s1 = np.array([ft_pat[v]["score"] for v in vols])
    d_auc, d_ci, p_gt0 = paired_auc_bootstrap(vols, yt, s0, s1, args.n_boot)
    print(f"\n  delta patient AUC = {d_auc:+.4f} [{d_ci[0]:+.3f}, {d_ci[1]:+.3f}]  "
          f"P(>0)={p_gt0:.3f}", flush=True)

    for r in (zs, ft):
        r.pop("_pat", None)
    json.dump({"mode": "single", "config": vars(args),
               "split": {"train": train_ids, "val": val_ids, "test": test_ids},
               "zero_shot": zs, "fine_tuned": ft, "delta_patient_auc": d_auc,
               "delta_patient_auc_ci": d_ci, "prob_improvement_gt0": p_gt0},
              open(OUT_DIR / "limt_finetune_result.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {OUT_DIR / 'limt_finetune_result.json'}", flush=True)
    print("LIMT FINE-TUNE (SINGLE) DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--single", action="store_true",
                    help="single train/val/test split instead of the default 5-fold pooled")
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny wiring test")
    args = ap.parse_args()

    if not INIT_CKPT.exists():
        raise SystemExit(f"missing init checkpoint {INIT_CKPT}")
    if not LIMT_META_CSV.exists():
        raise SystemExit(f"missing {LIMT_META_CSV} -- run limt_prep_jpeg.py first")

    df = pd.read_csv(LIMT_META_CSV)
    maxslice = df.groupby("volume")["slice_idx"].max().to_dict()
    if args.single:
        run_single(df, maxslice, args)
    else:
        run_kfold(df, maxslice, args)


if __name__ == "__main__":
    main()
