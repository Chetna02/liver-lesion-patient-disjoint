"""Five-fold cross-validation of the best single-split configuration.
    
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

import run_all_experiments as R
import run_25d_9ch as N9

OUT = R.OUT_DIR
CVB_DIR = OUT / "cv_best"
FLAGS = CVB_DIR / "flags"
LOCK = CVB_DIR / "cvbest.lock"
META_CSV = OUT / "metadata_slices.csv"
FOLDS_CSV = OUT / "volume_folds.csv"
EPOCHS = 30
SEED = 42
BACKBONES = ["resnet18", "vit", "swin"]
REPS = {"2d": "2d", "2.5d": "2.5d", "2.5d9ch": "2.5d_9ch"}  


def acquire_lock():
    CVB_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
        except Exception:
            old = None
        if old and R._pid_alive(old):
            print(f"another cv_best instance alive (pid {old}) -> exit", flush=True)
            sys.exit(0)
    LOCK.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: LOCK.exists() and LOCK.unlink())


def pick_best(tol=0.005):
    
    cands = []
    for pref, rep in REPS.items():
        for bb in BACKBONES:
            tag = f"{pref}_{bb}"
            jf = OUT / f"{tag}_metrics.json"
            if not jf.exists():
                continue
            m = json.load(open(jf))
            cands.append({"tag": tag, "rep": rep, "bb": bb,
                          "auc": m["slice"]["ROC-AUC"],
                          "patient_auc": m["patient"]["ROC-AUC"],
                          "npz": OUT / f"{tag}_test_predictions.npz", "json": jf})
    if not cands:
        return None
    top = max(c["auc"] for c in cands)
    near = [c for c in cands if top - c["auc"] <= tol]
    best = max(near, key=lambda c: (round(c["patient_auc"], 4), c["rep"] != "2.5d_9ch"))
    print(f"pick_best: top slice-AUC {top:.4f}; {len(near)} within {tol} -> chose "
          f"{best['tag']} (slice {best['auc']:.4f}, patient {best['patient_auc']:.4f})", flush=True)
    return best


def make_ds(rep, df, train):
    if rep == "2d":
        return R.Tumor2D(df, train)
    if rep == "2.5d":
        return R.Tumor25D(df, train)
    if rep == "2.5d_9ch":
        return N9.Tumor25D9(df, train)
    raise ValueError(rep)


def make_model(rep, bb):
    return N9.build9(bb) if rep == "2.5d_9ch" else R.build_model(bb)


def fold_split(meta, folds_df, k):
    test_vols = set(folds_df[folds_df.fold == k].volume)
    trv_vols = set(folds_df[folds_df.fold != k].volume)
    test_df = meta[meta.volume.isin(test_vols)].reset_index(drop=True)
    trv = meta[meta.volume.isin(trv_vols)].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(inner.split(trv, trv.tumor_present, groups=trv.volume))
    train_df, val_df = trv.iloc[tr_idx].reset_index(drop=True), trv.iloc[val_idx].reset_index(drop=True)
    s = [set(train_df.volume), set(val_df.volume), set(test_df.volume)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2]), "CV leak!"
    return train_df, val_df, test_df


def loaders(rep, tr, va, te):
    labels = tr.tumor_present.values.astype(int)
    cw = 1.0 / np.bincount(labels)
    sampler = WeightedRandomSampler(cw[labels], num_samples=len(labels), replacement=True)
    tl = DataLoader(make_ds(rep, tr, True), batch_size=R.BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    vl = DataLoader(make_ds(rep, va, False), batch_size=R.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    tel = DataLoader(make_ds(rep, te, False), batch_size=R.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    return tl, vl, tel


def train_fold(k, best, meta, folds_df):
    rep, bb = best["rep"], best["bb"]
    flag = FLAGS / f"fold{k}.done"
    if flag.exists():
        print(f"[cvbest fold{k}] flag present -> skip", flush=True)
        return

    if k == 0 and best["json"].exists() and best["npz"].exists():
        m = json.load(open(best["json"]))
        d = np.load(best["npz"])
        json.dump({"fold": 0, "slice": m["slice"], "patient": m["patient"]},
                  open(CVB_DIR / "fold0_metrics.json", "w"), indent=2)
        np.savez(CVB_DIR / "fold0_predictions.npz", y_true=d["y_true"], y_prob=d["y_prob"],
                 pat_true=d["pat_true"], pat_score=d["pat_score"])
        flag.write_text(f"reused {best['tag']}")
        print(f"[cvbest fold0] REUSED {best['tag']}", flush=True)
        return

    print(f"\n{'='*56}\n[cvbest fold{k}] {best['tag']} START\n{'='*56}", flush=True)
    tr, va, te = fold_split(meta, folds_df, k)
    tl, vl, tel = loaders(rep, tr, va, te)
    print(f"[cvbest fold{k}] vols tr/va/te={tr.volume.nunique()}/{va.volume.nunique()}/{te.volume.nunique()}", flush=True)

    model = make_model(rep, bb)
    pos = float((tr.tumor_present == 1).sum()); neg = float((tr.tumor_present == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=R.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt = CVB_DIR / f"fold{k}_checkpoint.pth"; bestp = CVB_DIR / f"fold{k}_best.pth"
    start, best_f1, noimp, freeze_ep, patience = 0, 0.0, 0, 3, 6
    R.set_freeze(model, True)
    if ckpt.exists():
        c = torch.load(ckpt, map_location=R.DEVICE, weights_only=False)
        model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optim"])
        scheduler.load_state_dict(c["sched"]); start = c["epoch"] + 1
        best_f1 = c["best_val_f1"]; noimp = c.get("no_improve", 0)
        if start >= freeze_ep:
            R.set_freeze(model, False)
        print(f"[cvbest fold{k}] resumed ep{start} best_val_f1={best_f1:.4f}", flush=True)

    from tqdm import tqdm
    for epoch in range(start, EPOCHS):
        if epoch == freeze_ep:
            R.set_freeze(model, False)
        model.train(); run = 0.0
        for imgs, lb in tqdm(tl, desc=f"[cvbest f{k}] ep{epoch+1}/{EPOCHS}", leave=False):
            imgs = imgs.to(R.DEVICE); lb = lb.to(R.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lb)
            loss.backward(); optimizer.step(); run += loss.item() * imgs.size(0)
        scheduler.step()
        yv, pv = R.eval_probs(model, vl)
        vf1 = f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)
        print(f"[cvbest fold{k}] ep{epoch+1} loss={run/len(tl.dataset):.4f} val_f1={vf1:.4f}", flush=True)
        if vf1 > best_f1 + 1e-6:
            best_f1 = vf1; noimp = 0; torch.save(model.state_dict(), bestp)
        else:
            noimp += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_val_f1": best_f1, "no_improve": noimp}, ckpt)
        if noimp >= patience:
            print(f"[cvbest fold{k}] early stop ep{epoch+1}", flush=True); break

    model.load_state_dict(torch.load(bestp, map_location=R.DEVICE, weights_only=False))
    yv, pv = R.eval_probs(model, vl)
    prec, rec, thrs = precision_recall_curve(yv, pv)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    thr = float(thrs[int(np.nanargmax(f1s))]) if len(thrs) else 0.5
    yt, pt = R.eval_probs(model, tel)
    sm = R.slice_metrics(yt, pt, thr)
    pm, pat = R.patient_metrics(te, pt, thr)
    print(f"[cvbest fold{k}] SLICE AUC={sm['ROC-AUC']:.4f} F1={sm['F1']:.4f} | "
          f"PATIENT AUC={pm['ROC-AUC']:.4f} F1={pm['F1']:.4f}", flush=True)
    np.savez(CVB_DIR / f"fold{k}_predictions.npz", y_true=yt.astype(np.int64), y_prob=pt.astype(np.float32),
             pat_true=pat.y_true.values.astype(np.int64), pat_score=pat.score.values.astype(np.float32))
    json.dump({"fold": k, "slice": sm, "patient": pm}, open(CVB_DIR / f"fold{k}_metrics.json", "w"), indent=2)
    flag.write_text("done")
    print(f"[cvbest fold{k}] DONE", flush=True)
    del model
    if R.DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def aggregate(best):
    per = []
    for k in range(5):
        f = CVB_DIR / f"fold{k}_metrics.json"
        if not f.exists():
            print(f"fold{k} missing -> cannot aggregate yet", flush=True); return
        per.append(json.load(open(f)))

    def ms(key, lvl):
        v = np.array([p[lvl][key] for p in per], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1))

    ys, ps, pts, pss = [], [], [], []
    for k in range(5):
        d = np.load(CVB_DIR / f"fold{k}_predictions.npz")
        ys.append(d["y_true"]); ps.append(d["y_prob"]); pts.append(d["pat_true"]); pss.append(d["pat_score"])
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    pts, pss = np.concatenate(pts), np.concatenate(pss)

    summary = {"best_model": best["tag"], "rep": best["rep"], "backbone": best["bb"],
               "single_fold_slice_AUC": best["auc"], "n_folds": 5, "per_fold": per,
               "slice_mean_std": {k: ms(k, "slice") for k in ["ROC-AUC", "F1", "Recall_Sensitivity", "Specificity"]},
               "patient_mean_std": {k: ms(k, "patient") for k in ["ROC-AUC", "F1", "Recall_Sensitivity", "Specificity"]},
               "pooled_slice_AUC": float(roc_auc_score(ys, ps)),
               "pooled_patient_AUC": float(roc_auc_score(pts, pss)),
               "n_patients_pooled": int(len(pts))}
    json.dump(summary, open(CVB_DIR / "cv_best_summary.json", "w"), indent=2)
    print(f"\n=== 5-FOLD CV: {best['tag']} ===", flush=True)
    for lvl in ("slice_mean_std", "patient_mean_std"):
        print(f" {lvl}:", flush=True)
        for k, (mn, sd) in summary[lvl].items():
            print(f"   {k:18s} {mn:.4f} +/- {sd:.4f}", flush=True)
    print(f" pooled slice AUC   = {summary['pooled_slice_AUC']:.4f}", flush=True)
    print(f" pooled patient AUC = {summary['pooled_patient_AUC']:.4f} (n={len(pts)})", flush=True)
    print(f"saved -> {CVB_DIR/'cv_best_summary.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    args = ap.parse_args()
    FLAGS.mkdir(parents=True, exist_ok=True)
    acquire_lock()
    R.VOL_MAXSLICE = R.build_volume_maxslice()
    best = pick_best()
    if best is None:
        print("no single-fold metrics found -> nothing to CV", flush=True); return
    print(f"BEST MODEL = {best['tag']} (slice AUC {best['auc']:.4f}) -> 5-fold CV", flush=True)
    json.dump(best, open(CVB_DIR / "chosen_model.json", "w"), indent=2, default=str)
    meta = pd.read_csv(META_CSV); folds_df = pd.read_csv(FOLDS_CSV)
    for k in ([args.only] if args.only is not None else [0, 1, 2, 3, 4]):
        train_fold(k, best, meta, folds_df)
        aggregate(best)
    print("\nCV_BEST COMPLETE" if args.only is None else "\nfold done", flush=True)


if __name__ == "__main__":
    main()
