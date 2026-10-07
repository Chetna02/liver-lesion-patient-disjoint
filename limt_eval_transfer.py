"""Zero-shot evaluation of the LiTS-trained models on LiMT.

Uses the model builder, transforms and metrics from run_all_experiments.py.
Reports slice- and patient-level metrics with patient-cluster bootstrap CIs,
at the threshold transferred from LiTS. --cv evaluates the five
cross-validated Swin-2.5D models and their ensemble.

    
"""
import pandas as pd  # before torch/sklearn on purpose (import-order segfault guard)

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score

from run_all_experiments import (  # noqa: E402
    _base_tf, build_model, eval_probs, slice_metrics, patient_metrics,
    BACKBONES, MODES, BATCH, DEVICE,
)
import config

BTC_DIR = config.WORK_DIR
LIMT_IMG_DIR = BTC_DIR / "limt_images"
LIMT_META_CSV = BTC_DIR / "limt_metadata_slices.csv"
OUT_DIR = BTC_DIR / "limt_transfer"


# ----------------------------------------------------------------------
# LiMT datasets -- same encoding/transform as the LiTS ones, but the neighbour
# lookup is parameterised by image dir instead of reading run_all_experiments'
# module-level IMG_DIR / VOL_MAXSLICE globals (which point at the LiTS set).
# ----------------------------------------------------------------------
class LimtTumor2D(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tf = _base_tf(False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        return self.tf(img), torch.tensor(float(r["tumor_present"]))


class LimtTumor25D(Dataset):
    """3 adjacent slices (z-1, z, z+1), channel 0 (liver window) of each, stacked --
    identical construction to run_all_experiments.Tumor25D."""

    def __init__(self, df, maxslice):
        self.df = df.reset_index(drop=True)
        self.maxslice = maxslice
        self.tf = _base_tf(False)

    def __len__(self):
        return len(self.df)

    def _liverchan(self, volume, idx):
        idx = int(min(max(idx, 0), self.maxslice[volume]))
        p = LIMT_IMG_DIR / f"{volume}_slice_{idx:03d}.jpg"
        return np.array(Image.open(p).convert("RGB"))[:, :, 0]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v, z = r["volume"], int(r["slice_idx"])
        stack = np.stack([self._liverchan(v, z - 1),
                          self._liverchan(v, z),
                          self._liverchan(v, z + 1)], axis=-1)
        return self.tf(Image.fromarray(stack)), torch.tensor(float(r["tumor_present"]))


def cluster_bootstrap_auc(groups, y_true, y_score, n_boot=1000, seed=42):
    """Percentile CI for AUC, resampling whole patients. Works for slice-level
    arrays (many rows per patient) and patient-level arrays (one row each)."""
    groups = np.asarray(groups)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    uniq = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in drawn])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) > 1:
            vals.append(roc_auc_score(yt, ys))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _make_loader(ds, num_workers):
    """DataLoader with optional parallel JPG decoding. The single-threaded default
    inherited from run_all_experiments.py leaves the GPU ~90% idle here: the 2.5D
    datasets read THREE JPGs per sample, so decoding, not compute, is the bottleneck
    on a 9,366-slice pass."""
    kw = dict(batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


def evaluate_checkpoint(mode, backbone, df, maxslice, n_boot, num_workers=0):
    tag = f"{mode}_{backbone}"
    best = BTC_DIR / f"{tag}_best.pth"
    mjson = BTC_DIR / f"{tag}_metrics.json"
    if not best.exists():
        print(f"[{tag}] no checkpoint -> skip", flush=True)
        return None
    thr_lits = json.load(open(mjson))["best_threshold"] if mjson.exists() else 0.5

    model = build_model(backbone)
    model.load_state_dict(torch.load(best, map_location=DEVICE, weights_only=False))

    ds = LimtTumor2D(df) if mode == "2d" else LimtTumor25D(df, maxslice)
    loader = _make_loader(ds, num_workers)
    y, p = eval_probs(model, loader)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # refit threshold on LiMT (upper bound, not a deployment number)
    prec, rec, thrs = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    thr_refit = float(thrs[int(np.nanargmax(f1s))]) if len(thrs) else 0.5

    res = {"mode": mode, "backbone": backbone,
           "threshold_transferred": float(thr_lits), "threshold_refit": thr_refit,
           "n_slices": int(len(y)), "n_slice_pos": int(y.sum())}

    for label, thr in (("transferred", thr_lits), ("refit", thr_refit)):
        sm = slice_metrics(y, p, thr)
        pm, pat = patient_metrics(df, p, thr)
        res[f"slice_{label}"] = sm
        res[f"patient_{label}"] = pm
        if label == "transferred":
            res["slice_auc_ci"] = cluster_bootstrap_auc(df["volume"].values, y, p, n_boot)
            res["patient_auc_ci"] = cluster_bootstrap_auc(
                pat["volume"].values, pat["y_true"].values, pat["score"].values, n_boot)
            res["_pat"] = pat

    st, pt = res["slice_transferred"], res["patient_transferred"]
    sr, pr = res["slice_refit"], res["patient_refit"]
    ci_s, ci_p = res["slice_auc_ci"], res["patient_auc_ci"]
    print(f"[{tag}] slice   AUC={st['ROC-AUC']:.4f} [{ci_s[0]:.3f}, {ci_s[1]:.3f}]  "
          f"F1={st['F1']:.4f} sens={st['Recall_Sensitivity']:.3f} spec={st['Specificity']:.3f}",
          flush=True)
    print(f"[{tag}] patient AUC={pt['ROC-AUC']:.4f} [{ci_p[0]:.3f}, {ci_p[1]:.3f}]  "
          f"F1={pt['F1']:.4f} sens={pt['Recall_Sensitivity']:.3f} spec={pt['Specificity']:.3f} "
          f"(n={pt['n_patients']})", flush=True)
    print(f"[{tag}] refit thr={thr_refit:.3f}: slice F1={sr['F1']:.4f} | "
          f"patient sens={pr['Recall_Sensitivity']:.3f} spec={pr['Specificity']:.3f}", flush=True)
    return res


def evaluate_cv(df, maxslice, n_boot, num_workers=0):
    
    cv_dir = BTC_DIR / "cv_best"
    ckpts = [BTC_DIR / "2.5d_swin_best.pth"] + [cv_dir / f"fold{k}_best.pth" for k in range(1, 5)]
    missing = [str(c) for c in ckpts if not c.exists()]
    if missing:
        print(f"CV arm: missing {len(missing)} checkpoint(s) -> skip\n  " +
              "\n  ".join(missing), flush=True)
        return None

    ds = LimtTumor25D(df, maxslice)
    loader = _make_loader(ds, num_workers)

    per_fold, pat_scores = [], []
    for k, cp in enumerate(ckpts):
        model = build_model("swin")
        model.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False))
        y, p = eval_probs(model, loader)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        prec, rec, thrs = precision_recall_curve(y, p)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        thr = float(thrs[int(np.nanargmax(f1s))]) if len(thrs) else 0.5
        sm = slice_metrics(y, p, thr)
        pm, pat = patient_metrics(df, p, thr)
        pat_scores.append(pat.set_index("volume")["score"])
        per_fold.append({"fold": k, "checkpoint": str(cp), "threshold_refit": thr,
                         "slice": sm, "patient": pm})
        print(f"  fold{k}: slice AUC={sm['ROC-AUC']:.4f}  patient AUC={pm['ROC-AUC']:.4f} "
              f"spec={pm['Specificity']:.3f}", flush=True)

    sl = [f["slice"]["ROC-AUC"] for f in per_fold]
    pa = [f["patient"]["ROC-AUC"] for f in per_fold]

    ens = pd.concat(pat_scores, axis=1).mean(axis=1).rename("score").reset_index()
    truth = df.groupby("volume")["tumor_present"].max().rename("y_true").reset_index()
    ens = ens.merge(truth, on="volume")
    ens_auc = float(roc_auc_score(ens.y_true, ens.score))
    ens_ci = cluster_bootstrap_auc(ens.volume.values, ens.y_true.values, ens.score.values, n_boot)

    print(f"\n  CV slice   AUC {np.mean(sl):.4f} +/- {np.std(sl, ddof=1):.4f}", flush=True)
    print(f"  CV patient AUC {np.mean(pa):.4f} +/- {np.std(pa, ddof=1):.4f}", flush=True)
    print(f"  ensemble patient AUC {ens_auc:.4f} [{ens_ci[0]:.3f}, {ens_ci[1]:.3f}]", flush=True)
    print(f"  (LiTS reference: slice 0.955 +/- 0.024, patient 0.866 +/- 0.070)", flush=True)

    return {"per_fold": per_fold,
            "slice_auc_mean": float(np.mean(sl)), "slice_auc_sd": float(np.std(sl, ddof=1)),
            "patient_auc_mean": float(np.mean(pa)), "patient_auc_sd": float(np.std(pa, ddof=1)),
            "ensemble_patient_auc": ens_auc, "ensemble_patient_auc_ci": ens_ci,
            "lits_reference": {"slice_auc": [0.955, 0.024], "patient_auc": [0.866, 0.070]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated mode:backbone (default: all available)")
    ap.add_argument("--cv", action="store_true",
                    help="also transfer the 5 cross-validated Swin-2.5D models "
                         "(the manuscript's headline result)")
    ap.add_argument("--liver-only", action="store_true",
                    help="restrict to liver-bearing slices (LiTS eval used all slices)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=6,
                    help="parallel JPG decoding workers (0 = single-threaded)")
    args = ap.parse_args()

    if not LIMT_META_CSV.exists():
        raise SystemExit(f"missing {LIMT_META_CSV} -- run limt_prep_jpeg.py first")

    df = pd.read_csv(LIMT_META_CSV)
    if args.liver_only:
        df = df[df["liver_present"] == 1].reset_index(drop=True)
    maxslice = pd.read_csv(LIMT_META_CSV).groupby("volume")["slice_idx"].max().to_dict()

    vol = df.groupby("volume")["tumor_present"].max()
    print(f"LiMT: {len(df)} slices | {df.volume.nunique()} cases "
          f"({int((vol == 1).sum())} diseased / {int((vol == 0).sum())} normal) | "
          f"{int(df.tumor_present.sum())} positive slices", flush=True)
    print(f"slices: {'liver-bearing only' if args.liver_only else 'all'}\n", flush=True)

    if args.models:
        pairs = [tuple(m.split(":")) for m in args.models.split(",")]
    else:
        pairs = [(m, b) for m in MODES for b in BACKBONES]

    results = {}
    for mode, backbone in pairs:
        r = evaluate_checkpoint(mode, backbone, df, maxslice, args.n_boot, args.num_workers)
        if r is not None:
            results[f"{mode}_{backbone}"] = r

    cv_result = None
    if args.cv:
        print("\n=== 5-fold CV Swin-2.5D, zero-shot on LiMT ===", flush=True)
        cv_result = evaluate_cv(df, maxslice, args.n_boot, args.num_workers)

    if not results:
        raise SystemExit("no checkpoints evaluated")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for tag, r in results.items():
        pat = r["_pat"]
        fpr, tpr, _ = roc_curve(pat["y_true"], pat["score"])
        axes[1].plot(fpr, tpr, lw=1.3,
                     label=f"{tag} ({r['patient_transferred']['ROC-AUC']:.3f})")
    for ax, title in zip(axes, ["slice level", "patient level"]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
        ax.set_title(f"Zero-shot LiTS $\\rightarrow$ LiMT, {title}")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[0].text(0.5, 0.5, "see JSON for slice-level curves",
                 ha="center", va="center", fontsize=8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_limt_transfer.png", dpi=150)
    plt.close(fig)

    rows = []
    for tag, r in results.items():
        r.pop("_pat", None)
        rows.append({
            "model": tag,
            "slice_AUC": r["slice_transferred"]["ROC-AUC"],
            "slice_F1": r["slice_transferred"]["F1"],
            "patient_AUC": r["patient_transferred"]["ROC-AUC"],
            "patient_F1": r["patient_transferred"]["F1"],
            "patient_sens": r["patient_transferred"]["Recall_Sensitivity"],
            "patient_spec": r["patient_transferred"]["Specificity"],
            "patient_spec_refit": r["patient_refit"]["Specificity"],
        })
    table = pd.DataFrame(rows).sort_values("patient_AUC", ascending=False)
    table.to_csv(OUT_DIR / "limt_transfer_table.csv", index=False)
    print("\n=== ZERO-SHOT LiTS -> LiMT ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print("\nNOTE: specificity rests on 50 normal LiMT cases vs 13 in all of LiTS.",
          flush=True)

    json.dump({"liver_only": args.liver_only, "n_boot": args.n_boot,
               "results": results, "cv": cv_result},
              open(OUT_DIR / "limt_transfer.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {OUT_DIR}", flush=True)
    print("LIMT TRANSFER EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
