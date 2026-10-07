"""Robustness of Swin-2.5D to Gaussian noise and reduced resolution.

Each fold's model is evaluated on its held-out patients and the results are
pooled out-of-fold. Also writes oof_slice_predictions.csv, used by
error_analysis.py.

    python robustness_eval.py
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from run_all_experiments import DEVICE  # noqa: E402
from cam_swin import load_25d, load_model, META, FOLDS, OUT_DIR  # noqa: E402

NOISE = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]     # Gaussian sigma on the [-1,1] input
RES = [224, 112, 64, 48, 32, 24, 16]            # downsample-then-upsample size
BATCH = 64


def gather_fold(fold, meta, folds, vol_max, max_per_vol):
    """Return (tensor NxCxHxW on CPU, labels N, volumes N, slice_idx N) for this
    fold's held-out test slices. Loaded once, reused across all perturbation levels."""
    fo = dict(zip(folds.volume, folds.fold))
    vols = [v for v in folds.volume if fo[v] == fold]
    rng = np.random.default_rng(42 + fold)
    xs, ys, vv, zz = [], [], [], []
    for v in vols:
        sub = meta[meta.volume == v]
        if max_per_vol and len(sub) > max_per_vol:
            sub = sub.iloc[rng.choice(len(sub), size=max_per_vol, replace=False)]
        for _, r in sub.iterrows():
            xs.append(load_25d(v, int(r.slice_idx), vol_max)[0])
            ys.append(int(r.tumor_present)); vv.append(v); zz.append(int(r.slice_idx))
    return torch.stack(xs), np.array(ys), np.array(vv), np.array(zz)


def perturb(x, kind, level):
    if kind == "noise":
        if level == 0:
            return x
        g = torch.randn(x.shape, generator=torch.Generator(device=x.device).manual_seed(0),
                        device=x.device)
        return x + level * g
    # resolution: down to `level` then back to 224
    if level >= 224:
        return x
    d = F.interpolate(x, size=(level, level), mode="bilinear", align_corners=False)
    return F.interpolate(d, size=(224, 224), mode="bilinear", align_corners=False)


@torch.no_grad()
def infer(model, X, kind, level):
    probs = []
    for i in range(0, len(X), BATCH):
        xb = X[i:i + BATCH].to(DEVICE)
        xb = perturb(xb, kind, level)
        probs.append(torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy())
    return np.concatenate(probs)


def pooled_auc(labels, probs, vols):
    slice_auc = roc_auc_score(labels, probs)
    df = pd.DataFrame({"v": vols, "y": labels, "p": probs})
    pat = df.groupby("v").agg(y=("y", "max"),
                              s=("p", lambda s: np.sort(s.values)[-3:].mean()))
    return float(slice_auc), float(roc_auc_score(pat.y, pat.s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_vol", type=int, default=0,
                    help="cap slices sampled per test volume (0=ALL -> exact headline baseline)")
    args = ap.parse_args()

    meta = pd.read_csv(META); folds = pd.read_csv(FOLDS)
    vol_max = meta.groupby("volume")["slice_idx"].max().to_dict()

    # fold-streaming: cache ONE fold's tensors at a time (caching all 58k at once
    # would need ~35 GB), run every perturbation level on it, keep only the (tiny)
    # probability arrays, then free the tensors before the next fold.
    levels = {"noise": NOISE, "res": RES}
    preds = {("noise", lv): [] for lv in NOISE}
    preds.update({("res", lv): [] for lv in RES})
    all_y, all_v, all_z, clean_prob = [], [], [], []

    for fold in range(5):
        X, y, v, z = gather_fold(fold, meta, folds, vol_max, args.max_per_vol)
        model = load_model(fold)
        print(f"  fold {fold}: {len(X)} slices, {int(y.sum())} tumor+ -> inferring all levels",
              flush=True)
        for kind, lvs in levels.items():
            for lv in lvs:
                preds[(kind, lv)].append(infer(model, X, kind, lv))
        all_y.append(y); all_v.append(v); all_z.append(z)
        clean_prob.append(preds[("noise", 0.0)][-1])   # this fold's unperturbed probs
        del X, model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    y = np.concatenate(all_y); v = np.concatenate(all_v); z = np.concatenate(all_z)

    def build_curve(kind, lvs):
        out = []
        for lv in lvs:
            p = np.concatenate(preds[(kind, lv)])
            sa, pa = pooled_auc(y, p, v)
            out.append({"level": lv, "slice_auc": sa, "patient_auc": pa})
            print(f"  {kind} {lv}: slice AUC={sa:.4f}  patient AUC={pa:.4f}", flush=True)
        return out

    print("\n== NOISE ==", flush=True)
    noise_curve = build_curve("noise", NOISE)
    print("\n== RESOLUTION ==", flush=True)
    res_curve = build_curve("res", RES)

    os.makedirs(OUT_DIR, exist_ok=True)
    # dump clean OOF per-slice predictions for the error analysis
    cp = np.concatenate(clean_prob)
    pd.DataFrame({"volume": v, "slice_idx": z, "label": y, "prob": cp}).to_csv(
        os.path.join(OUT_DIR, "oof_slice_predictions.csv"), index=False)
    json.dump({"noise": noise_curve, "resolution": res_curve,
               "max_per_vol": args.max_per_vol},
              open(os.path.join(OUT_DIR, "robustness.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    nx = [c["level"] for c in noise_curve]
    ax[0].plot(nx, [c["slice_auc"] for c in noise_curve], "o-", label="slice")
    ax[0].plot(nx, [c["patient_auc"] for c in noise_curve], "s--", label="patient")
    ax[0].set_xlabel("Gaussian noise $\\sigma$"); ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("Robustness to noise"); ax[0].legend(); ax[0].set_ylim(0.5, 1.0)
    rx = [c["level"] for c in res_curve]
    ax[1].plot(rx, [c["slice_auc"] for c in res_curve], "o-", label="slice")
    ax[1].plot(rx, [c["patient_auc"] for c in res_curve], "s--", label="patient")
    ax[1].set_xlabel("effective resolution (px, upsampled to 224)")
    ax[1].set_title("Robustness to resolution"); ax[1].invert_xaxis()
    ax[1].legend(); ax[1].set_ylim(0.5, 1.0)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "robustness.png")
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"\nsaved -> {out}", flush=True)
    print("ROBUSTNESS DONE", flush=True)


if __name__ == "__main__":
    main()
