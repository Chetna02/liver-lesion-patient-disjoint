"""Slice-level ROC-AUC of the five-fold Swin-2.5D ensemble on LiMT (zero-shot).

Averages the per-slice probabilities of the five cross-validated models.

    python limt_ensemble_eval.py
"""
import pandas as pd  # before torch/sklearn (import-order segfault guard)
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

from run_all_experiments import build_model, eval_probs, slice_metrics, patient_metrics, DEVICE
import torch
from limt_eval_transfer import (LimtTumor25D, cluster_bootstrap_auc, _make_loader,
                                BTC_DIR, LIMT_META_CSV)

df = pd.read_csv(LIMT_META_CSV)                      # all slices (matches evaluate_cv default)
maxslice = df.groupby("volume")["slice_idx"].max().to_dict()
vol = df.groupby("volume")["tumor_present"].max()
print(f"LiMT: {len(df)} slices | {df.volume.nunique()} cases "
      f"({int((vol==1).sum())} diseased / {int((vol==0).sum())} normal)", flush=True)

ckpts = [BTC_DIR / "2.5d_swin_best.pth"] + [BTC_DIR / "cv_best" / f"fold{k}_best.pth" for k in range(1, 5)]
assert all(c.exists() for c in ckpts), "missing checkpoint(s)"

ds = LimtTumor25D(df, maxslice)
loader = _make_loader(ds, num_workers=0)   # num_workers=0: 

slice_probs, pat_scores, per_fold_slice_auc = [], [], []
y_ref = None
for k, cp in enumerate(ckpts):
    model = build_model("swin")
    model.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False))
    y, p = eval_probs(model, loader)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    if y_ref is None:
        y_ref = y
    else:
        assert np.array_equal(y_ref, y), "label order changed across folds -- alignment broken"
    slice_probs.append(p)
    per_fold_slice_auc.append(roc_auc_score(y, p))
    # per-fold patient score (top-3-mean per volume), for the patient-ensemble sanity check
    _, pat = patient_metrics(df, p, 0.5)
    pat_scores.append(pat.set_index("volume")["score"])
    print(f"  fold{k}: slice AUC={per_fold_slice_auc[-1]:.4f}", flush=True)

# ---- SLICE ensemble: mean of per-slice probabilities ----
p_ens = np.mean(np.vstack(slice_probs), axis=0)
slice_ens_auc = float(roc_auc_score(y_ref, p_ens))
slice_ens_ci = cluster_bootstrap_auc(df["volume"].values, y_ref, p_ens, n_boot=1000)

# ---- PATIENT ensemble --
ens = pd.concat(pat_scores, axis=1).mean(axis=1).rename("score").reset_index()
truth = df.groupby("volume")["tumor_present"].max().rename("y_true").reset_index()
ens = ens.merge(truth, on="volume")
pat_ens_auc = float(roc_auc_score(ens.y_true, ens.score))

slice_mean = float(np.mean(per_fold_slice_auc)); slice_sd = float(np.std(per_fold_slice_auc, ddof=1))
print("\n=== RESULTS ===", flush=True)
print(f"per-fold slice AUC mean+/-sd : {slice_mean:.4f} +/- {slice_sd:.4f}  (json had 0.9234 +/- 0.0079)", flush=True)
print(f"ENSEMBLE slice AUC           : {slice_ens_auc:.4f}  [{slice_ens_ci[0]:.3f}, {slice_ens_ci[1]:.3f}]  <-- NEW", flush=True)
print(f"ENSEMBLE patient AUC (check) : {pat_ens_auc:.4f}  (json had 0.8854)", flush=True)

json.dump({"ensemble_slice_auc": slice_ens_auc, "ensemble_slice_auc_ci": slice_ens_ci,
           "per_fold_slice_auc": [float(x) for x in per_fold_slice_auc],
           "slice_auc_mean": slice_mean, "slice_auc_sd": slice_sd,
           "ensemble_patient_auc_recomputed": pat_ens_auc},
          open(BTC_DIR / "limt_transfer" / "ensemble_slice_auc.json", "w"), indent=2)
print("\nsaved -> limt_transfer/ensemble_slice_auc.json", flush=True)
