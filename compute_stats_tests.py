"""Statistical tests computed from saved predictions (no training).

  - DeLong test for paired ROC-AUC differences between backbones and between
    input representations on the shared fold-0 test set
  - McNemar's exact test on the thresholded decisions
  - Wilcoxon signed-rank test, zero-shot vs fine-tuned patient scores on LiMT

  
"""
import os
import json
import itertools

import numpy as np
import pandas as pd
from scipy import stats
import config

BTC = str(config.WORK_DIR)
FT = os.path.join(BTC, "limt_finetune_seeded", "limt_finetune_kfold.json")
OUT = os.path.join(BTC, "stats_tests.json")


# ------------------------------------------------------------------------
# DeLong (Sun & Xu 2014 fast midrank)
# ------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """predictions_sorted_transposed: (k, n) with the m positive cases first.
    Returns (aucs (k,), covariance (k,k))."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r, :] = _compute_midrank(predictions_sorted_transposed[r, m:])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test that AUC_a == AUC_b on the same cases.
    Returns (auc_a, auc_b, z, p)."""
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort(kind="mergesort")  # positives first, stable
    label_1_count = int(y_true.sum())
    preds = np.vstack([np.asarray(prob_a, float)[order],
                       np.asarray(prob_b, float)[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ------------------------------------------------------------------------
# McNemar (paired classifier accuracy at the operating point)
# ------------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b):
    """Exact McNemar on two classifiers' hard predictions over the same cases.
    Compares the discordant pairs: n_ab = A wrong & B right, n_ba = A right & B
    wrong. Exact two-sided binomial p on min(n_ab, n_ba) ~ Bin(n_ab+n_ba, 0.5)
    (correct for any discordant count; no large-sample chi-square approximation)."""
    y = np.asarray(y_true).astype(int)
    a_ok = (np.asarray(pred_a).astype(int) == y)
    b_ok = (np.asarray(pred_b).astype(int) == y)
    n_ab = int(np.sum(~a_ok & b_ok))   # A wrong, B right  (favours B)
    n_ba = int(np.sum(a_ok & ~b_ok))   # A right, B wrong  (favours A)
    n_dis = n_ab + n_ba
    if n_dis == 0:
        return {"n_a_wrong_b_right": 0, "n_a_right_b_wrong": 0, "p": 1.0,
                "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean())}
    p = float(stats.binomtest(min(n_ab, n_ba), n_dis, 0.5).pvalue)
    return {"n_a_wrong_b_right": n_ab, "n_a_right_b_wrong": n_ba, "p": p,
            "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean())}


# ------------------------------------------------------------------------
# (A) backbone comparison on the shared single-fold test set
# ------------------------------------------------------------------------
def load_pred(tag, level):
    d = np.load(os.path.join(BTC, f"{tag}_test_predictions.npz"))
    if level == "slice":
        return d["y_true"], d["y_prob"]
    return d["pat_true"], d["pat_score"]


def load_threshold(tag):
    m = json.load(open(os.path.join(BTC, f"{tag}_metrics.json")))
    return float(m["best_threshold"])


def backbone_delong():
    reps = {"2d": ["2d_resnet18", "2d_vit", "2d_swin"],
            "2.5d": ["2.5d_resnet18", "2.5d_vit", "2.5d_swin"]}
    nice = {"resnet18": "ResNet-18", "vit": "ViT", "swin": "Swin"}
    out = {}
    for level in ("slice", "patient"):
        out[level] = []
        for rep, tags in reps.items():
            # verify identical labels across the three models (paired test premise)
            y_ref = load_pred(tags[0], level)[0]
            for t in tags[1:]:
                assert np.array_equal(load_pred(t, level)[0], y_ref), \
                    f"{t} labels differ from {tags[0]} at {level} level -- not paired!"
            for a, b in itertools.combinations(tags, 2):
                yt, pa = load_pred(a, level)
                _, pb = load_pred(b, level)
                auc_a, auc_b, z, p = delong_test(yt, pa, pb)
                out[level].append({
                    "rep": rep,
                    "model_a": nice[a.split("_", 1)[1]], "auc_a": round(auc_a, 4),
                    "model_b": nice[b.split("_", 1)[1]], "auc_b": round(auc_b, 4),
                    "z": round(z, 3), "p": p})
    # input-representation contrast within Swin (2d vs 2.5d vs 9ch)
    swin = {"2d": "2d_swin", "2.5d": "2.5d_swin", "9ch": "2.5d9ch_swin"}
    out["swin_rep"] = []
    for level in ("slice", "patient"):
        y_ref = load_pred(swin["2d"], level)[0]
        for a, b in itertools.combinations(swin, 2):
            yt, pa = load_pred(swin[a], level)
            _, pb = load_pred(swin[b], level)
            auc_a, auc_b, z, p = delong_test(yt, pa, pb)
            out["swin_rep"].append({
                "level": level, "rep_a": a, "auc_a": round(auc_a, 4),
                "rep_b": b, "auc_b": round(auc_b, 4), "z": round(z, 3), "p": p})
    return out


def backbone_mcnemar():
    """McNemar at each model's own operating threshold, slice level (2.5D)."""
    nice = {"resnet18": "ResNet-18", "vit": "ViT", "swin": "Swin"}
    tags = ["2.5d_resnet18", "2.5d_vit", "2.5d_swin"]
    preds = {}
    for t in tags:
        y, p = load_pred(t, "slice")
        preds[t] = (y, (p >= load_threshold(t)).astype(int))
    y_ref = preds[tags[0]][0]
    out = []
    for a, b in itertools.combinations(tags, 2):
        assert np.array_equal(preds[a][0], y_ref) and np.array_equal(preds[b][0], y_ref)
        r = mcnemar_test(y_ref, preds[a][1], preds[b][1])
        out.append({"model_a": nice[a.split("_", 1)[1]], "acc_a": round(r["acc_a"], 4),
                    "model_b": nice[b.split("_", 1)[1]], "acc_b": round(r["acc_b"], 4),
                    "n_a_wrong_b_right": r["n_a_wrong_b_right"],
                    "n_a_right_b_wrong": r["n_a_right_b_wrong"], "p": r["p"]})
    return out


def finetune_mcnemar():
    """McNemar on LiMT case-level decisions: zero-shot vs fine-tuned Swin-2.5D,
    each at its own F1-optimal pooled threshold (from the real kfold run)."""
    k = json.load(open(FT))
    pc = k["per_case"]
    ids = sorted(pc)
    y = np.array([pc[c]["y"] for c in ids])
    zs = np.array([pc[c]["zero_shot"] for c in ids])
    ft = np.array([pc[c]["fine_tuned"] for c in ids])
    thr_zs = k["zero_shot"]["threshold"]
    thr_ft = k["fine_tuned"]["threshold"]
    r = mcnemar_test(y, (zs >= thr_zs).astype(int), (ft >= thr_ft).astype(int))
    return {"acc_zero_shot": round(r["acc_a"], 4), "acc_fine_tuned": round(r["acc_b"], 4),
            "n_zs_wrong_ft_right": r["n_a_wrong_b_right"],
            "n_zs_right_ft_wrong": r["n_a_right_b_wrong"], "p": r["p"]}


# ------------------------------------------------------------------------
# (B) Wilcoxon signed-rank on the LiMT fine-tuning arm
# ------------------------------------------------------------------------
def finetune_wilcoxon():
    k = json.load(open(FT))
    pc = k["per_case"]
    ids = sorted(pc)
    zs = np.array([pc[c]["zero_shot"] for c in ids])
    ft = np.array([pc[c]["fine_tuned"] for c in ids])
    y = np.array([pc[c]["y"] for c in ids])

    # paired one-sided (fine-tuned > zero-shot) on per-case scores, all 150 cases
    w_all = stats.wilcoxon(ft, zs, alternative="greater", zero_method="wilcox")
    # diseased-only: does fine-tuning raise the score on true-positive cases?
    dm = y == 1
    w_dis = stats.wilcoxon(ft[dm], zs[dm], alternative="greater", zero_method="wilcox")
    # normals: does fine-tuning LOWER the score (fewer false positives)?
    nm = y == 0
    w_norm = stats.wilcoxon(ft[nm], zs[nm], alternative="less", zero_method="wilcox")

    med_delta = float(np.median(ft - zs))
    return {
        "n_cases": len(ids), "n_diseased": int(dm.sum()), "n_normal": int(nm.sum()),
        "median_delta_score": round(med_delta, 4),
        "auc_zero_shot": k["zero_shot"]["auc"], "auc_fine_tuned": k["fine_tuned"]["auc"],
        "delta_auc": k["delta_patient_auc"], "delta_auc_ci": k["delta_patient_auc_ci"],
        "wilcoxon_all": {"stat": float(w_all.statistic), "p": float(w_all.pvalue)},
        "wilcoxon_diseased": {"stat": float(w_dis.statistic), "p": float(w_dis.pvalue)},
        "wilcoxon_normal": {"stat": float(w_norm.statistic), "p": float(w_norm.pvalue)},
    }


def main():
    # self-check DeLong on a trivial case (two identical predictors -> p==1)
    rng = np.random.default_rng(0)
    yy = (rng.random(200) > 0.6).astype(int)
    pp = rng.random(200)
    _, _, _, p_id = delong_test(yy, pp, pp)
    assert abs(p_id - 1.0) < 1e-6, f"DeLong self-check failed: p={p_id}"

    bb = backbone_delong()
    bb_mc = backbone_mcnemar()
    ft = finetune_wilcoxon()
    ft_mc = finetune_mcnemar()
    json.dump({"backbone_delong": bb, "backbone_mcnemar": bb_mc,
               "finetune_wilcoxon": ft, "finetune_mcnemar": ft_mc},
              open(OUT, "w"), indent=2)

    print("=== (A) DeLong: backbone AUC comparison (shared single-fold test set) ===")
    for level in ("slice", "patient"):
        print(f"\n  [{level} level]")
        for r in bb[level]:
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
                  "*" if r["p"] < 0.05 else "ns"
            print(f"    {r['rep']:4s}  {r['model_a']:10s}({r['auc_a']:.3f}) vs "
                  f"{r['model_b']:10s}({r['auc_b']:.3f})  z={r['z']:+.2f}  "
                  f"p={r['p']:.4f}  {sig}")
    print("\n  [Swin input representation, DeLong]")
    for r in bb["swin_rep"]:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
              "*" if r["p"] < 0.05 else "ns"
        print(f"    {r['level']:7s} {r['rep_a']:4s}({r['auc_a']:.3f}) vs "
              f"{r['rep_b']:4s}({r['auc_b']:.3f})  z={r['z']:+.2f} p={r['p']:.4f} {sig}")

    print("\n=== McNemar: backbone decisions at operating point (2.5D, slice level) ===")
    for r in bb_mc:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
              "*" if r["p"] < 0.05 else "ns"
        print(f"    {r['model_a']:10s}(acc {r['acc_a']:.3f}) vs {r['model_b']:10s}"
              f"(acc {r['acc_b']:.3f})  discordant {r['n_a_wrong_b_right']}/"
              f"{r['n_a_right_b_wrong']}  p={r['p']:.2e}  {sig}")

    print("\n=== McNemar: LiMT zero-shot vs fine-tuned decisions (150 cases) ===")
    print(f"    acc {ft_mc['acc_zero_shot']:.3f} -> {ft_mc['acc_fine_tuned']:.3f}  "
          f"(zs-wrong/ft-right={ft_mc['n_zs_wrong_ft_right']}, "
          f"zs-right/ft-wrong={ft_mc['n_zs_right_ft_wrong']})  p={ft_mc['p']:.2e}")

    print("\n=== (B) Wilcoxon signed-rank: LiMT zero-shot vs fine-tuned (150 cases) ===")
    print(f"  patient AUC {ft['auc_zero_shot']:.4f} -> {ft['auc_fine_tuned']:.4f} "
          f"(Delta {ft['delta_auc']:+.4f}, median per-case score Delta {ft['median_delta_score']:+.4f})")
    print(f"  all cases  (ft>zs): W={ft['wilcoxon_all']['stat']:.0f}  p={ft['wilcoxon_all']['p']:.2e}")
    print(f"  diseased   (ft>zs): W={ft['wilcoxon_diseased']['stat']:.0f}  "
          f"p={ft['wilcoxon_diseased']['p']:.2e}  (n={ft['n_diseased']})")
    print(f"  normals    (ft<zs): W={ft['wilcoxon_normal']['stat']:.0f}  "
          f"p={ft['wilcoxon_normal']['p']:.2e}  (n={ft['n_normal']})")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
