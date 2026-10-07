"""Error analysis of the out-of-fold Swin-2.5D predictions on LiTS.
    
"""
import os
import json

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from cam_swin import load_25d, META, FOLDS, RAW_MASK_DIR, OUT_DIR  # noqa: E402
import config

OOF_CSV = os.path.join(OUT_DIR, "oof_slice_predictions.csv")
HU_CSV = config.HU_CSV


def tumor_area_per_slice(volumes):
    """voxel count of tumor (label==2) per (volume, slice) -- lesion size proxy."""
    out = {}
    for v in volumes:
        seg = np.asanyarray(nib.load(
            os.path.join(RAW_MASK_DIR, f"segmentation-{v.split('-')[1]}.nii")).dataobj)
        area = (seg == 2).sum(axis=(0, 1))       # per-slice tumor voxel count
        out[v] = area
    return out


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    if not os.path.exists(OOF_CSV):
        raise SystemExit(f"missing {OOF_CSV} -- run robustness_eval.py --max_per_vol 0 first")
    df = pd.read_csv(OOF_CSV)
    tau = youden(df.label, df.prob)
    df["pred"] = (df.prob >= tau).astype(int)
    df["kind"] = np.select(
        [(df.label == 1) & (df.pred == 1), (df.label == 0) & (df.pred == 0),
         (df.label == 0) & (df.pred == 1), (df.label == 1) & (df.pred == 0)],
        ["TP", "TN", "FP", "FN"])
    n = len(df)
    counts = df.kind.value_counts().to_dict()
    tp, tn = counts.get("TP", 0), counts.get("TN", 0)
    fp, fn = counts.get("FP", 0), counts.get("FN", 0)
    print(f"threshold(Youden)={tau:.3f}  n={n}", flush=True)
    print(f"TP={tp} TN={tn} FP={fp} FN={fn} | "
          f"sens={tp/(tp+fn):.3f} spec={tn/(tn+fp):.3f} "
          f"FP-rate={fp/(fp+tn):.3f} FN-rate={fn/(fn+tp):.3f}", flush=True)

    # -- lesion size: tumor area per slice ----------------------------------
    areas = tumor_area_per_slice(df.volume.unique())
    df["tumor_area"] = [areas[r.volume][int(r.slice_idx)] if int(r.slice_idx) < len(areas[r.volume]) else 0
                        for r in df.itertuples()]
    pos = df[df.label == 1].copy()
    bins = [0, 50, 200, 500, 2000, np.inf]
    labs = ["<50", "50-200", "200-500", "500-2k", ">2k"]
    pos["sizebin"] = pd.cut(pos.tumor_area, bins=bins, labels=labs)
    recall_by_size = pos.groupby("sizebin", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "recall": (g.pred == 1).mean(),
                             "median_prob": g.prob.median()}))
    print("\n== recall vs lesion size (voxels) ==", flush=True)
    print(recall_by_size.to_string(), flush=True)

    # -- false positives: proximity to a real tumor slice (near-miss?) ------
    tumor_slices = {v: set(pos[pos.volume == v].slice_idx) for v in pos.volume.unique()}
    def dist_to_tumor(v, z):
        ts = tumor_slices.get(v, set())
        return min([abs(z - t) for t in ts]) if ts else 9999
    fp_df = df[df.kind == "FP"].copy()
    fp_df["dist"] = [dist_to_tumor(r.volume, r.slice_idx) for r in fp_df.itertuples()]
    near = (fp_df.dist <= 3).mean()
    print(f"\n== false positives (n={fp}) ==", flush=True)
    print(f"  within 3 slices of a real tumor: {near:.2%} (near-miss / partial-volume)", flush=True)
    print(f"  median distance to nearest tumor slice: {fp_df.dist.median():.0f}", flush=True)

    # -- diffuse parenchyma / contrast: stratify errors by liver HU ---------
    hu_res = None
    if os.path.exists(HU_CSV):
        hu = pd.read_csv(HU_CSV)
        hu = hu[hu.dataset == "LiTS"][["case_id", "hu_median"]].rename(
            columns={"case_id": "volume"})
        m = df.merge(hu, on="volume", how="left").dropna(subset=["hu_median"])
        if len(m):
            m["hu_bin"] = pd.qcut(m.hu_median, q=3, labels=["low HU", "mid HU", "high HU"])
            def err_rates(g):
                p, nn = g[g.label == 1], g[g.label == 0]
                return pd.Series({
                    "n_vol": g.volume.nunique(),
                    "FN_rate": (p.pred == 0).mean() if len(p) else np.nan,
                    "FP_rate": (nn.pred == 1).mean() if len(nn) else np.nan,
                    "hu_range": f"{g.hu_median.min():.0f}-{g.hu_median.max():.0f}"})
            hu_res = m.groupby("hu_bin", observed=True).apply(err_rates)
            print("\n== errors vs liver HU (diffuse/contrast proxy) ==", flush=True)
            print(hu_res.to_string(), flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    summ = {"threshold": tau, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
            "fp_near_tumor_frac": float(near),
            "fp_median_dist": float(fp_df.dist.median()),
            "recall_by_size": json.loads(recall_by_size.reset_index().to_json(orient="records")),
            "errors_by_hu": (json.loads(hu_res.reset_index().astype(str).to_json(orient="records"))
                             if hu_res is not None else None)}
    json.dump(summ, open(os.path.join(OUT_DIR, "error_analysis.json"), "w"), indent=2)

    # -- quantitative figure: recall-vs-size + error-vs-HU ------------------
    ncol = 2 if hu_res is not None else 1
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
    if ncol == 1:
        ax = [ax]
    rb = recall_by_size.reset_index()
    ax[0].bar(rb.sizebin.astype(str), rb.recall, color="#4C72B0")
    for i, (r, nn) in enumerate(zip(rb.recall, rb.n)):
        ax[0].text(i, r + 0.02, f"n={int(nn)}", ha="center", fontsize=7)
    ax[0].set_xlabel("tumor area in slice (voxels)"); ax[0].set_ylabel("recall (TP rate)")
    ax[0].set_title("Recall vs lesion size"); ax[0].set_ylim(0, 1.05)
    if hu_res is not None:
        hb = hu_res.reset_index()
        x = np.arange(len(hb)); w = 0.38
        ax[1].bar(x - w/2, hb.FN_rate.astype(float), w, label="FN rate (missed tumor)", color="#C44E52")
        ax[1].bar(x + w/2, hb.FP_rate.astype(float), w, label="FP rate (false alarm)", color="#DD8452")
        ax[1].set_xticks(x); ax[1].set_xticklabels(hb.hu_bin.astype(str))
        ax[1].set_xlabel("median liver HU (diffuse/contrast proxy)"); ax[1].set_ylabel("error rate")
        ax[1].set_title("Errors vs liver attenuation"); ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "error_analysis.png"), dpi=150); plt.close(fig)

    # -- qualitative FP / FN panels -----------------------------------------
    vol_max = pd.read_csv(META).groupby("volume")["slice_idx"].max().to_dict()
    def panel(rows, title, fname, show_contour):
        rows = rows.head(4)
        if len(rows) == 0:
            return
        fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 3.2))
        if len(rows) == 1:
            axes = [axes]
        for ax_, r in zip(axes, rows.itertuples()):
            x = load_25d(r.volume, int(r.slice_idx), vol_max)[0, 1].numpy()
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            ax_.imshow(x, cmap="gray")
            if show_contour:
                seg = np.asanyarray(nib.load(os.path.join(
                    RAW_MASK_DIR, f"segmentation-{r.volume.split('-')[1]}.nii")).dataobj)
                ms = (seg[:, :, int(r.slice_idx)] == 2).astype(np.float32)
                ms = F.interpolate(torch.from_numpy(ms)[None, None], (224, 224),
                                   mode="nearest")[0, 0].numpy()
                if ms.sum():
                    ax_.contour(ms, [0.5], colors="lime", linewidths=1.2)
            ax_.set_title(f"{r.volume} z={int(r.slice_idx)}\np={r.prob:.2f}", fontsize=7)
            ax_.axis("off")
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.close(fig)

    # most confident FPs (high prob, no tumor), one per volume for diversity
    fp_show = (fp_df.sort_values("prob", ascending=False)
               .drop_duplicates("volume"))
    panel(fp_show, "Representative false positives (tumor-free liver called tumor)",
          "fp_examples.png", show_contour=False)
    # FNs in a VISIBLE size range (100-1000 vox) so the missed lesion is discernible,
    # one per volume; the absolute-smallest FNs are sub-visible and uninformative
    fn_df = df[(df.kind == "FN") & (df.tumor_area.between(100, 1000))].copy()
    fn_show = fn_df.sort_values("tumor_area").drop_duplicates("volume")
    panel(fn_show, "Representative false negatives (small missed lesions, GT contour)",
          "fn_examples.png", show_contour=True)

    print(f"\nsaved -> {OUT_DIR}", flush=True)
    print("ERROR ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
