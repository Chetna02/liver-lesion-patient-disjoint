"""Transfer-strategy ablation on the LiMT folds.

    scratch : ImageNet initialisation, trained on LiMT only
    mixed   : ImageNet initialisation, trained on LiTS + LiMT pooled
Zero-shot and LiTS->LiMT fine-tuning results are read from limt_finetune.py.

    
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import json
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score

import limt_finetune as LF  # noqa: E402
from run_all_experiments import build_model, _base_tf, BATCH, DEVICE  # noqa: E402
import config

LITS_IMG_DIR = str(config.WORK_DIR / "images")
LIMT_IMG_DIR = str(LF.LIMT_IMG_DIR)
LITS_META = str(config.WORK_DIR / "metadata_slices.csv")
OUT = os.path.join(str(LF.OUT_DIR), "ablation")
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# -- mixed (LiTS+LiMT) 2.5D dataset: channel-0 of 3 adjacent slices, per-source dir --
class MixedTumor25DTrain(Dataset):
    def __init__(self, df, lits_max, limt_max, train=True):
        self.df = df.reset_index(drop=True)
        self.lits_max, self.limt_max = lits_max, limt_max
        self.tf = _base_tf(train)

    def __len__(self):
        return len(self.df)

    def _chan0(self, src, v, idx):
        d, mx = (LITS_IMG_DIR, self.lits_max) if src == "lits" else (LIMT_IMG_DIR, self.limt_max)
        idx = int(min(max(idx, 0), mx[v]))
        a = np.array(Image.open(os.path.join(d, f"{v}_slice_{idx:03d}.jpg")).convert("RGB"))
        return a[:, :, 0]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        s, v, z = r["source"], r["volume"], int(r["slice_idx"])
        stack = np.stack([self._chan0(s, v, z - 1), self._chan0(s, v, z),
                          self._chan0(s, v, z + 1)], axis=-1)
        return self.tf(Image.fromarray(stack)), torch.tensor(float(r["tumor_present"]))


def train_scratch(limt_df, maxslice, tr_ids, va_ids, args):
    """ImageNet-init, train on LiMT training split only (no LiTS)."""
    _, _, _, tl, vl, _ = LF.loaders_for(limt_df, maxslice, tr_ids, va_ids, tr_ids[:1])
    tr_df = limt_df[limt_df.volume.isin(tr_ids)]
    model = build_model("swin")            # ImageNet-pretrained stem, no LiTS weights
    LF.finetune(model, tl, vl, tr_df, args.epochs, args.lr, args.weight_decay,
                args.patience, smoke=args.smoke, tag="scratch ")
    return model


def train_mixed(limt_df, lits_df, limt_max, lits_max, tr_ids, va_ids, args):
    """ImageNet-init, train on ALL LiTS slices + the LiMT training split, with a
    source-and-class balanced sampler (4 cells equally likely) capped to a small
    epoch so LiTS's 8x size does not drown the LiMT signal or blow up runtime."""
    lits = lits_df.assign(source="lits")[["source", "volume", "slice_idx", "tumor_present"]]
    limt = (limt_df[limt_df.volume.isin(tr_ids)].assign(source="limt")
            [["source", "volume", "slice_idx", "tumor_present"]])
    comb = pd.concat([lits, limt], ignore_index=True)

    key = comb.source + "_" + comb.tumor_present.astype(str)
    cnt = key.value_counts().to_dict()
    w = key.map(lambda k: 1.0 / cnt[k]).values
    cap = min(len(comb), 3 * len(limt))    # ~3x LiMT-fold slices per epoch
    sampler = WeightedRandomSampler(w, num_samples=int(cap), replacement=True)

    ds = MixedTumor25DTrain(comb, lits_max, limt_max, train=True)
    tl = DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    va_df = limt_df[limt_df.volume.isin(va_ids)]
    vl = DataLoader(LF.LimtTumor25D(va_df, limt_max), batch_size=BATCH, shuffle=False,
                    num_workers=0, pin_memory=True)
    print(f"  mixed pool: {len(lits)} LiTS + {len(limt)} LiMT slices, cap {int(cap)}/epoch",
          flush=True)
    model = build_model("swin")
    LF.finetune(model, tl, vl, comb, args.epochs, args.lr, args.weight_decay,
                args.patience, smoke=args.smoke, tag="mixed ")
    return model


def run_mode(mode, limt_df, lits_df, limt_max, lits_max, args):
    case_lab = limt_df.groupby("volume")["tumor_present"].max()
    cases = case_lab.index.to_numpy(); labels = case_lab.to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)  # == fine-tune folds

    per_case = {}
    for k, (trv_idx, te_idx) in enumerate(sgkf.split(cases, labels, groups=cases)):
        test_ids = sorted(cases[te_idx].tolist())
        tr_ids, va_ids = train_test_split(cases[trv_idx], test_size=0.15,
                                          stratify=labels[trv_idx], random_state=SEED + k)
        tr_ids, va_ids = sorted(tr_ids.tolist()), sorted(va_ids.tolist())
        print(f"\n--- {mode} fold {k}: train={len(tr_ids)} val={len(va_ids)} "
              f"test={len(test_ids)} ---", flush=True)

        if mode == "scratch":
            model = train_scratch(limt_df, limt_max, tr_ids, va_ids, args)
        else:
            model = train_mixed(limt_df, lits_df, limt_max, lits_max, tr_ids, va_ids, args)

        te_df = limt_df[limt_df.volume.isin(test_ids)].reset_index(drop=True)
        te_loader = DataLoader(LF.LimtTumor25D(te_df, limt_max), batch_size=BATCH,
                               shuffle=False, num_workers=0, pin_memory=True)
        case = LF.case_scores(model, te_df, te_loader)
        per_case.update({v: {"y": ys[0], mode: ys[1]} for v, ys in case.items()})
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    vols = sorted(per_case)
    yt = np.array([per_case[v]["y"] for v in vols])
    sc = np.array([per_case[v][mode] for v in vols])
    auc = float(roc_auc_score(yt, sc))
    print(f"\n=== {mode.upper()}: pooled patient AUC = {auc:.4f} (n={len(vols)}) ===", flush=True)
    return {"auc": auc, "per_case": per_case}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="scratch,mixed")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)     # higher than fine-tune (from-scratch)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seed_everything(SEED)

    limt_df = pd.read_csv(LF.LIMT_META_CSV)
    lits_df = pd.read_csv(LITS_META)
    limt_max = limt_df.groupby("volume")["slice_idx"].max().to_dict()
    lits_max = lits_df.groupby("volume")["slice_idx"].max().to_dict()

    os.makedirs(OUT, exist_ok=True)
    results = {}
    for mode in [m.strip() for m in args.modes.split(",")]:
        results[mode] = run_mode(mode, limt_df, lits_df, limt_max, lits_max, args)
        json.dump(results, open(os.path.join(OUT, "ablation_results.json"), "w"),
                  indent=2, default=float)

    # -- assemble the 4-way comparison against the seeded fine-tune arm ------
    kf = os.path.join(str(LF.OUT_DIR), "..", "limt_finetune_seeded", "limt_finetune_kfold.json")
    kf = os.path.normpath(kf)
    if os.path.exists(kf):
        ft = json.load(open(kf))
        zs_auc, seq_auc = ft["zero_shot"]["auc"], ft["fine_tuned"]["auc"]
        rows = [("zero-shot (LiTS only)", zs_auc),
                ("scratch (LiMT only)", results.get("scratch", {}).get("auc")),
                ("mixed (LiTS+LiMT)", results.get("mixed", {}).get("auc")),
                ("sequential (LiTS->LiMT)", seq_auc)]
        print("\n=== TRANSFER-STRATEGY ABLATION (patient AUC, LiMT 5-fold OOF) ===", flush=True)
        for name, a in rows:
            print(f"  {name:26s} {a:.4f}" if a is not None else f"  {name:26s}  --", flush=True)

        # paired bootstrap vs sequential, over shared cases, for each computed mode
        pc_ft = ft["per_case"]
        for mode in results:
            pc = results[mode]["per_case"]
            vols = sorted(set(pc) & set(pc_ft))
            yt = np.array([pc[v]["y"] for v in vols])
            s_mode = np.array([pc[v][mode] for v in vols])
            s_seq = np.array([pc_ft[v]["fine_tuned"] for v in vols])
            d, ci, p = LF.paired_auc_bootstrap(vols, yt, s_mode, s_seq, args.n_boot)
            print(f"  sequential - {mode}: dAUC={-d:+.4f} [{-ci[1]:+.3f},{-ci[0]:+.3f}] "
                  f"P(seq>mode)={1-p:.3f}", flush=True)
        json.dump({"table": rows, "modes": {m: results[m]["auc"] for m in results},
                   "zero_shot": zs_auc, "sequential": seq_auc},
                  open(os.path.join(OUT, "ablation_summary.json"), "w"), indent=2, default=float)

    print("\nABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
