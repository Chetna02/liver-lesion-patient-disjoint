"""Train and test ResNet-18, ViT and Swin with 2D and 2.5D input on the fold-0 split.

2D   : the three intensity windows of one slice.
2.5D : the liver window of slices z-1, z, z+1.

Runs are checkpointed and skipped once finished, so the script can be restarted.

    python run_all_experiments.py
    python run_all_experiments.py --only 2.5d:swin
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd            
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve, confusion_matrix)
import config

OUT_DIR = config.WORK_DIR
IMG_DIR = OUT_DIR / "images"
FLAG_DIR = OUT_DIR / "flags"
LOCK_FILE = OUT_DIR / "run.lock"
CMP_DIR = OUT_DIR / "comparison"
META_CSV = OUT_DIR / "metadata_slices.csv"
TRAIN_CSV = OUT_DIR / "train_slices.csv"    # real slices only (2.5D needs true neighbors)
VAL_CSV = OUT_DIR / "val_slices.csv"
TEST_CSV = OUT_DIR / "test_slices.csv"
PREP_SCRIPT = Path(__file__).with_name("prep_patientsplit.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pid_alive(pid):
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    except Exception:
        return False


def acquire_lock():
    if LOCK_FILE.exists():
        try:
            old = int(LOCK_FILE.read_text().strip())
        except Exception:
            old = None
        if old and _pid_alive(old):
            print(f"another instance is already running (pid {old}) -> exiting", flush=True)
            sys.exit(0)
        print(f"stale lock (pid {old} dead) -> taking over", flush=True)
    LOCK_FILE.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: LOCK_FILE.exists() and LOCK_FILE.unlink())
BACKBONES = {
    "resnet18": "resnet18",
    "vit": "vit_base_patch16_224",
    "swin": "swin_base_patch4_window7_224",
}
MODES = ["2d", "2.5d"]
IMG_SIZE = 224
BATCH = 32

LEAKY_BASELINE = {
    "resnet18": {"F1": 0.949, "ROC-AUC": 0.9986},
    "vit":      {"F1": 0.959, "ROC-AUC": 0.9988},
    "swin":     {"F1": 0.959, "ROC-AUC": 0.9990},
}


# ---------------- prep guard ----------------
def ensure_prep():
    if TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists() and META_CSV.exists():
        print("prep outputs present -> skip prep", flush=True)
        return
    print("prep outputs missing -> running prep_patientsplit.py ...", flush=True)
    subprocess.run([sys.executable, str(PREP_SCRIPT)], check=True)


# ---------------- neighbor index (for 2.5D) ----------------
def build_volume_maxslice():
    meta = pd.read_csv(META_CSV)
    return meta.groupby("volume")["slice_idx"].max().to_dict()


VOL_MAXSLICE = None


def slice_path(volume, idx):
    return IMG_DIR / f"{volume}_slice_{idx:03d}.jpg"


# ---------------- datasets ----------------
class Tumor2D(Dataset):
    """Project-native 3-channel windowed JPG, single slice."""
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.tf = _base_tf(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        return self.tf(img), torch.tensor(float(r["tumor_present"]))


class Tumor25D(Dataset):
    """3 adjacent slices (z-1,z,z+1), liver-window channel (channel 0), stacked."""
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.tf = _base_tf(train)

    def __len__(self):
        return len(self.df)

    def _liverchan(self, volume, idx):
        maxs = VOL_MAXSLICE[volume]
        idx = int(min(max(idx, 0), maxs))
        p = slice_path(volume, idx)
        arr = np.array(Image.open(p).convert("RGB"))[:, :, 0]  # channel 0 = liver window
        return arr

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v, z = r["volume"], int(r["slice_idx"])
        stack = np.stack([self._liverchan(v, z - 1),
                          self._liverchan(v, z),
                          self._liverchan(v, z + 1)], axis=-1)  # HxWx3 uint8
        return self.tf(Image.fromarray(stack)), torch.tensor(float(r["tumor_present"]))


def _base_tf(train):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops += [transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
    return transforms.Compose(ops)


def make_loaders(mode):
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # re-assert no patient leak
    s_tr, s_va, s_te = set(train_df.volume), set(val_df.volume), set(test_df.volume)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "patient leak!"

    DS = Tumor2D if mode == "2d" else Tumor25D
    train_ds, val_ds, test_ds = DS(train_df, True), DS(val_df, False), DS(test_df, False)

    labels = train_df["tumor_present"].values.astype(int)
    cw = 1.0 / np.bincount(labels)
    sampler = WeightedRandomSampler(cw[labels], num_samples=len(labels), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    return train_df, val_df, test_df, train_loader, val_loader, test_loader


# ---------------- model ----------------
def build_model(backbone):
    m = timm.create_model(BACKBONES[backbone], pretrained=True, num_classes=1)
    return m.to(DEVICE)


def set_freeze(model, freeze):
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in model.get_classifier().parameters():
        p.requires_grad = True


# ---------------- eval helpers ----------------
def eval_probs(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, lb in loader:
            out = model(imgs.to(DEVICE))
            ps.extend(torch.sigmoid(out).float().cpu().numpy().flatten())
            ys.extend(lb.numpy().flatten())
    return np.array(ys).astype(int), np.array(ps)


def slice_metrics(y, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {"ROC-AUC": float(roc_auc_score(y, p)),
            "Accuracy": float(accuracy_score(y, yp)),
            "Precision": float(precision_score(y, yp, zero_division=0)),
            "Recall_Sensitivity": float(recall_score(y, yp, zero_division=0)),
            "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "F1": float(f1_score(y, yp, zero_division=0))}


def patient_metrics(test_df, p, thr):
    te = test_df.copy()
    te["prob"] = p
    pat = te.groupby("volume").agg(
        y_true=("tumor_present", "max"),
        score=("prob", lambda s: np.sort(s.values)[-3:].mean())).reset_index()
    pred = (pat.score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(pat.y_true, pred, labels=[0, 1]).ravel()
    return {"n_patients": int(len(pat)),
            "ROC-AUC": float(roc_auc_score(pat.y_true, pat.score)),
            "Accuracy": float(accuracy_score(pat.y_true, pred)),
            "Precision": float(precision_score(pat.y_true, pred, zero_division=0)),
            "Recall_Sensitivity": float(recall_score(pat.y_true, pred, zero_division=0)),
            "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "F1": float(f1_score(pat.y_true, pred, zero_division=0))}, pat


# ---------------- one experiment ----------------
def run_one(mode, backbone, epochs):
    tag = f"{mode}_{backbone}"
    flag = FLAG_DIR / f"{tag}.done"
    if flag.exists():
        print(f"[{tag}] flag present -> skip", flush=True)
        return
    print(f"\n{'='*60}\n[{tag}] START  ({time.strftime('%H:%M:%S')})\n{'='*60}", flush=True)

    train_df, val_df, test_df, tl, vl, tel = make_loaders(mode)
    model = build_model(backbone)
    pos = float((train_df.tumor_present == 1).sum()); neg = float((train_df.tumor_present == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt = OUT_DIR / f"{tag}_checkpoint.pth"
    best_path = OUT_DIR / f"{tag}_best.pth"
    start_epoch, best_val_f1, no_improve, hist = 0, 0.0, 0, []
    freeze_epochs, patience = 3, 6
    set_freeze(model, True)

    if ckpt.exists():
        ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
        best_val_f1 = ck["best_val_f1"]; no_improve = ck.get("no_improve", 0)
        if start_epoch >= freeze_epochs:
            set_freeze(model, False)
        print(f"[{tag}] resumed from epoch {start_epoch} (best_val_f1={best_val_f1:.4f})", flush=True)

    for epoch in range(start_epoch, epochs):
        if epoch == freeze_epochs:
            set_freeze(model, False)
        model.train(); run = 0.0
        for imgs, lb in tqdm(tl, desc=f"[{tag}] ep{epoch+1}/{epochs}", leave=False):
            imgs = imgs.to(DEVICE); lb = lb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lb)
            loss.backward(); optimizer.step(); run += loss.item() * imgs.size(0)
        scheduler.step()
        yv, pv = eval_probs(model, vl)
        vf1 = f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)
        print(f"[{tag}] ep{epoch+1} loss={run/len(tl.dataset):.4f} val_f1={vf1:.4f}", flush=True)

        improved = vf1 > best_val_f1 + 1e-6
        if improved:
            best_val_f1 = vf1; no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_val_f1": best_val_f1,
                    "no_improve": no_improve}, ckpt)
        hist.append({"epoch": epoch + 1, "val_f1": vf1})
        if no_improve >= patience:
            print(f"[{tag}] early stop at epoch {epoch+1}", flush=True)
            break

    # ---- final eval with val-tuned threshold ----
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False))
    yv, pv = eval_probs(model, vl)
    prec, rec, thrs = precision_recall_curve(yv, pv)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    thr = float(thrs[int(np.nanargmax(f1s))]) if len(thrs) else 0.5

    yt, pt = eval_probs(model, tel)
    sm = slice_metrics(yt, pt, thr)
    pm, pat = patient_metrics(test_df, pt, thr)
    print(f"[{tag}] SLICE  AUC={sm['ROC-AUC']:.4f} F1={sm['F1']:.4f} "
          f"sens={sm['Recall_Sensitivity']:.4f} spec={sm['Specificity']:.4f}", flush=True)
    print(f"[{tag}] PATIENT AUC={pm['ROC-AUC']:.4f} F1={pm['F1']:.4f} "
          f"sens={pm['Recall_Sensitivity']:.4f} spec={pm['Specificity']:.4f} (n={pm['n_patients']})", flush=True)

    np.savez(OUT_DIR / f"{tag}_test_predictions.npz",
             y_true=yt.astype(np.int64), y_prob=pt.astype(np.float32),
             pat_true=pat.y_true.values.astype(np.int64), pat_score=pat.score.values.astype(np.float32))
    json.dump({"mode": mode, "backbone": backbone, "best_threshold": thr,
               "slice": sm, "patient": pm, "history": hist},
              open(OUT_DIR / f"{tag}_metrics.json", "w"), indent=2)
    flag.write_text("done")
    print(f"[{tag}] DONE", flush=True)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ---------------- comparison ----------------
def build_comparison():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CMP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        for bb in BACKBONES:
            f = OUT_DIR / f"{mode}_{bb}_metrics.json"
            if not f.exists():
                continue
            m = json.load(open(f))
            rows.append({
                "mode": mode, "backbone": bb,
                "slice_AUC": m["slice"]["ROC-AUC"], "slice_F1": m["slice"]["F1"],
                "slice_sens": m["slice"]["Recall_Sensitivity"], "slice_spec": m["slice"]["Specificity"],
                "patient_AUC": m["patient"]["ROC-AUC"], "patient_F1": m["patient"]["F1"],
                "patient_sens": m["patient"]["Recall_Sensitivity"], "patient_spec": m["patient"]["Specificity"],
                "leaky_AUC_orig": LEAKY_BASELINE.get(bb, {}).get("ROC-AUC"),
                "leaky_F1_orig": LEAKY_BASELINE.get(bb, {}).get("F1"),
            })
    if not rows:
        print("no metrics yet for comparison", flush=True)
        return
    df = pd.DataFrame(rows)
    df.to_csv(CMP_DIR / "comparison_table.csv", index=False)
    print("\n=== COMPARISON TABLE ===", flush=True)
    print(df.to_string(index=False), flush=True)

    # ROC curves (slice-level)
    fig, ax = plt.subplots(figsize=(6, 6))
    from sklearn.metrics import roc_curve
    for mode in MODES:
        for bb in BACKBONES:
            npz = OUT_DIR / f"{mode}_{bb}_test_predictions.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
            auc = roc_auc_score(d["y_true"], d["y_prob"])
            ax.plot(fpr, tpr, label=f"{mode} {bb} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Slice-level ROC (patient-disjoint test)")
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(CMP_DIR / "roc_slice.png", dpi=150); plt.close(fig)

    # 2D vs 2.5D patient-AUC bars
    fig, ax = plt.subplots(figsize=(7, 4))
    bbs = list(BACKBONES); x = np.arange(len(bbs)); w = 0.35
    for j, mode in enumerate(MODES):
        vals = [df[(df["mode"] == mode) & (df["backbone"] == b)]["patient_AUC"].values for b in bbs]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (j - 0.5) * w, vals, w, label=mode)
    ax.set_xticks(x); ax.set_xticklabels(bbs); ax.set_ylabel("patient-level AUC")
    ax.set_title("2D vs 2.5D (patient-level)"); ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(CMP_DIR / "patient_auc_2d_vs_25d.png", dpi=150); plt.close(fig)
    print(f"\nsaved comparison -> {CMP_DIR}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--only", default=None, help="mode:backbone, e.g. 2d:resnet18")
    args = ap.parse_args()

    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    acquire_lock()
    ensure_prep()
    global VOL_MAXSLICE
    VOL_MAXSLICE = build_volume_maxslice()

    if args.only:
        mode, bb = args.only.split(":")
        run_one(mode, bb, args.epochs)
        build_comparison()
        return

    # ResNet first (fastest) so any bug surfaces cheapest, then the rest.
    order = [("2d", "resnet18"), ("2d", "vit"), ("2d", "swin"),
             ("2.5d", "resnet18"), ("2.5d", "vit"), ("2.5d", "swin")]
    for mode, bb in order:
        run_one(mode, bb, args.epochs)
        build_comparison()   # refresh after each so partial results are always available
    print("\nALL EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
