"""Nine-channel variant: all three windows for each of three adjacent slices.
    
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import timm

import run_all_experiments as R

OUT = R.OUT_DIR
DIR9 = OUT / "ch9_25d"
FLAGS9 = DIR9 / "flags"
LOCK = DIR9 / "ch9.lock"
BACKBONES = R.BACKBONES
EPOCHS = 30
ORDER = ["resnet18", "vit", "swin"]


def acquire_lock():
    DIR9.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
        except Exception:
            old = None
        if old and R._pid_alive(old):
            print(f"another 9ch instance alive (pid {old}) -> exit", flush=True)
            sys.exit(0)
    LOCK.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: LOCK.exists() and LOCK.unlink())


def slice3(v, idx):
    maxs = R.VOL_MAXSLICE[v]
    idx = int(min(max(idx, 0), maxs))
    return np.array(Image.open(R.slice_path(v, idx)).convert("RGB"))  # (H,W,3) uint8 = [liver,custom,hist]


class Tumor25D9(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v, z = r["volume"], int(r["slice_idx"])
        stack = np.concatenate([slice3(v, z - 1), slice3(v, z), slice3(v, z + 1)], axis=-1)  # (H,W,9)
        t = torch.from_numpy(stack).permute(2, 0, 1).float() / 255.0                          # (9,H,W)
        t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        if self.train:
            if random.random() < 0.5:
                t = torch.flip(t, dims=[2])
            t = TF.rotate(t, random.uniform(-15, 15))
        t = (t - 0.5) / 0.5
        return t, torch.tensor(float(r["tumor_present"]))


def make_loaders():
    tr = pd.read_csv(R.TRAIN_CSV); va = pd.read_csv(R.VAL_CSV); te = pd.read_csv(R.TEST_CSV)
    s_tr, s_va, s_te = set(tr.volume), set(va.volume), set(te.volume)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "patient leak!"
    labels = tr.tumor_present.values.astype(int)
    cw = 1.0 / np.bincount(labels)
    sampler = WeightedRandomSampler(cw[labels], num_samples=len(labels), replacement=True)
    tl = DataLoader(Tumor25D9(tr, True), batch_size=R.BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    vl = DataLoader(Tumor25D9(va, False), batch_size=R.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    tel = DataLoader(Tumor25D9(te, False), batch_size=R.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    return tr, va, te, tl, vl, tel


def build9(bb):
    return timm.create_model(BACKBONES[bb], pretrained=True, num_classes=1, in_chans=9).to(R.DEVICE)


def run_one(bb):
    tag = f"2.5d9ch_{bb}"
    flag = FLAGS9 / f"{bb}.done"
    if flag.exists():
        print(f"[{tag}] flag present -> skip", flush=True); return
    print(f"\n{'='*56}\n[{tag}] START\n{'='*56}", flush=True)
    tr, va, te, tl, vl, tel = make_loaders()
    model = build9(bb)
    pos = float((tr.tumor_present == 1).sum()); neg = float((tr.tumor_present == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=R.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt = DIR9 / f"{bb}_checkpoint.pth"; best = DIR9 / f"{bb}_best.pth"
    start, best_f1, noimp, freeze_ep, patience = 0, 0.0, 0, 3, 6
    R.set_freeze(model, True)
    if ckpt.exists():
        c = torch.load(ckpt, map_location=R.DEVICE, weights_only=False)
        model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optim"])
        scheduler.load_state_dict(c["sched"]); start = c["epoch"] + 1
        best_f1 = c["best_val_f1"]; noimp = c.get("no_improve", 0)
        if start >= freeze_ep:
            R.set_freeze(model, False)
        print(f"[{tag}] resumed ep{start} best_val_f1={best_f1:.4f}", flush=True)

    from tqdm import tqdm
    for epoch in range(start, EPOCHS):
        if epoch == freeze_ep:
            R.set_freeze(model, False)
        model.train(); run = 0.0
        for imgs, lb in tqdm(tl, desc=f"[{tag}] ep{epoch+1}/{EPOCHS}", leave=False):
            imgs = imgs.to(R.DEVICE); lb = lb.to(R.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lb)
            loss.backward(); optimizer.step(); run += loss.item() * imgs.size(0)
        scheduler.step()
        yv, pv = R.eval_probs(model, vl)
        vf1 = f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)
        print(f"[{tag}] ep{epoch+1} loss={run/len(tl.dataset):.4f} val_f1={vf1:.4f}", flush=True)
        if vf1 > best_f1 + 1e-6:
            best_f1 = vf1; noimp = 0; torch.save(model.state_dict(), best)
        else:
            noimp += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_val_f1": best_f1, "no_improve": noimp}, ckpt)
        if noimp >= patience:
            print(f"[{tag}] early stop ep{epoch+1}", flush=True); break

    model.load_state_dict(torch.load(best, map_location=R.DEVICE, weights_only=False))
    yv, pv = R.eval_probs(model, vl)
    prec, rec, thrs = precision_recall_curve(yv, pv)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    thr = float(thrs[int(np.nanargmax(f1s))]) if len(thrs) else 0.5
    yt, pt = R.eval_probs(model, tel)
    sm = R.slice_metrics(yt, pt, thr)
    pm, pat = R.patient_metrics(te, pt, thr)
    print(f"[{tag}] SLICE AUC={sm['ROC-AUC']:.4f} F1={sm['F1']:.4f} | "
          f"PATIENT AUC={pm['ROC-AUC']:.4f} F1={pm['F1']:.4f}", flush=True)
    np.savez(OUT / f"{tag}_test_predictions.npz", y_true=yt.astype(np.int64), y_prob=pt.astype(np.float32),
             pat_true=pat.y_true.values.astype(np.int64), pat_score=pat.score.values.astype(np.float32))
    json.dump({"mode": "2.5d_9ch", "backbone": bb, "best_threshold": thr, "slice": sm, "patient": pm},
              open(OUT / f"{tag}_metrics.json", "w"), indent=2)
    flag.write_text("done")
    print(f"[{tag}] DONE", flush=True)
    del model
    if R.DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def summarize():
    rows = []
    for bb in ORDER:
        for tag, rep in [(f"2d_{bb}", "2D (3win x1slice)"),
                         (f"2.5d_{bb}", "2.5D (1win x3slice)"),
                         (f"2.5d9ch_{bb}", "2.5D-9ch (3win x3slice)")]:
            f = OUT / f"{tag}_metrics.json"
            if f.exists():
                m = json.load(open(f))
                rows.append({"backbone": bb, "representation": rep,
                             "slice_AUC": m["slice"]["ROC-AUC"], "slice_F1": m["slice"]["F1"],
                             "patient_AUC": m["patient"]["ROC-AUC"], "patient_F1": m["patient"]["F1"]})
    if rows:
        df = pd.DataFrame(rows)
        (OUT / "comparison").mkdir(exist_ok=True)
        df.to_csv(OUT / "comparison" / "comparison_with_9ch.csv", index=False)
        print("\n=== 2D vs 2.5D vs 2.5D-9ch ===", flush=True)
        print(df.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    FLAGS9.mkdir(parents=True, exist_ok=True)
    acquire_lock()
    R.VOL_MAXSLICE = R.build_volume_maxslice()
    for bb in ([args.only] if args.only else ORDER):
        run_one(bb)
        summarize()
    print("\n9CH 2.5D COMPLETE", flush=True)


if __name__ == "__main__":
    main()
