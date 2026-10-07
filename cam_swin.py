"""Grad-CAM and Score-CAM for the Swin-2.5D model, with a pointing-game / IoU
evaluation against the LiTS tumour masks.

Each fold's checkpoint is applied only to that fold's held-out patients.

   
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score  # noqa: F401 (kept for parity)

from run_all_experiments import build_model, _base_tf, DEVICE  # noqa: E402
import config

BTC = str(config.WORK_DIR)
IMG_DIR = os.path.join(BTC, "images")
META = os.path.join(BTC, "metadata_slices.csv")
FOLDS = os.path.join(BTC, "volume_folds.csv")
RAW_MASK_DIR = str(config.LITS_MASK_DIR)
OUT_DIR = os.path.join(BTC, "explainability")

CKPT = {0: os.path.join(BTC, "2.5d_swin_best.pth")}
for k in (1, 2, 3, 4):
    CKPT[k] = os.path.join(BTC, "cv_best", f"fold{k}_best.pth")

TF = _base_tf(False)  # Resize(224)+ToTensor+Normalize([0.5]*3,[0.5]*3)


# -- 2.5D input: channel-0 (liver window) of 3 adjacent slices, mirrors Tumor25D --
def load_25d(volume, z, vol_max):
    def chan0(idx):
        idx = int(min(max(idx, 0), vol_max[volume]))
        p = os.path.join(IMG_DIR, f"{volume}_slice_{idx:03d}.jpg")
        return np.array(Image.open(p).convert("RGB"))[:, :, 0]
    stack = np.stack([chan0(z - 1), chan0(z), chan0(z + 1)], axis=-1)
    return TF(Image.fromarray(stack)).unsqueeze(0)  # (1,3,224,224)


def reshape_bhwc(t):
    """(B,7,7,C) channels-last -> (B,C,7,7)."""
    return t.permute(0, 3, 1, 2).contiguous()


class SwinGradCAM:
    def __init__(self, model):
        self.model = model
        self.act = None
        self.grad = None
        self.h = model.layers[-1].register_forward_hook(self._fwd)

    def _fwd(self, mod, inp, out):
        self.act = out
        if out.requires_grad:                    # skip under no_grad (Score-CAM passes)
            out.register_hook(self._save_grad)

    def _save_grad(self, g):
        self.grad = g

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze()          # single logit (target = tumor)
        logit.backward()
        a = reshape_bhwc(self.act)               # (1,C,7,7)
        g = reshape_bhwc(self.grad)              # (1,C,7,7)
        w = g.mean(dim=(2, 3), keepdim=True)     # (1,C,1,1)
        cam = F.relu((w * a).sum(dim=1, keepdim=True))   # (1,1,7,7)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), float(torch.sigmoid(logit))

    def close(self):
        self.h.remove()


@torch.no_grad()
def score_cam(model, x, act_layer_out, batch=64):
    """Gradient-free Score-CAM from the last-stage activation maps."""
    a = reshape_bhwc(act_layer_out)[0]           # (C,7,7)
    C = a.shape[0]
    maps = F.interpolate(a.unsqueeze(1), size=(224, 224), mode="bilinear",
                         align_corners=False).squeeze(1)   # (C,224,224)
    mn = maps.amin(dim=(1, 2), keepdim=True)
    mx = maps.amax(dim=(1, 2), keepdim=True)
    maps = (maps - mn) / (mx - mn + 1e-8)
    scores = []
    for i in range(0, C, batch):
        chunk = maps[i:i + batch].unsqueeze(1)   # (b,1,224,224)
        masked = x * chunk                        # broadcast over 3 channels
        s = torch.sigmoid(model(masked).squeeze(-1))  # (b,)
        scores.append(s)
    w = torch.softmax(torch.cat(scores), dim=0).view(-1, 1, 1)
    cam = F.relu((w * maps).sum(dim=0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def load_model(fold):
    m = build_model("swin")
    m.load_state_dict(torch.load(CKPT[fold], map_location=DEVICE, weights_only=False))
    return m.eval()


def mask_slice_224(volume, z):
    seg = nib.load(os.path.join(RAW_MASK_DIR, f"segmentation-{volume.split('-')[1]}.nii"))
    arr = np.asanyarray(seg.dataobj)
    m = (arr[:, :, z] == 2).astype(np.float32)
    t = torch.from_numpy(m)[None, None]
    t = F.interpolate(t, size=(224, 224), mode="nearest")[0, 0]
    return t.numpy()


def pointing_game(limit_per_vol, cache_masks=True):
    meta = pd.read_csv(META)
    folds = pd.read_csv(FOLDS)
    fold_of = dict(zip(folds.volume, folds.fold))
    vol_max = meta.groupby("volume")["slice_idx"].max().to_dict()

   
    TOL_PX = 32   # 224/7 -> native CAM cell size
    strict = tol = liver_e = 0.0
    rnd_strict = rnd_tol = rnd_liver = 0.0
    n = 0
    per_vol_seg = {}
    rng = np.random.default_rng(42)
    results = []
    for fold in range(5):
        model = load_model(fold)
        cam = SwinGradCAM(model)
        vols = [v for v in folds.volume if fold_of[v] == fold]
        for v in vols:
            pos = meta[(meta.volume == v) & (meta.tumor_present == 1)]["slice_idx"].values
            if len(pos) == 0:
                continue
            if limit_per_vol and len(pos) > limit_per_vol:
                pos = rng.choice(pos, size=limit_per_vol, replace=False)
            if v not in per_vol_seg:
                seg = nib.load(os.path.join(RAW_MASK_DIR,
                               f"segmentation-{v.split('-')[1]}.nii"))
                per_vol_seg[v] = np.asanyarray(seg.dataobj)
            segv = per_vol_seg[v]
            for z in pos:
                x = load_25d(v, int(z), vol_max).to(DEVICE)
                heat, prob = cam(x)
                def to224(a):
                    return F.interpolate(torch.from_numpy(a.astype(np.float32))[None, None],
                                         size=(224, 224), mode="nearest")[0, 0].numpy()
                mt = to224(segv[:, :, int(z)] == 2) > 0.5          # tumor
                liver = to224(segv[:, :, int(z)] >= 1) > 0.5       # liver+tumor
                if mt.sum() == 0:
                    continue
                py, px = np.unravel_index(int(heat.argmax()), heat.shape)
                dist_to_tumor = distance_transform_edt(~mt)        # px distance field
                s_hit = mt[py, px]
                t_hit = dist_to_tumor[py, px] <= TOL_PX
                le = heat[liver].sum() / (heat.sum() + 1e-8)       # CAM energy in liver
                # matched random baselines
                r_strict = mt.mean()
                r_tol = (dist_to_tumor <= TOL_PX).mean()
                r_liver = liver.mean()
                strict += s_hit; tol += t_hit; liver_e += le
                rnd_strict += r_strict; rnd_tol += r_tol; rnd_liver += r_liver
                n += 1
                results.append(dict(volume=v, z=int(z), prob=prob, strict_hit=int(s_hit),
                                    tol_hit=int(t_hit), liver_energy=le,
                                    peak_dist=float(dist_to_tumor[py, px])))
        cam.close()
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  fold {fold}: n={n} strictPG={strict/max(1,n):.3f} "
              f"tolPG={tol/max(1,n):.3f} liverE={liver_e/max(1,n):.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    summ = {
        "n_slices": n, "limit_per_vol": limit_per_vol, "tol_px": TOL_PX,
        "strict_pg": strict / max(1, n), "strict_pg_random": rnd_strict / max(1, n),
        "tol_pg": tol / max(1, n), "tol_pg_random": rnd_tol / max(1, n),
        "cam_liver_energy": liver_e / max(1, n), "liver_area_frac_random": rnd_liver / max(1, n),
    }
    summ["strict_lift"] = summ["strict_pg"] / max(1e-9, summ["strict_pg_random"])
    summ["tol_lift"] = summ["tol_pg"] / max(1e-9, summ["tol_pg_random"])
    summ["liver_lift"] = summ["cam_liver_energy"] / max(1e-9, summ["liver_area_frac_random"])
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "pointing_game_per_slice.csv"),
                                 index=False)
    json.dump(summ, open(os.path.join(OUT_DIR, "pointing_game_summary.json"), "w"), indent=2)
    print(f"\n=== CAM LOCALIZATION (OOF, {n} tumor+ slices, tol={TOL_PX}px) ===", flush=True)
    print(f"  strict pointing game : {summ['strict_pg']:.3f}  (random {summ['strict_pg_random']:.4f}, "
          f"{summ['strict_lift']:.0f}x)", flush=True)
    print(f"  tolerance PG (@{TOL_PX}px): {summ['tol_pg']:.3f}  (random {summ['tol_pg_random']:.3f}, "
          f"{summ['tol_lift']:.1f}x)", flush=True)
    print(f"  CAM energy in liver  : {summ['cam_liver_energy']:.3f}  (liver area {summ['liver_area_frac_random']:.3f}, "
          f"{summ['liver_lift']:.1f}x)", flush=True)
    print(f"saved -> {OUT_DIR}", flush=True)
    return summ


def make_figure(n_examples=4):
    """Qualitative: CT | Grad-CAM | Score-CAM | tumor contour, for a few high-conf
    tumor slices drawn from different folds."""
    meta = pd.read_csv(META)
    folds = pd.read_csv(FOLDS)
    fold_of = dict(zip(folds.volume, folds.fold))
    vol_max = meta.groupby("volume")["slice_idx"].max().to_dict()

    picks = []
    rng = np.random.default_rng(7)
    for fold in range(5):
        vols = [v for v in folds.volume if fold_of[v] == fold]
        for v in vols:
            pos = meta[(meta.volume == v) & (meta.tumor_present == 1)]["slice_idx"].values
            if len(pos):
                picks.append((fold, v, int(rng.choice(pos))))
                break
        if len(picks) >= n_examples:
            break

    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.7 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for r, (fold, v, z) in enumerate(picks):
        model = load_model(fold)
        cam = SwinGradCAM(model)
        x = load_25d(v, z, vol_max).to(DEVICE)
        heat, prob = cam(x)
        sc = score_cam(model, x, cam.act)
        cam.close()
        ct = x[0, 1].detach().cpu().numpy()          # centre slice
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
        seg = nib.load(os.path.join(RAW_MASK_DIR, f"segmentation-{v.split('-')[1]}.nii"))
        mslice = (np.asanyarray(seg.dataobj)[:, :, z] == 2).astype(np.float32)
        mt = F.interpolate(torch.from_numpy(mslice)[None, None], size=(224, 224),
                           mode="nearest")[0, 0].numpy()
        for c, (img, ov, title) in enumerate([
                (ct, None, f"{v} z={z}\nCT (p={prob:.2f})"),
                (ct, heat, "Grad-CAM"),
                (ct, sc, "Score-CAM"),
                (ct, None, "tumor GT")]):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            if ov is not None:
                ax.imshow(ov, cmap="jet", alpha=0.45)
            if c == 3:
                ax.contour(mt, levels=[0.5], colors="lime", linewidths=1.2)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "cam_qualitative.png")
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"saved figure -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_per_vol", type=int, default=8,
                    help="max tumor+ slices sampled per volume for the pointing game (0=all)")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.limit_per_vol = 1
    pointing_game(args.limit_per_vol)
    if args.figure:
        make_figure()
    print("CAM DONE", flush=True)


if __name__ == "__main__":
    main()
