"""Saliency figure for LiMT slices (zero-shot Swin-2.5D model).

Same body-outline masking as cam_figure_lits.py.

    
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cam_swin as CS
from run_all_experiments import _base_tf, DEVICE
import config

os.makedirs(config.FIG_DIR, exist_ok=True)
OUT = str(config.FIG_DIR / "cam_limt.png")
LIMT_IMG = str(config.WORK_DIR / "limt_images")
LIMT_MASK = str(config.LIMT_DIR / "raw_seg_binary")
META = str(config.WORK_DIR / "limt_metadata_slices.csv")
TF = _base_tf(False)

meta = pd.read_csv(META)
vol_max = meta.groupby("volume")["slice_idx"].max().to_dict()
_seg = {}


def seg_of(v):
    if v not in _seg:
        _seg[v] = np.asanyarray(nib.load(os.path.join(LIMT_MASK, f"{v}.nii.gz")).dataobj)
    return _seg[v]


def load_25d(v, z):
    def chan0(idx):
        idx = int(min(max(idx, 0), vol_max[v]))
        return np.array(Image.open(os.path.join(LIMT_IMG, f"{v}_slice_{idx:03d}.jpg")).convert("RGB"))[:, :, 0]
    stack = np.stack([chan0(z - 1), chan0(z), chan0(z + 1)], axis=-1)
    return TF(Image.fromarray(stack)).unsqueeze(0)


def mask224(v, z):
    seg = seg_of(v); z = int(min(z, seg.shape[2] - 1))
    m = (seg[:, :, z] == 2).astype(np.float32)
    return F.interpolate(torch.from_numpy(m)[None, None], size=(224, 224), mode="nearest")[0, 0].numpy()


def body_mask(ct, thr=0.08, dilate=6):
    """Body silhouette from the normalised CT: threshold, fill holes, keep the
    largest connected component, dilate slightly so lesions at the liver edge
    are not clipped. Returns a boolean (224,224) mask."""
    b = ct > thr
    b = ndi.binary_fill_holes(b)
    lab, n = ndi.label(b)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        b = lab == (int(np.argmax(sizes)) + 1)
    b = ndi.binary_dilation(b, iterations=dilate)
    return b


# choose 4 clear-lesion slices from distinct volumes (largest in-plane lesion area)
pos = meta[meta.tumor_present == 1]
areas = []
for v in pos.volume.unique():
    sub = pos[pos.volume == v].slice_idx.values
    best = max(sub, key=lambda z: mask224(v, int(z)).sum())
    areas.append((float(mask224(v, int(best)).sum()), v, int(best)))
areas.sort(reverse=True)
picks = areas[:4]

model = CS.load_model(0).to(DEVICE)   # zero-shot LiTS-trained Swin-2.5D (2.5d_swin_best)
cam = CS.SwinGradCAM(model)

fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.7 * len(picks)))
if len(picks) == 1:
    axes = axes[None, :]
for r, (a, v, z) in enumerate(picks):
    x = load_25d(v, z).to(DEVICE)
    heat, prob = cam(x)
    sc = CS.score_cam(model, x, cam.act)
    ct = x[0, 1].detach().cpu().numpy(); ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
    mt = mask224(v, z)
    bm = body_mask(ct)                       # <-- corner-artifact fix
    heat_m = np.ma.masked_where(~bm, heat)   # transparent outside body
    sc_m = np.ma.masked_where(~bm, sc)
    for c, (ov, title) in enumerate([(None, f"{v}  z={z}\nCT (p={prob:.2f})"),
                                     (heat_m, "Grad-CAM"), (sc_m, "Score-CAM"), (None, "tumour GT")]):
        ax = axes[r, c]; ax.imshow(ct, cmap="gray")
        if ov is not None:
            ax.imshow(ov, cmap="jet", alpha=0.45)
        if c == 3:
            ax.contour(mt, levels=[0.5], colors="lime", linewidths=1.3)
        ax.set_title(title, fontsize=8); ax.axis("off")
cam.close()
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", OUT)
print("picks (area, vol, z):", [(round(a), v, z) for a, v, z in picks])
