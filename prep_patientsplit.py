"""Extract LiTS slices as 3-channel windowed JPEGs and build the patient-wise split.

Each axial slice is encoded as liver window / custom window / histogram-scaled
intensity. A slice is labelled positive if its mask contains the tumour label (2).
Volumes are split with StratifiedGroupKFold so that no patient appears in more
than one of train / val / test; the fold assignment is saved to volume_folds.csv
and reused by every other script.

    
"""
import os
import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import config

# ---------------- paths ----------------
VOLUME_DIRS = [str(d) for d in config.LITS_VOLUME_DIRS]
MASK_DIR = str(config.LITS_MASK_DIR)

OUT_DIR = config.WORK_DIR
IMG_DIR = OUT_DIR / "images"
AUG_IMG_DIR = OUT_DIR / "images_augmented"
CHECKPOINT_FILE = OUT_DIR / "slice_checkpoint.pkl"
CSV_FILE = OUT_DIR / "metadata_slices.csv"
TRAIN_CSV = OUT_DIR / "train_slices.csv"
VAL_CSV = OUT_DIR / "val_slices.csv"
TEST_CSV = OUT_DIR / "test_slices.csv"
TRAIN_BAL_CSV = OUT_DIR / "train_slices_balanced.csv"
FOLDS_CSV = OUT_DIR / "volume_folds.csv"   # volume -> outer fold id (for optional CV)

SEED = 42
N_SPLITS = 5   # 5-fold -> ~20% test, then ~20% of remainder as val
dicom_windows = {"liver": (150, 30), "custom": (400, 50)}


# ---------------- HU windowing ----------------
def windowed(img, width, level):
    lo, hi = level - width / 2, level + width / 2
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo)


def freqhist_bins(x, n_bins=100):
    imsd = np.sort(x.flatten())
    t = np.concatenate([[0.001], np.arange(n_bins) / n_bins + (1 / (2 * n_bins)), [0.999]])
    t = (len(imsd) * t).astype(int)
    return np.unique(imsd[t])


def hist_scaled(x, brks=None):
    if brks is None:
        brks = freqhist_bins(x)
    ys = np.linspace(0., 1., len(brks))
    return np.interp(x.flatten(), brks, ys).reshape(x.shape).clip(0., 1.)


def to_nchan(x, wins):
    chans = [windowed(x, w, l) for (w, l) in wins]
    chans.append(hist_scaled(x))
    return (np.stack(chans, axis=-1) * 255).astype(np.uint8)


def save_jpg(img, path):
    Image.fromarray(img).save(path, quality=95)


# ---------------- 1. slice extraction (checkpointed) ----------------
def extract_slices(limit=None):
    os.makedirs(IMG_DIR, exist_ok=True)
    volume_files = []
    for d in VOLUME_DIRS:
        volume_files.extend(sorted(Path(d).glob("volume-*.nii")))
    volume_files = sorted(volume_files, key=lambda p: int(p.stem.split("-")[1]))
    if limit:
        volume_files = volume_files[:limit]
    print(f"Found {len(volume_files)} volumes to consider", flush=True)

    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "rb") as f:
            ck = pickle.load(f)
        processed, records = ck["processed_volumes"], ck["records"]
        print(f"Resuming: {len(processed)} volumes already processed", flush=True)
    else:
        processed, records = set(), []

    for vol_path in tqdm(volume_files, desc="Extract"):
        vol_name = vol_path.stem
        if vol_name in processed:
            continue
        vol_id = vol_name.split("-")[1]
        mask_path = Path(MASK_DIR) / f"segmentation-{vol_id}.nii"
        if not mask_path.exists():
            print(f"  missing mask for {vol_name}, skip", flush=True)
            continue

        img = nib.load(str(vol_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()
        if img.shape != mask.shape:
            if img.shape[::-1] == mask.shape:
                img = np.transpose(img, (2, 1, 0))
            else:
                print(f"  shape mismatch {vol_name}: {img.shape} vs {mask.shape}, skip", flush=True)
                continue

        for s in range(img.shape[2]):
            tumor = int(2 in np.unique(mask[:, :, s]))
            rgb = to_nchan(img[:, :, s], [dicom_windows["liver"], dicom_windows["custom"]])
            out_name = f"{vol_name}_slice_{s:03d}.jpg"
            save_jpg(rgb, IMG_DIR / out_name)
            records.append({"volume": vol_name, "slice_idx": s,
                            "image_path": str(IMG_DIR / out_name), "tumor_present": tumor})

        processed.add(vol_name)
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump({"processed_volumes": processed, "records": records}, f)

    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE, index=False)
    print(f"\nExtracted {len(df)} slices from {df.volume.nunique()} volumes", flush=True)
    print("slice tumor distribution:\n", df.tumor_present.value_counts(), flush=True)
    return df


# ---------------- 2. patient-wise split ----------------
def make_patient_split(df, force=False):
    if not force and TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists():
        print("split CSVs already exist -> loading", flush=True)
        return (pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV))

    # volume-level label for stratification = does the volume contain ANY tumor slice
    vol_lab = df.groupby("volume")["tumor_present"].max()
    print(f"volume-level: {int((vol_lab==1).sum())} tumor / {int((vol_lab==0).sum())} normal "
          f"of {len(vol_lab)} volumes", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, df["tumor_present"], groups=df["volume"]))

    # record each volume's outer fold id (for optional k-fold CV later)
    fold_of_vol = {}
    for fid, (_, te_idx) in enumerate(folds):
        for v in df.iloc[te_idx]["volume"].unique():
            fold_of_vol[v] = fid
    pd.Series(fold_of_vol, name="fold").rename_axis("volume").reset_index().to_csv(FOLDS_CSV, index=False)

    # outer: fold 0 -> test
    trv_idx, test_idx = folds[0]
    test_df = df.iloc[test_idx].reset_index(drop=True)
    trv = df.iloc[trv_idx].reset_index(drop=True)

    # inner: carve val out of remaining, still grouped by volume
    inner = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(inner.split(trv, trv["tumor_present"], groups=trv["volume"]))
    train_df = trv.iloc[tr_idx].reset_index(drop=True)
    val_df = trv.iloc[val_idx].reset_index(drop=True)

    # ---- the guarantee: no patient in more than one split ----
    s_tr, s_va, s_te = set(train_df.volume), set(val_df.volume), set(test_df.volume)
    assert not (s_tr & s_va), f"train/val leak: {s_tr & s_va}"
    assert not (s_tr & s_te), f"train/test leak: {s_tr & s_te}"
    assert not (s_va & s_te), f"val/test leak: {s_va & s_te}"

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    def desc(name, d):
        pos = int((d.tumor_present == 1).sum())
        frac = f"{pos/len(d):.1%}" if len(d) else "n/a"
        print(f"  {name:5s}: {len(d):6d} slices | {d.volume.nunique():3d} volumes | "
              f"{pos:5d} tumor ({frac})", flush=True)

    print("PATIENT-WISE SPLIT (disjoint volumes verified):", flush=True)
    desc("train", train_df); desc("val", val_df); desc("test", test_df)
    return train_df, val_df, test_df


# ---------------- 3. augmentation (train only, regenerated) ----------------
def augment_train(train_df, force=False):
    if not force and TRAIN_BAL_CSV.exists():
        print("balanced train CSV exists -> loading", flush=True)
        return pd.read_csv(TRAIN_BAL_CSV)

    import albumentations as A
    # dropped VerticalFlip / RandomRotate90 (anatomically implausible on axial CT)
    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=0.5),
    ])

    if AUG_IMG_DIR.exists():
        shutil.rmtree(AUG_IMG_DIR)
    AUG_IMG_DIR.mkdir(parents=True)

    tumor_train = train_df[train_df["tumor_present"] == 1]
    print(f"augmenting {len(tumor_train)} train tumor slices x3", flush=True)
    aug_records = []
    for _, row in tqdm(tumor_train.iterrows(), total=len(tumor_train), desc="Augment"):
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        for i in range(3):
            aug_img = augmenter(image=img)["image"]
            aug_name = Path(row["image_path"]).stem + f"_aug{i}.jpg"
            aug_path = AUG_IMG_DIR / aug_name
            Image.fromarray(aug_img).save(aug_path, quality=95)
            aug_records.append({"volume": row["volume"], "slice_idx": row["slice_idx"],
                                "image_path": str(aug_path), "tumor_present": 1})

    train_bal = pd.concat([train_df, pd.DataFrame(aug_records)], ignore_index=True)
    train_bal.to_csv(TRAIN_BAL_CSV, index=False)
    print(f"balanced train: {len(train_bal)} slices, tumor frac "
          f"{(train_bal.tumor_present==1).mean():.3f}", flush=True)
    return train_bal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N volumes (smoke)")
    ap.add_argument("--force-split", action="store_true", help="recompute split+aug")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if CSV_FILE.exists() and not args.limit:
        print("metadata_slices.csv exists -> loading (skip extraction)", flush=True)
        df = pd.read_csv(CSV_FILE)
    else:
        df = extract_slices(limit=args.limit)

    train_df, val_df, test_df = make_patient_split(df, force=args.force_split)
    augment_train(train_df, force=args.force_split)
    print("\nPREP DONE", flush=True)


if __name__ == "__main__":
    main()
