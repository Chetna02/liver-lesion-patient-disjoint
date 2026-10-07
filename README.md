# Patient-disjoint liver-lesion classification on CT

Code for the paper *"Patient-disjoint evaluation of CNN and transformer models for binary
liver-lesion classification on CT: correcting data leakage and testing cross-dataset
generalization"* (submitted to the Journal of Electronic Imaging).

Slices are classified as lesion-bearing or lesion-free with ResNet-18, ViT-B/16 and
Swin-B (timm, ImageNet weights), using 2D, 2.5D and nine-channel inputs. All
experiments on LiTS use a patient-wise split, so no patient contributes slices to
more than one of train / validation / test. The LiTS-trained models are then
evaluated on LiMT zero-shot and after fine-tuning.

## Data

Both datasets are public and are not redistributed here.

| Dataset | Use | Source |
|---|---|---|
| LiTS 2017 (131 training volumes) | training, CV, internal test | https://competitions.codalab.org/competitions/17094 |
| LiMT (150 volumes, 4 lesion types + normals) | external validation | see the LiMT paper cited in the manuscript |

Paths are set in `config.py` or through environment variables:

```
LITS_DIR       folder with volume_pt1..volume_pt7 (or volume-*.nii directly)
LITS_MASK_DIR  folder with segmentation-*.nii      (default: LITS_DIR/segmentations)
LIMT_DIR       LiMT root (images_out/, masks/, limt_labels.json)
WORK_DIR       all generated files                  (default: ./work)
```

## Setup

Tested with Python 3.9, PyTorch 2.4.1 (CUDA 12.1) on a single NVIDIA GPU.

```
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from https://pytorch.org first
if needed.

## Running

The scripts are meant to be run in this order; later steps read the outputs of
earlier ones from `WORK_DIR`.

**LiTS**

| Step | Script | Produces |
|---|---|---|
| 1 | `prep_patientsplit.py` | windowed JPEG slices, slice labels, patient-wise folds |
| 2 | `run_all_experiments.py` | 3 backbones x {2D, 2.5D} on the fold-0 split |
| 3 | `run_25d_9ch.py` | nine-channel input, same split |
| 4 | `run_cv_best.py` | 5-fold CV of the best configuration (Swin, 2.5D) |
| 5 | `leaky_split_experiment.py` | same pipeline with a slice-wise split (leakage control) |
| 6 | `imbalance_ablation.py` | weighted sampler / weighted loss ablation |
| 7 | `cam_swin.py`, `cam_figure_lits.py` | Grad-CAM / Score-CAM, pointing game, figure |
| 8 | `robustness_eval.py` | noise and resolution perturbations (also writes OOF predictions) |
| 9 | `error_analysis.py` | false positives / negatives by lesion size and case |

**LiMT**

| Step | Script | Produces |
|---|---|---|
| 10 | `limt_manifest.py` | case labels and harmonised masks |
| 11 | `limt_prep_jpeg.py` | slices encoded exactly as for LiTS |
| 12 | `limt_eval_transfer.py` (`--cv`) | zero-shot transfer of the LiTS models |
| 13 | `limt_ensemble_eval.py` | slice-level AUC of the 5-fold ensemble |
| 14 | `limt_finetune_seeded.py` | 5-fold fine-tuning on LiMT (calls `limt_finetune.py`) |
| 15 | `limt_ablation.py` | ImageNet-only and pooled LiTS+LiMT training |
| 16 | `limt_lesion_breakdown.py` | sensitivity per lesion type |

**Statistics**

`compute_stats_tests.py` runs the DeLong, McNemar and Wilcoxon tests on the saved
predictions once steps 2-4 and 14 are done.

`imbalance_ablation.py`, `limt_finetune.py`, `limt_ablation.py` and `cam_swin.py`


## Notes

- Fine-tuning results were produced with `limt_finetune_seeded.py`. Even with fixed
  seeds and deterministic cuDNN, small differences can appear on other GPUs or
  driver versions.
- `error_analysis.py` can additionally stratify errors by liver attenuation if
  `HU_CSV` points to a CSV with columns `dataset, case_id, hu_median`; without it
  that part is skipped.

## Citation

If you use this code, please cite the paper (details will be added on publication).

## Contact

Chetna Sinha, National Institute of Technology Raipur — csinha.phd2022.etc@nitrr.ac.in
