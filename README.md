# Bayesian network modelling of EEG microstate transitions for schizophrenia

Course project, 23CSE301 (Machine Learning). Resting-state EEG from 14 patients with
schizophrenia and 14 healthy controls is turned into a sequence of four recurring scalp
patterns (microstates A–D). Each person is then described by how long each state lasts, how
often it appears and how it switches to the next one. These features, and embeddings from a
pretrained EEG model, are used to classify patients vs controls. A Bayesian network of the
transitions is in progress.

**Status:** all seven first-evaluation items have a first result. See
[outputs/items/README.md](outputs/items/README.md) for where each result is. Still to do:
Bayesian network, bootstrap edge stability, group statistics, report.

> With 28 subjects every result here is **exploratory**. One subject is about 3.6 percentage
> points of accuracy. Results are reported with 95% confidence intervals and a permutation
> test, and no feature is claimed as a validated biomarker.

---

## Data source

RepOD: *EEG in schizophrenia*, Olejarczyk E. & Jernajczyk W., Institute of Psychiatry and
Neurology, Warsaw. Repository for Open Data, 2017.
DOI [10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441)

- 14 patients (`s01`–`s14`, label 1) and 14 healthy controls (`h01`–`h14`, label 0)
- 19 channels (10-20 system), 250 Hz, eyes closed, about 15 minutes each
- Paper: Olejarczyk & Jernajczyk (2017), *Graph-based analysis of brain connectivity in
  schizophrenia*, PLOS ONE 12(11): e0188629

The EDF files are **not** in this repository. Download them and put them in `dataset/`
(`h01.edf` … `h14.edf`, `s01.edf` … `s14.edf`).

---

## Setup (Windows, PowerShell, Python 3.13)

```powershell
cd path\to\MLproject
python -m venv eeg_env
eeg_env\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index URL is needed because `requirements.txt` pins the CPU build of PyTorch
(`torch==2.14.1+cpu`). No GPU is needed: every script here runs on a laptop CPU.

Run every script from the `MLproject` folder as `eeg_env\Scripts\python.exe scripts\<name>.py`.
Scripts find their files from their own location, so they also work when started from another
folder. Every script has its settings in one block at the top and uses seed 42.

---

## Folder guide

| Folder | What is in it |
|---|---|
| `dataset/` | the 28 EDF recordings (**not in git**, see Data source) |
| `scripts/` | the project pipeline and the seven items (run order below) |
| `labs/` | lab-sheet exercises (Labs 3–10) redone on the project data; `lab_common.py` holds shared settings |
| `outputs/` | the tables and figures the pipeline writes |
| `outputs/items/` | one file (or folder) per project item, with its own [README](outputs/items/README.md) |
| `outputs/labs/` | lab results (tables and figures) |
| `outputs/permutation/` | permutation-test results (is a result better than chance?) |
| `ml project dump/` | old scripts and superseded tables (**not in git**, kept locally only) |

**Large files not in git** (in `.gitignore`); share them through a drive, or rebuild them with
the script named:

| File | Size | Rebuilt by |
|---|---|---|
| `outputs/shared_microstate_model_v3.pkl` | 7 MB | `tune_microstates_v3.py` |
| `outputs/epochs_15min.npz` | 420 MB | `make_epoch_array.py` |
| `outputs/epochs_15min_labram.npz` | 340 MB | `make_epoch_array.py --preset labram` |
| `outputs/embeddings_labram.npz` | 4 MB | `item2_embeddings.py` |

The pretrained LaBraM weights (23 MB) download automatically from Hugging Face
(`braindecode/labram-pretrained`) the first time item 2 or 3 runs.

---

## Run order

You can skip part A: its outputs (the feature tables) are already in git. Run it only to
rebuild everything from the EDF files.

### A. From raw EEG to the feature tables

| # | Script | What it does | Writes |
|---|---|---|---|
| 1 | `check_bad_channels.py` | lists flat / noisy channels per subject (prints only) | — |
| 2 | `tune_microstates_v3.py` | fits ONE shared 4-state microstate model for all 28 subjects (no labels used) | `shared_microstate_model_v3.pkl`, `v3_maps_k4.png` |
| 3 | `name_states.py` | suggests which state is A, B, C, D by map shape (prints only) | — |
| 4 | `check_map_stability.py` | re-fits with other seeds / 4 subjects left out; maps stay stable (prints only) | — |
| 5 | `extract_epoch_features.py` | 5-s epochs, first 15 min, rejects outlier epochs, 24 features per epoch and per subject | `eeg_epoch_features_15min.csv`, `eeg_features_epoch_mean_15min.csv`, `..._sd_15min.csv`, `epoch_rejection_report_15min.csv` |
| 6 | `make_final_features.py` | drops 8 redundant columns WITHOUT using labels: 24 → 16 features | `features_final_16.csv`, `features_final_16_no_s14.csv`, `features_final_16_list.txt` |
| 7 | `plot_labelled_maps.py` | the four maps with their A–D names | `maps_labelled_ABCD.png` |
| 8 | `make_epoch_array.py` | the kept raw epochs as one array (same 4,786 epochs as step 5) | `epochs_15min.npz` |
| 9 | `make_epoch_array.py --preset labram` | the same epochs, 0.1–75 Hz, 50 Hz notch, 200 Hz, for LaBraM | `epochs_15min_labram.npz` |

### B. Evaluation and the seven items

| # | Script | Item | Time (CPU) |
|---|---|---|---|
| 10 | `evaluate.py` | the shared scoring function (leave-one-subject-out, 95% CI) + a logistic-regression baseline | seconds |
| 11 | `labs\lab09_a3_classifiers.py` | **1**: 11 classifiers on the 16 features | 3 min |
| 12 | `item2_embeddings.py` | **2**: frozen LaBraM embeddings + classifiers (needs step 9) | 2 min |
| 13 | `item3_finetune.py` | **3**: fine-tuned LaBraM, 7-fold subject-grouped (needs step 9; resumes if stopped) | about 1 hour |
| 14 | `item4_feature_reduction.py` | **4**: 24 vs 16 vs top-k vs L1 selection | 1 min |
| 15 | `labs\lab09_a4_pipeline.py`, `labs\lab09_a5_lime.py` | **5**: pipeline + LIME | 1 min |
| 16 | `item7_clustering.py` | **7**: clustering vs diagnosis | 1 min |
| 17 | `permutation_test.py` | 1,000 label shuffles for 4 pre-chosen models (needs step 12; resumes if stopped) | about 25 min |
| 18 | `item6_comparison.py` | **6**: comparison with published results (reads the results above) | seconds |
| 19 | `collect_items.py` | copies the item 1 and 5 results into `outputs/items/` | seconds |

### C. Lab exercises (`labs/`)

Each file is named `labNN_aX_….py` after the lab-sheet question it answers. For example,
`labs\lab05_a8_a9_k_sweep.py` is Lab 5 A8–A9. They all use the same 16-feature table and write
to `outputs/labs/`. The unit tests run as scripts:
`eeg_env\Scripts\python.exe labs\lab04_a2_unit_tests.py` and `labs\lab06_a2_unit_tests.py`.
Lab 8 A1–A11 (logic gates, customer table) are separate submissions and are not here.

---

## Rules every analysis follows

1. **Split by subject.** All epochs of one person stay on the same side of every split,
   bootstrap and permutation.
2. **Learn from the training part only.** Scaling, feature selection, PCA, binning and tuning
   are fitted inside each training fold, never on all subjects first.
3. **Leave-one-subject-out** (28 folds) unless stated otherwise. Epoch predictions are averaged
   into one prediction per subject, and epoch-level accuracy is never reported.
4. **Report every result twice:** with s14 (main) and without s14 (see below).
5. **No augmentation, no overlapping windows.**
6. Feature names come from `outputs/features_final_16_list.txt`. The epoch table has extra
   columns that are not features.

## Why s14 is reported separately

Subject `s14` has a 36-minute recording, and its signal changes abruptly at about 18 minutes:
the share of time in state C jumps by 0.415, against 0.191 for the next-largest jump (`s11`).
See `outputs/inspect_s14.png`. So that every subject is treated the same way, the **first 15
minutes** are used for everyone, and every result is also given **without s14** (27 subjects:
14 controls, 13 patients) to show how much it depends on that one person. This decision was
made without looking at the labels.

## Microstate names

The tables use state numbers. Their usual names are: **state0 = A, state2 = B, state3 = C,
state1 = D**. The names were matched by map shape; C and D are the least certain. The shared
model explains 67.4% of the signal (GEV 0.674).

---

## Main results so far (details in `outputs/items/`)

| Approach | With s14 | Without s14 | Better than chance? (permutation, corrected) |
|---|---|---|---|
| Microstate features, L1 selection + logistic regression | 78.6% | 81.5% | yes (p = 0.033 / 0.014) |
| Frozen LaBraM embeddings + logistic regression | 75.0% | 74.1% | borderline (p = 0.059 / 0.050) |
| Fine-tuned LaBraM (7-fold, different protocol) | 71.4% | 77.8% | not tested |
| Published, same dataset, subject-wise (Kim et al. 2021) | 75.64% | | |

Leave-one-subject-out accuracy unless stated. Higher published numbers on this dataset (90–99%)
come from splitting epochs instead of subjects, or from data augmentation. See
[item 6](outputs/items/item6_comparison.md).
