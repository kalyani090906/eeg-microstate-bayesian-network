"""
extract_epoch_features.py - from the cleaned EEG to the feature tables (the CSV files).

For every subject:
  1. clean the recording (eeg_preprocessing: channels, bad channels, 2-20 Hz, average reference)
  2. cut it into 5-second epochs, first 15 minutes, and reject abnormal epochs
  3. label every sample of every kept epoch with the shared 4-state model (microstate_model)
  4. compute the 24 features of each epoch with the formulas in microstate_features:
       duration, occurrence, coverage of each state (12) + 12 transition probabilities
  5. one row per subject: mean of the epochs (transitions from the pooled counts), and the SD

Saves (in outputs/):
  eeg_epoch_features_15min.csv        one row per kept EPOCH (subject, label, epoch, 24 features)
  eeg_features_epoch_mean_15min.csv   one row per SUBJECT (the 24-feature table)
  eeg_features_epoch_sd_15min.csv     one row per SUBJECT: how much each feature varies between epochs
  epoch_rejection_report_15min.csv    how many epochs each subject kept

IMPORTANT for anyone using the epoch-level table: rows from the same subject are NOT
independent. Always split / resample by the `subject` column, never by row.

Run:  eeg_env\\Scripts\\python.exe scripts\\extract_epoch_features.py      (about 15 minutes)
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
SUFFIX = "_15min"              # the analysis window (first 15 minutes, see eeg_preprocessing)
MIN_KEPT_FRACTION = 0.6        # warn if a subject keeps fewer epochs than this
# --------------------------------------------------------------

import time

import numpy as np
import pandas as pd

import eeg_preprocessing as pre
import microstate_features as mf
import microstate_model as mm

# Step 0: the shared model (fitted once by tune_microstates_v3.py) and the 24 column names
model = mm.load_model()
K = model.cluster_centers_.shape[0]
names = mf.feature_names(K)
print(f"Loaded shared model with {K} states.")

epoch_rows, mean_rows, sd_rows, report_rows = [], [], [], []
for f in pre.recording_files():
    t0 = time.perf_counter()
    subject, label = pre.subject_and_label(f)

    # Steps 1-2: clean, cut into epochs, keep the normal ones
    raw = pre.load_clean(f)
    epochs = pre.cut_epochs(raw)
    keep = pre.kept_epochs(epochs)
    sfreq = raw.info["sfreq"]

    # Steps 3-4: label each kept epoch and compute its 24 features
    rows, pooled_counts = [], np.zeros((K, K))
    for i in keep:
        _, labels = mm.label_epoch(model, epochs[i])
        row, counts = mf.epoch_features(labels, sfreq, K)
        pooled_counts += counts
        rows.append({"subject": subject, "label": label, "epoch": int(i), **row})
    epoch_rows += rows

    # Step 5: one row per subject (mean, and SD between epochs)
    mean_rows.append({"subject": subject, "label": label, **mf.subject_features(rows, pooled_counts, K, "mean")})
    sd_rows.append({"subject": subject, "label": label, **mf.subject_features(rows, pooled_counts, K, "sd")})

    fraction = len(keep) / len(epochs)
    report_rows.append({"subject": subject, "epochs_total": len(epochs), "epochs_kept": len(keep),
                        "fraction_kept": round(fraction, 3)})
    flag = "   <-- many epochs rejected, check this recording" if fraction < MIN_KEPT_FRACTION else ""
    print(f"{subject}: kept {len(keep)}/{len(epochs)} epochs  ({time.perf_counter() - t0:.0f}s){flag}",
          flush=True)

# Step 6: save the four tables
files = {f"eeg_epoch_features{SUFFIX}.csv": pd.DataFrame(epoch_rows),
         f"eeg_features_epoch_mean{SUFFIX}.csv": pd.DataFrame(mean_rows)[["subject", "label"] + names],
         f"eeg_features_epoch_sd{SUFFIX}.csv": pd.DataFrame(sd_rows)[["subject", "label"] + names],
         f"epoch_rejection_report{SUFFIX}.csv": pd.DataFrame(report_rows)}
for name, table in files.items():
    table.to_csv(OUTPUT_DIR / name, index=False)
print(f"\nEpoch-level rows: {len(epoch_rows)}")
print("Saved in outputs/: " + ", ".join(files))
