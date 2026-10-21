"""
extract_epoch_features.py   (PROJECT script - put it in MLproject/scripts/)

Step 1 of the dataset improvement.

What it does
  1. Loads the shared k=4 model you already fitted (shared_microstate_model_v3.pkl).
  2. For each subject: same preprocessing as before (rename channels, interpolate
     the known bad channels, 2-20 Hz filter, average reference).
  3. Cuts the recording into 5-second epochs and REJECTS epochs whose variance is
     abnormal for that subject (blinks, movement, flat stretches).
  4. Labels every kept epoch with the shared model and computes the same features
     as before (duration / occurrence / coverage per state + transition probabilities).
  5. Saves:
       outputs/eeg_epoch_features.csv       one row per kept EPOCH (has subject + label)
       outputs/eeg_features_epoch_mean.csv  one row per SUBJECT, same columns as the old table
       outputs/eeg_features_epoch_sd.csv    one row per SUBJECT, how much each feature varies
                                            from epoch to epoch within that person
       outputs/epoch_rejection_report.csv   how many epochs were kept for each subject

IMPORTANT for anyone using the epoch-level table: rows from the same subject are NOT
independent. Always split / resample by the `subject` column, never by row.

Run:  python extract_epoch_features.py
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
import mne

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_FILE = "shared_microstate_model_v3.pkl"

# ---- settings (same as the tuned pipeline) ----
LO, HI = 2, 20                 # band-pass filter (Hz)
EPOCH_SEC = 5.0                # epoch length, as in the two reference papers
REJECT_Z = 3.5                 # reject epochs with |robust z-score of log-variance| above this
FACTOR, HALF_WINDOW = 5, 5     # light smoothing chosen in the tuning round
MIN_KEPT_FRACTION = 0.6        # warn if a subject keeps fewer epochs than this
MAX_EPOCHS = 180               # analysis window = first 180 epochs (15 min); None = use everything
OUT_SUFFIX = "_15min" if MAX_EPOCHS else ""   # so the full-length files are not overwritten

RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BADS = {"s07.edf": ["C4"], "h14.edf": ["F7"], "s11.edf": ["F7"], "s12.edf": ["P7"]}

with open(os.path.join(OUTPUT_DIR, MODEL_FILE), "rb") as fh:
    model = pickle.load(fh)
K = model.cluster_centers_.shape[0]
print(f"Loaded shared model with {K} states.")

feat_cols = []
for s in range(K):
    feat_cols += [f"state{s}_duration", f"state{s}_occurrence", f"state{s}_coverage"]
trans_cols = [f"trans_{i}to{j}" for i in range(K) for j in range(K) if i != j]
all_cols = feat_cols + trans_cols

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".edf"))
epoch_rows, subj_mean_rows, subj_sd_rows, report_rows = [], [], [], []
printed_keys = False

for f in files:
    t0 = time.perf_counter()
    subj = f.replace(".edf", "")
    label = 1 if subj.startswith("s") else 0

    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, f), preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if f in BADS:
        raw.info["bads"] = BADS[f]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.filter(LO, HI, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")

    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_SEC, preload=True, verbose="ERROR")
    if MAX_EPOCHS:
        epochs = epochs[:MAX_EPOCHS]      # same time window for everybody (first 15 min)
    n_total = len(epochs)

    # variance-based rejection (per subject, robust to a few huge outliers)
    logvar = np.log(epochs.get_data().var(axis=(1, 2)) + 1e-30)
    med = np.median(logvar)
    mad = np.median(np.abs(logvar - med)) + 1e-12
    z = 0.6745 * (logvar - med) / mad
    keep_idx = np.where(np.abs(z) <= REJECT_Z)[0]
    n_kept = len(keep_idx)

    count_sum = np.zeros((K, K))
    subj_epoch_rows = []
    for i in keep_idx:
        seg = model.predict(epochs[i], factor=FACTOR, half_window_size=HALF_WINDOW, verbose="ERROR")
        par = seg.compute_parameters()
        if not printed_keys:
            print("compute_parameters keys (first epoch):", sorted(par.keys()))
            printed_keys = True
        counts = np.asarray(seg.compute_transition_matrix(stat="count"), dtype=float)
        count_sum += counts

        row = {"subject": subj, "label": label, "epoch": int(i)}
        for s in range(K):
            occ = float(par.get(f"{s}_occurrences", np.nan))
            dur = float(par.get(f"{s}_meandurs", np.nan))
            row[f"state{s}_duration"] = dur if occ > 0 else np.nan   # no occurrence -> no duration
            row[f"state{s}_occurrence"] = occ
            row[f"state{s}_coverage"] = float(par.get(f"{s}_timecov", np.nan))
        for a in range(K):
            tot = counts[a].sum()
            for b in range(K):
                if a != b:
                    row[f"trans_{a}to{b}"] = counts[a, b] / tot if tot > 0 else np.nan
        subj_epoch_rows.append(row)

    epoch_rows += subj_epoch_rows
    ed = pd.DataFrame(subj_epoch_rows)

    mean_row = {"subject": subj, "label": label}
    sd_row = {"subject": subj, "label": label}
    for c in feat_cols:
        mean_row[c] = float(np.nanmean(ed[c]))
        sd_row[c] = float(np.nanstd(ed[c]))
    for a in range(K):               # subject-level transitions from POOLED counts (more accurate
        tot = count_sum[a].sum()     # than averaging many small per-epoch probabilities)
        for b in range(K):
            if a != b:
                c = f"trans_{a}to{b}"
                mean_row[c] = count_sum[a, b] / tot if tot > 0 else np.nan
                sd_row[c] = float(np.nanstd(ed[c]))
    subj_mean_rows.append(mean_row)
    subj_sd_rows.append(sd_row)

    report_rows.append({"subject": subj, "epochs_total": n_total, "epochs_kept": n_kept,
                        "fraction_kept": round(n_kept / n_total, 3)})
    flag = "   <-- many epochs rejected, check this recording" if n_kept / n_total < MIN_KEPT_FRACTION else ""
    print(f"{subj}: kept {n_kept}/{n_total} epochs  ({time.perf_counter() - t0:.0f}s){flag}", flush=True)

names = [f"eeg_epoch_features{OUT_SUFFIX}.csv", f"eeg_features_epoch_mean{OUT_SUFFIX}.csv",
         f"eeg_features_epoch_sd{OUT_SUFFIX}.csv", f"epoch_rejection_report{OUT_SUFFIX}.csv"]
pd.DataFrame(epoch_rows).to_csv(os.path.join(OUTPUT_DIR, names[0]), index=False)
pd.DataFrame(subj_mean_rows)[["subject", "label"] + all_cols].to_csv(
    os.path.join(OUTPUT_DIR, names[1]), index=False)
pd.DataFrame(subj_sd_rows)[["subject", "label"] + all_cols].to_csv(
    os.path.join(OUTPUT_DIR, names[2]), index=False)
pd.DataFrame(report_rows).to_csv(os.path.join(OUTPUT_DIR, names[3]), index=False)

print(f"\nEpoch-level rows: {len(epoch_rows)}")
print("Saved in outputs/: " + ", ".join(names))