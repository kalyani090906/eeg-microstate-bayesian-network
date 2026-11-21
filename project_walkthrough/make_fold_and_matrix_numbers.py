"""
make_fold_and_matrix_numbers.py - two sets of numbers for the project walkthrough (asked by the faculty).

  1. The train/test split fold by fold for the main model (L1 selection + logistic regression,
     maps refitted in each fold): who is tested, how many HC / SC are in training, the training
     accuracy, which features L1 kept, P(patient), the prediction. Recomputed from
     MLproject/outputs/nested_loso/fold_features_16.csv with exactly the project's model, and
     checked against the saved predictions (nested_loso_per_subject.csv).
  2. How the 19 x 231,250 matrix of one subject (h01) shrinks, step by step, with real counts,
     plus a picture of the matrix.

Writes: figures/data_matrix.png, fold_by_fold_with_s14.csv, fold_by_fold_without_s14.csv,
        matrix_numbers_h01.csv

Run (from MLproject):  eeg_env\\Scripts\\python.exe project_walkthrough\\make_fold_and_matrix_numbers.py
"""

# ---------------- settings (change only here) ----------------
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                         # the MLproject folder
OUT = PROJECT / "outputs"
SUBJECT = "h01.edf"
SEED = 42
# --------------------------------------------------------------

sys.path.insert(0, str(PROJECT / "scripts"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import eeg_preprocessing as pre
import microstate_features as mf
import microstate_model as mm

NAMES = {"state0": "A", "state1": "D", "state2": "B", "state3": "C"}


def short(col):
    # state3_duration -> C dur, trans_0to2 -> A->B
    if col.startswith("trans_"):
        i, j = col[6], col[9]
        return f"{NAMES['state' + i]}->{NAMES['state' + j]}"
    s, kind = col.split("_")
    return f"{NAMES[s]} {'dur' if kind == 'duration' else 'occ'}"


# Part 1: fold by fold, main model
features = [ln.strip() for ln in (OUT / "features_final_16_list.txt").read_text().splitlines() if ln.strip()]
folds = pd.read_csv(OUT / "nested_loso" / "fold_features_16.csv")
saved = pd.read_csv(OUT / "nested_loso" / "nested_loso_per_subject.csv")
saved = saved[(saved.model == "L1 selection + logistic regression") & (saved.maps == "maps refitted in each fold")]

for run in ["with s14", "without s14"]:
    rows = []
    part = folds[folds.run == run]
    for t, tab in part.groupby("test_subject", sort=True):
        # Step 1: split - the test subject's row vs the other subjects' rows (all from fold t's maps)
        train, test = tab[tab.subject != t], tab[tab.subject == t]
        # Step 2: the same model as nested_loso_refit_maps.py: scaler -> L1 selection -> logistic regression
        model = make_pipeline(StandardScaler(), Pipeline([
            ("select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=SEED))),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]))
        model.fit(train[features].values, train["label"].values)
        kept = [short(f) for f, k in zip(features, model[-1].named_steps["select"].get_support()) if k]
        p_train = model.predict_proba(train[features].values)[:, 1]
        p = float(model.predict_proba(test[features].values)[0, 1])
        true = int(test["label"].iloc[0])
        rows.append({"test_subject": t, "true": "SC" if true else "HC",
                     "train_HC": int((train.label == 0).sum()), "train_SC": int((train.label == 1).sum()),
                     "train_accuracy": round(float(((p_train >= 0.5) == train.label.values).mean()), 3),
                     "features_kept": ", ".join(kept), "P_patient": round(p, 3),
                     "predicted": "SC" if p >= 0.5 else "HC", "correct": (p >= 0.5) == bool(true)})
    df = pd.DataFrame(rows)
    # Step 3: check against the saved predictions of the project
    ref = saved[saved.table == run].set_index("subject")["prob_patient"]
    diff = np.abs(df.set_index("test_subject")["P_patient"] - ref.loc[df.test_subject].values).max()
    print(f"{run}: {df.correct.sum()}/{len(df)} correct, mean train acc {df.train_accuracy.mean():.3f}, "
          f"max diff vs saved {diff:.4f}")
    df.to_csv(HERE / f"fold_by_fold_{run.replace(' ', '_')}.csv", index=False)

# Part 2: the matrix of one subject, step by step
model = mm.load_model()
raw = pre.load_raw(SUBJECT)
clean = pre.filter_and_reference(raw)
data = clean.get_data()                                         # 19 x samples
fs = clean.info["sfreq"]
epochs = pre.cut_epochs(clean)
keep = pre.kept_epochs(epochs)

from pycrostates.preprocessing import extract_gfp_peaks
peaks = extract_gfp_peaks(clean, min_peak_distance=2, verbose="ERROR").get_data()
gfp = peaks.std(axis=0)
after_cut = int((gfp <= np.percentile(gfp, 95)).sum())

n_segments, n_trans, labelled = [], 0, 0
for i in keep:
    _, lab = mm.label_epoch(model, epochs[i])
    segs = [s for s in mf.segments(lab) if s[0] != -1]
    n_segments.append(len(segs))
    labelled += int((lab != -1).sum())
    n_trans += int(mf.transition_counts(lab, 4).sum())

numbers = [
    ("raw recording", f"19 x {data.shape[1]:,}", f"{data.size:,}", f"{data.shape[1] / fs:.0f} s at 250 Hz"),
    ("after cleaning", f"19 x {data.shape[1]:,}", f"{data.size:,}", "same shape, values changed"),
    ("epochs (first 15 min)", f"{len(epochs)} x 19 x 1,250", f"{len(epochs) * 19 * 1250:,}", f"{len(epochs) * 1250:,} samples used"),
    ("kept epochs", f"{len(keep)} x 19 x 1,250", f"{len(keep) * 19 * 1250:,}", f"{len(epochs) - len(keep)} rejected"),
    ("GFP curve", f"1 x {data.shape[1]:,}", f"{data.shape[1]:,}", "19 numbers -> 1 per sample"),
    ("GFP peaks", f"{peaks.shape[1]:,} x 19", f"{peaks.size:,}", "whole recording"),
    ("after dropping loudest 5%", f"{after_cut:,} x 19", f"{after_cut * 19:,}", ""),
    ("random sample for fitting", "1,500 x 19", f"{1500 * 19:,}", "seed 42"),
    ("labels of kept epochs", f"{len(keep)} x 1,250", f"{len(keep) * 1250:,}", f"{labelled:,} labelled (edges -1)"),
    ("segments", f"{sum(n_segments):,}", "", f"mean {np.mean(n_segments):.1f} per epoch"),
    ("transitions counted", f"{n_trans:,}", "", "pooled for the 12 probabilities"),
    ("features per epoch", f"{len(keep)} x 24", f"{len(keep) * 24:,}", ""),
    ("features per person", "1 x 24 -> 1 x 16", "16", "mean over epochs; 8 redundant dropped"),
]
pd.DataFrame(numbers, columns=["step", "shape", "numbers", "note"]).to_csv(HERE / "matrix_numbers_h01.csv", index=False)
for r in numbers:
    print(" | ".join(r))

# Picture: the whole matrix, and 2 s of it
fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.45})
uv = data * 1e6
step = 50                                                        # every 50th sample, only for drawing
axes[0].imshow(uv[:, ::step], aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30,
               extent=[0, data.shape[1], 18.5, -0.5])
axes[0].set_yticks(range(19))
axes[0].set_yticklabels(clean.ch_names, fontsize=7)
axes[0].set_xlabel("sample number (0 ... 231,250); 250 samples = 1 second")
axes[0].set_title(f"h01 after cleaning: a matrix of 19 rows (electrodes) x {data.shape[1]:,} columns (samples)")
s0 = int(60 * fs)
im = axes[1].imshow(uv[:, s0:s0 + 500], aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30,
                    extent=[s0, s0 + 500, 18.5, -0.5])
col = s0 + 137
axes[1].axvline(col, color="black", lw=2)
axes[1].set_title("zoom: 2 seconds from 60 s in; the black line = one column = 19 numbers = one scalp map", fontsize=11)
axes[1].set_yticks(range(19))
axes[1].set_yticklabels(clean.ch_names, fontsize=8)
axes[1].set_xlabel("sample number (2 seconds = 500 samples shown)")
fig.colorbar(im, ax=axes, label="voltage (µV), red + / blue -", shrink=0.8)
fig.savefig(HERE / "figures" / "data_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved data_matrix.png")
