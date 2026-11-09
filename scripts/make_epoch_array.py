"""
make_epoch_array.py - the raw EEG of the kept epochs as one array, for the deep-learning items
(pretrained LaBraM embeddings, item 2, and fine-tuning, item 3).

extract_epoch_features.py keeps only the microstate FEATURES of each epoch. A pretrained EEG
model needs the epochs themselves, so this script saves them.

What it does (using eeg_preprocessing, so it is identical to the feature pipeline)
  1. Read each recording, rename channels, rebuild the bad channels.
  2. Decide WHICH epochs to keep exactly like extract_epoch_features.py: 5-second epochs, first
     180 (15 minutes), reject an epoch when |robust z-score of its log-variance| > 3.5.
     This decision always uses the 2-20 Hz signal, so the kept epochs are identical to the
     feature table even if the model needs another band.
  3. Save the kept epochs filtered with OUT_BAND (default 2-20 Hz, 250 Hz). A pretrained model may
     expect another band or sampling rate: use a preset (below).
  4. Check the result against outputs/eeg_epoch_features_15min.csv: the same (subject, epoch) pairs.

Output: outputs/epochs_15min.npz with
   X          float32 array (n_epochs, n_channels, n_samples), in volts as MNE gives them
   subject    text array, one per epoch (h01 ... s14)
   label      0 = healthy, 1 = schizophrenia
   epoch      epoch number inside the recording (the same number as in eeg_epoch_features_15min.csv)
   ch_names   the channel names, in the order of X
   sfreq      sampling rate of X
This file is large (about 420 MB at 250 Hz): share it through a drive. It is NOT for git.

Run:  eeg_env\\Scripts\\python.exe scripts\\make_epoch_array.py                   (2-20 Hz, 250 Hz)
      eeg_env\\Scripts\\python.exe scripts\\make_epoch_array.py --preset labram   (LaBraM input)
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
REF_TABLE = OUTPUT_DIR / "eeg_epoch_features_15min.csv"     # used only for the final check
OUT_FILE = "epochs_15min.npz"
OUT_BAND = (2, 20)             # band of the saved epochs; None = no extra filtering
OUT_SFREQ = None               # e.g. 200 to resample; None = keep 250 Hz
OUT_NOTCH = None               # e.g. 50 to remove mains hum (Poland: 50 Hz); None = no notch

# PRESETS: run "... make_epoch_array.py --preset labram" to use these instead
PRESETS = {
    # LaBraM was pretrained on 0.1-75 Hz, 50 Hz notch, 200 Hz (Jiang et al., ICLR 2024)
    "labram": {"OUT_BAND": (0.1, 75), "OUT_SFREQ": 200, "OUT_NOTCH": 50,
               "OUT_FILE": "epochs_15min_labram.npz"},
}
# --------------------------------------------------------------

import argparse

import numpy as np
import pandas as pd

import eeg_preprocessing as pre

parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=sorted(PRESETS), help="use a model's preset settings")
args = parser.parse_args()
if args.preset:
    globals().update(PRESETS[args.preset])
    print(f"Preset '{args.preset}': band {OUT_BAND} Hz, notch {OUT_NOTCH}, "
          f"{OUT_SFREQ} Hz -> outputs/{OUT_FILE}")

X_parts, subj, lab, ep_no = [], [], [], []
ch_names, sfreq_out = None, None
for f in pre.recording_files():
    subject, label = pre.subject_and_label(f)
    base = pre.load_raw(f)

    # Step 2: which epochs are kept - decided on the 2-20 Hz signal, as in the feature pipeline
    selection_epochs = pre.cut_epochs(pre.filter_and_reference(base, pre.BAND))
    keep = pre.kept_epochs(selection_epochs)

    # Step 3: the epochs to save - same boundaries, possibly another band / sampling rate
    out_epochs = pre.cut_epochs(pre.filter_and_reference(base, OUT_BAND, OUT_NOTCH))
    if OUT_SFREQ:
        out_epochs = out_epochs.resample(OUT_SFREQ, verbose="ERROR")
    X_parts.append(out_epochs.get_data()[keep].astype(np.float32))
    subj += [subject] * len(keep)
    lab += [label] * len(keep)
    ep_no += [int(i) for i in keep]
    ch_names = out_epochs.ch_names
    sfreq_out = float(out_epochs.info["sfreq"])
    print(f"{subject}: kept {len(keep)}/{len(selection_epochs)} epochs", flush=True)

X = np.concatenate(X_parts, axis=0)
np.savez_compressed(OUTPUT_DIR / OUT_FILE, X=X, subject=np.array(subj), label=np.array(lab),
                    epoch=np.array(ep_no), ch_names=np.array(ch_names), sfreq=sfreq_out)
print(f"\nSaved outputs/{OUT_FILE}: X shape {X.shape} (epochs, channels, samples), sfreq {sfreq_out} Hz")

# Step 4: the same (subject, epoch) pairs as the feature table?
if REF_TABLE.exists():
    ref = pd.read_csv(REF_TABLE)
    mine = set(zip(subj, ep_no))
    theirs = set(zip(ref["subject"], ref["epoch"].astype(int)))
    if mine == theirs:
        print(f"CHECK OK: the same {len(mine)} (subject, epoch) pairs as {REF_TABLE.name}")
    else:
        print(f"CHECK FAILED: {len(mine)} epochs here, {len(theirs)} in {REF_TABLE.name}; "
              f"{len(mine - theirs)} only here, {len(theirs - mine)} only there")
else:
    print(f"(no {REF_TABLE.name} found, check skipped)")
