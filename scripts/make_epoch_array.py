"""
make_epoch_array.py   (PROJECT script - put it in MLproject/scripts/)

The SHARED EPOCH SCRIPT for the deep-learning items (pretrained embeddings and fine-tuning).

extract_epoch_features.py already cuts the recordings into 5-second epochs and rejects the
abnormal ones, but it only keeps the microstate FEATURES. A pretrained EEG model needs the
raw epochs themselves. This script does the same cutting and the same rejection, and saves
the epochs as one array.

What it does
  1. Same preprocessing as extract_epoch_features.py: rename channels, interpolate the known bad
     channels, average reference.
  2. Decides WHICH epochs to keep exactly like extract_epoch_features.py: 5-second epochs, first
     180 (15 minutes), reject an epoch when |robust z-score of its log-variance| > 3.5.
     This decision always uses the 2-20 Hz signal, so the kept epochs are identical to the
     feature table even if you choose a different band for the model below.
  3. Saves the kept epochs, filtered with OUT_BAND (default 2-20 Hz, the same as our features).
     A pretrained model may expect another band or sampling rate: change OUT_BAND / OUT_SFREQ
     below (see the model's README).
  4. Checks the result against outputs/eeg_epoch_features_15min.csv: the same (subject, epoch) pairs.

Output: outputs/epochs_15min.npz  with
   X          float32 array (n_epochs, n_channels, n_samples), in volts as MNE gives them
   subject    text array, one per epoch (h01 ... s14)
   label      0 = healthy, 1 = schizophrenia
   epoch      epoch number inside the recording (the same number as in eeg_epoch_features_15min.csv)
   ch_names   the channel names, in the order of X
   sfreq      sampling rate of X
This file is large (about 450 MB at 250 Hz): share it through a drive. It is NOT for git.

Run:  python make_epoch_array.py
"""
import os
import numpy as np
import pandas as pd
import mne

# ---------------- settings (change here only) ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
REF_TABLE = "eeg_epoch_features_15min.csv"     # used only for the final check
OUT_FILE = "epochs_15min.npz"

# these must stay the same as in extract_epoch_features.py
SELECT_BAND = (2, 20)          # band used to DECIDE which epochs are abnormal
EPOCH_SEC = 5.0
MAX_EPOCHS = 180               # first 15 minutes
REJECT_Z = 3.5

# these you may change for the pretrained model
OUT_BAND = (2, 20)             # band of the saved epochs; None = no extra filtering
OUT_SFREQ = None               # e.g. 200 to resample; None = keep 250 Hz
# --------------------------------------------------------------

RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BADS = {"s07.edf": ["C4"], "h14.edf": ["F7"], "s11.edf": ["F7"], "s12.edf": ["P7"]}


def load_clean(path, fname):
    """Read one EDF and do the preprocessing that does not depend on the band."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if fname in BADS:
        raw.info["bads"] = BADS[fname]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw


def band(raw, lo_hi):
    out = raw.copy()
    if lo_hi is not None:
        out.filter(lo_hi[0], lo_hi[1], verbose="ERROR")
    out.set_eeg_reference("average", verbose="ERROR")
    return out


files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".edf"))
X_parts, subj, lab, ep_no = [], [], [], []
ch_names, sfreq_out = None, None

for f in files:
    s = f.replace(".edf", "")
    label = 1 if s.startswith("s") else 0
    base = load_clean(os.path.join(DATA_DIR, f), f)

    # 1) which epochs are kept: same rule as extract_epoch_features.py
    sel = band(base, SELECT_BAND)
    ep_sel = mne.make_fixed_length_epochs(sel, duration=EPOCH_SEC, preload=True, verbose="ERROR")
    if MAX_EPOCHS:
        ep_sel = ep_sel[:MAX_EPOCHS]
    n_total = len(ep_sel)
    logvar = np.log(ep_sel.get_data().var(axis=(1, 2)) + 1e-30)
    med = np.median(logvar)
    mad = np.median(np.abs(logvar - med)) + 1e-12
    z = 0.6745 * (logvar - med) / mad
    keep_idx = np.where(np.abs(z) <= REJECT_Z)[0]

    # 2) the epochs to save: same boundaries, possibly another band / sampling rate
    out = band(base, OUT_BAND)
    ep_out = mne.make_fixed_length_epochs(out, duration=EPOCH_SEC, preload=True, verbose="ERROR")
    if MAX_EPOCHS:
        ep_out = ep_out[:MAX_EPOCHS]
    if OUT_SFREQ:
        ep_out = ep_out.resample(OUT_SFREQ, verbose="ERROR")
    data = ep_out.get_data()[keep_idx].astype(np.float32)

    X_parts.append(data)
    subj += [s] * len(keep_idx)
    lab += [label] * len(keep_idx)
    ep_no += [int(i) for i in keep_idx]
    ch_names = ep_out.ch_names
    sfreq_out = float(ep_out.info["sfreq"])
    print(f"{s}: kept {len(keep_idx)}/{n_total} epochs", flush=True)

X = np.concatenate(X_parts, axis=0)
np.savez_compressed(os.path.join(OUTPUT_DIR, OUT_FILE), X=X, subject=np.array(subj),
                    label=np.array(lab), epoch=np.array(ep_no),
                    ch_names=np.array(ch_names), sfreq=sfreq_out)
print(f"\nSaved outputs/{OUT_FILE}: X shape {X.shape} (epochs, channels, samples), sfreq {sfreq_out} Hz")

# ---- check against the feature table ----
ref_path = os.path.join(OUTPUT_DIR, REF_TABLE)
if os.path.exists(ref_path):
    ref = pd.read_csv(ref_path)
    mine = set(zip(subj, ep_no))
    theirs = set(zip(ref["subject"], ref["epoch"].astype(int)))
    if mine == theirs:
        print(f"CHECK OK: the same {len(mine)} (subject, epoch) pairs as {REF_TABLE}")
    else:
        print(f"CHECK FAILED: {len(mine)} epochs here, {len(theirs)} in {REF_TABLE}; "
              f"{len(mine - theirs)} only here, {len(theirs - mine)} only there")
else:
    print(f"(no {REF_TABLE} found, check skipped)")