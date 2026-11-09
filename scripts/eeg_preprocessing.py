"""
eeg_preprocessing.py - reading and cleaning the EEG, cutting it into epochs, rejecting bad epochs.
Every pipeline script imports these functions, so each step is defined in ONE place.

Steps for one recording (RepOD: 19 channels, 10-20 system, 250 Hz, eyes closed):
  1. read the EDF file
  2. rename the old channel names to the modern ones (T3->T7, T4->T8, T5->P7, T6->P8),
     so MNE can place the electrodes on a standard 10-20 head
  3. rebuild the 4 known bad channels from their neighbours (spherical-spline interpolation);
     the list comes from check_bad_channels.py
  4. band-pass filter 2-20 Hz (the band used for microstates in [9] and [10])
  5. average reference: subtract the mean of the 19 channels at every sample, so the map does
     not depend on one reference electrode (a microstate map is a pattern ACROSS electrodes)
  6. cut into 5-second non-overlapping epochs and keep the first 180 (15 minutes) for everyone
  7. reject abnormal epochs: robust z-score of each epoch's log-variance, |z| > 3.5 is rejected
     (blinks, movement, flat stretches). Done per subject.

Nothing to run here.
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BADS = {"s07.edf": ["C4"], "h14.edf": ["F7"], "s11.edf": ["F7"], "s12.edf": ["P7"]}
BAND = (2, 20)             # Hz, band-pass for microstates
EPOCH_SEC = 5.0            # epoch length, as in [9] and [10]
MAX_EPOCHS = 180           # first 180 epochs = 15 minutes (same window for everyone, see s14)
REJECT_Z = 3.5             # reject an epoch when |robust z of its log-variance| > this
# --------------------------------------------------------------

import mne
import numpy as np


def recording_files():
    # All EDF files, sorted: h01.edf ... h14.edf, s01.edf ... s14.edf
    return sorted(f.name for f in DATA_DIR.glob("*.edf"))


def subject_and_label(fname):
    # "s07.edf" -> ("s07", 1): s = schizophrenia (label 1), h = healthy control (label 0)
    subject = fname.replace(".edf", "")
    return subject, (1 if subject.startswith("s") else 0)


def load_raw(fname, interpolate=True):
    # Steps 1-3: read, rename channels, put them on the 10-20 head, rebuild the bad channels
    raw = mne.io.read_raw_edf(DATA_DIR / fname, preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if interpolate and fname in BADS:
        raw.info["bads"] = BADS[fname]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw


def filter_and_reference(raw, band=BAND, notch=None):
    # Steps 4-5 on a copy: (optional mains notch), band-pass filter, average reference
    out = raw.copy()
    if notch is not None:
        out.notch_filter(notch, verbose="ERROR")
    if band is not None:
        out.filter(band[0], band[1], verbose="ERROR")
    out.set_eeg_reference("average", verbose="ERROR")
    return out


def load_clean(fname, band=BAND):
    # Steps 1-5 in one call: the cleaned recording used for microstates
    return filter_and_reference(load_raw(fname), band)


def cut_epochs(raw, max_epochs=MAX_EPOCHS):
    # Step 6: 5-second non-overlapping epochs, first max_epochs only (None = all)
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_SEC, preload=True, verbose="ERROR")
    return epochs[:max_epochs] if max_epochs else epochs


def kept_epochs(epochs, reject_z=REJECT_Z):
    # Step 7: robust z-score of log-variance per epoch (median / MAD, so a few huge outliers do
    # not hide each other); returns the indices of the epochs that are kept
    logvar = np.log(epochs.get_data().var(axis=(1, 2)) + 1e-30)
    med = np.median(logvar)
    mad = np.median(np.abs(logvar - med)) + 1e-12
    z = 0.6745 * (logvar - med) / mad
    return np.where(np.abs(z) <= reject_z)[0]
