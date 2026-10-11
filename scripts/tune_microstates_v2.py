"""
tune_microstates_v2.py  (PROJECT script - put it in MLproject/scripts/)

Round 2. Round 1 showed two things:
  (1) heavy smoothing (factor=10, half_window=10) is a big part of the problem
  (2) even with NO smoothing one state still dominates -> the shared model itself
      is probably being pulled toward artifacts / a few loud subjects.

This script fixes the model-fitting side:
  - drops the loudest GFP peaks of each subject (likely blinks / movement)
  - gives every subject the SAME number of peaks (so no subject dominates)
Then it tests a few ways of cleaning up the labels afterwards.

Filter is fixed at 2-20 Hz (what the papers used, and it had slightly higher GEV).
Run:  python tune_microstates_v2.py
"""
import os
import numpy as np
import mne
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tuning")
os.makedirs(OUT_DIR, exist_ok=True)

RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BADS = {"s07.edf": ["C4"], "h14.edf": ["F7"], "s11.edf": ["F7"], "s12.edf": ["P7"]}
LO, HI = 2, 20
K_VALUES = [4, 5, 6]
DROP_TOP_PERCENT = 5        # drop the loudest 5% of peaks in each subject
PEAKS_PER_SUBJECT = 1500    # same number for everybody
rng = np.random.default_rng(42)

# name -> predict() arguments.  min_segment_length is in samples (250 Hz: 5 = 20 ms)
SETTINGS = {
    "none":              dict(factor=0, half_window_size=1, min_segment_length=0),
    "none + min20ms":    dict(factor=0, half_window_size=1, min_segment_length=5),
    "light":             dict(factor=5, half_window_size=3, min_segment_length=0),
    "light + min20ms":   dict(factor=5, half_window_size=3, min_segment_length=5),
}

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".edf"))
raws, peak_arrays, info = [], [], None

for f in files:
    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, f), preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if f in BADS:
        raw.info["bads"] = BADS[f]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.filter(LO, HI, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raws.append(raw)

    p = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
    d = p.get_data()
    info = info or p.info
    gfp = d.std(axis=0)                                   # strength of each peak
    keep = gfp <= np.percentile(gfp, 100 - DROP_TOP_PERCENT)
    d = d[:, keep]
    n = min(PEAKS_PER_SUBJECT, d.shape[1])
    d = d[:, rng.choice(d.shape[1], size=n, replace=False)]
    peak_arrays.append(d)
    print(f"{f}: kept {n} peaks")

pooled = ChData(np.concatenate(peak_arrays, axis=1), info)
print(f"\nPooled peaks: {pooled.get_data().shape[1]}  (equal share per subject)\n")

print(f"{'k':>2} {'GEV':>6} {'setting':>16} | {'biggest-state cov':>17} {'#subj >70%':>10} "
      f"{'mean dur ms':>11} | mean coverage of each state")

for k in K_VALUES:
    model = ModKMeans(n_clusters=k, n_init=50, random_state=42)
    model.fit(pooled, n_jobs=1, verbose="ERROR")
    try:
        fig = model.plot(show=False)
        fig.savefig(os.path.join(OUT_DIR, f"v2_maps_k{k}.png"), dpi=120)
    except Exception as e:
        print("   (map plot skipped:", e, ")")

    for name, args in SETTINGS.items():
        covs_all, durs = [], []
        for raw in raws:
            seg = model.predict(raw, verbose="ERROR", **args)
            par = seg.compute_parameters()
            covs_all.append([par.get(f"{s}_timecov", np.nan) for s in range(k)])
            durs.append(np.nanmean([par.get(f"{s}_meandurs", np.nan) for s in range(k)]))
        covs_all = np.array(covs_all)
        biggest = np.nanmax(covs_all, axis=1)
        per_state = " ".join(f"{c:.2f}" for c in np.nanmean(covs_all, axis=0))
        print(f"{k:>2} {model.GEV_:>6.3f} {name:>16} | {biggest.mean():>17.2f} "
              f"{int((biggest > 0.7).sum()):>10d} {np.mean(durs) * 1000:>11.0f} | {per_state}")

print("\nDone. Send me this table and the v2_maps_k*.png pictures from outputs/tuning/.")