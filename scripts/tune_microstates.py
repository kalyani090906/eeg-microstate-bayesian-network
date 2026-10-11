"""
tune_microstates.py  (PROJECT script - put it in MLproject/scripts/)

Goal: find preprocessing + smoothing settings where NO single microstate
swallows the recording, and the average state lasts ~60-120 ms.

It tries:  2 filter bands  x  k = 3..6  x  3 smoothing settings
and prints one summary line per combination. Nothing is saved except
a few map pictures, so it is safe to run.

Run:  python tune_microstates.py
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

BANDS = {"1-40Hz (yours)": (1, 40), "2-20Hz (papers)": (2, 20)}
K_VALUES = [3, 4, 5, 6]
# name -> arguments for model.predict
SMOOTHING = {
    "none":   dict(factor=0, half_window_size=1, min_segment_length=0),
    "light":  dict(factor=5, half_window_size=3, min_segment_length=0),
    "yours":  dict(factor=10, half_window_size=10, min_segment_length=0),
}

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".edf"))


def load(fname, lo, hi):
    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, fname), preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if fname in BADS:
        raw.info["bads"] = BADS[fname]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.filter(lo, hi, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    return raw


for band_name, (lo, hi) in BANDS.items():
    print("\n" + "=" * 78)
    print(f"BAND: {band_name}")
    print("=" * 78)

    raws, peak_arrays, info = [], [], None
    for f in files:
        raw = load(f, lo, hi)
        raws.append(raw)
        p = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
        peak_arrays.append(p.get_data())
        info = info or p.info
    pooled = ChData(np.concatenate(peak_arrays, axis=1), info)

    print(f"{'k':>2} {'GEV':>6} {'smooth':>7} | {'biggest-state coverage (mean)':>30} "
          f"{'#subjects with a state >70%':>28} {'mean duration (ms)':>20}")

    for k in K_VALUES:
        # n_init=30 keeps this fast; your final model can use 100 again
        model = ModKMeans(n_clusters=k, n_init=30, random_state=42)
        model.fit(pooled, n_jobs=1, verbose="ERROR")

        try:
            fig = model.plot(show=False)
            fig.savefig(os.path.join(OUT_DIR, f"maps_{lo}-{hi}Hz_k{k}.png"), dpi=120)
        except Exception as e:  # plotting is a bonus, never block the table
            print("   (map plot skipped:", e, ")")

        for sm_name, sm_args in SMOOTHING.items():
            max_covs, durs = [], []
            for raw in raws:
                seg = model.predict(raw, verbose="ERROR", **sm_args)
                params = seg.compute_parameters()
                covs = [params.get(f"{s}_timecov", np.nan) for s in range(k)]
                ds = [params.get(f"{s}_meandurs", np.nan) for s in range(k)]
                max_covs.append(np.nanmax(covs))
                durs.append(np.nanmean(ds))
            max_covs = np.array(max_covs)
            print(f"{k:>2} {model.GEV_:>6.3f} {sm_name:>7} | "
                  f"{max_covs.mean():>30.2f} {int((max_covs > 0.7).sum()):>28d} "
                  f"{np.mean(durs) * 1000:>20.0f}")

