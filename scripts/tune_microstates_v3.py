"""
tune_microstates_v3.py  (PROJECT script - put it in MLproject/scripts/)

Faster round 3. Fits the k=4 shared model ONCE (loudest 5% of peaks dropped,
equal peaks per subject, 2-20 Hz), saves it, then tests a few label-cleanup
settings on only 6 subjects and prints the time each one takes.

Run:  python tune_microstates_v3.py
"""
import os
import time
import pickle
import numpy as np
import mne
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BADS = {"s07.edf": ["C4"], "h14.edf": ["F7"], "s11.edf": ["F7"], "s12.edf": ["P7"]}
LO, HI = 2, 20
K = 4
DROP_TOP_PERCENT = 5
PEAKS_PER_SUBJECT = 1500
TEST_SUBJECTS = ["h01.edf", "h02.edf", "h03.edf", "s01.edf", "s02.edf", "s03.edf"]
rng = np.random.default_rng(42)

SETTINGS = {
    "none":            dict(factor=0, half_window_size=1, min_segment_length=0),
    "light":           dict(factor=5, half_window_size=3, min_segment_length=0),
    "moderate":        dict(factor=5, half_window_size=5, min_segment_length=0),
    "light+min20ms":   dict(factor=5, half_window_size=3, min_segment_length=5),
}

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".edf"))
test_raws, peak_arrays, info = {}, [], None

for f in files:
    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, f), preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if f in BADS:
        raw.info["bads"] = BADS[f]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.filter(LO, HI, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    if f in TEST_SUBJECTS:
        test_raws[f] = raw

    p = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
    d = p.get_data()
    info = info or p.info
    gfp = d.std(axis=0)
    d = d[:, gfp <= np.percentile(gfp, 100 - DROP_TOP_PERCENT)]
    n = min(PEAKS_PER_SUBJECT, d.shape[1])
    peak_arrays.append(d[:, rng.choice(d.shape[1], size=n, replace=False)])
    print(f"{f}: loaded", flush=True)

pooled = ChData(np.concatenate(peak_arrays, axis=1), info)
model = ModKMeans(n_clusters=K, n_init=50, random_state=42)
model.fit(pooled, n_jobs=1, verbose="ERROR")
print(f"\nModel fitted. GEV = {model.GEV_:.3f}", flush=True)

with open(os.path.join(OUT_DIR, "shared_microstate_model_v3.pkl"), "wb") as fh:
    pickle.dump(model, fh)
try:
    model.plot(show=False).savefig(os.path.join(OUT_DIR, "v3_maps_k4.png"), dpi=120)
except Exception as e:
    print("(map plot skipped:", e, ")")

print(f"\n{'setting':>14} | {'biggest cov':>11} {'#subj>70%':>9} {'mean dur ms':>11} "
      f"{'seconds':>8} | mean coverage per state", flush=True)

for name, args in SETTINGS.items():
    t0 = time.perf_counter()
    covs_all, durs = [], []
    for f, raw in test_raws.items():
        seg = model.predict(raw, verbose="ERROR", **args)
        par = seg.compute_parameters()
        covs_all.append([par.get(f"{s}_timecov", np.nan) for s in range(K)])
        durs.append(np.nanmean([par.get(f"{s}_meandurs", np.nan) for s in range(K)]))
    covs_all = np.array(covs_all)
    biggest = np.nanmax(covs_all, axis=1)
    per_state = " ".join(f"{c:.2f}" for c in np.nanmean(covs_all, axis=0))
    print(f"{name:>14} | {biggest.mean():>11.2f} {int((biggest > 0.7).sum()):>9d} "
          f"{np.mean(durs) * 1000:>11.0f} {time.perf_counter() - t0:>8.0f} | {per_state}",
          flush=True)

print("\nDone. Send me this table and outputs/v3_maps_k4.png.")