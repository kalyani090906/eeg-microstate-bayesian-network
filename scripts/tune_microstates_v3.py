"""
tune_microstates_v3.py - fit the ONE shared 4-state microstate model for all 28 subjects.

(Round 3 of tuning; rounds 1 and 2 are in the ml project dump.)
  1. clean every recording (2-20 Hz, average reference; eeg_preprocessing)
  2. from each subject take its GFP peaks, drop the loudest 5%, pick 1,500 at random
     (the same random generator, seed 42, used in file order)
  3. pool the 28 x 1,500 = 42,000 peak maps and run modified k-means with k = 4,
     50 random restarts (microstate_model). No diagnosis labels are used.
  4. save the model (outputs/shared_microstate_model_v3.pkl) and a picture of the 4 maps
  5. compare 4 label-smoothing settings on 6 subjects (biggest coverage, mean duration, time);
     "moderate" (factor 5, half window 5) was chosen and is used in extract_epoch_features.py

Run:  eeg_env\\Scripts\\python.exe scripts\\tune_microstates_v3.py
WARNING: this OVERWRITES the saved model. Re-fitting can number the states differently, so all
features and the A-D names would have to be redone. Only run it to rebuild from scratch.
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
SEED = 42
TEST_SUBJECTS = ["h01.edf", "h02.edf", "h03.edf", "s01.edf", "s02.edf", "s03.edf"]
SETTINGS = {                 # label-smoothing settings compared in step 5
    "none":            dict(factor=0, half_window_size=1, min_segment_length=0),
    "light":           dict(factor=5, half_window_size=3, min_segment_length=0),
    "moderate":        dict(factor=5, half_window_size=5, min_segment_length=0),
    "light+min20ms":   dict(factor=5, half_window_size=3, min_segment_length=5),
}
# --------------------------------------------------------------

import time

import numpy as np

import eeg_preprocessing as pre
import microstate_model as mm

# Steps 1-2: clean each recording and sample its GFP peaks (one generator, in file order)
rng = np.random.default_rng(SEED)
test_raws, peak_arrays, info = {}, [], None
for f in pre.recording_files():
    raw = pre.load_clean(f)
    if f in TEST_SUBJECTS:
        test_raws[f] = raw
    peaks, peak_info = mm.sample_gfp_peaks(raw, rng)
    peak_arrays.append(peaks)
    info = info or peak_info
    print(f"{f}: loaded", flush=True)

# Steps 3-4: fit the shared model on all pooled peaks, save it and draw the maps
model = mm.fit_shared_model(peak_arrays, info, seed=SEED)
print(f"\nModel fitted. GEV = {model.GEV_:.3f}", flush=True)
mm.save_model(model)
try:
    model.plot(show=False).savefig(OUTPUT_DIR / "v3_maps_k4.png", dpi=120)
except Exception as e:
    print("(map plot skipped:", e, ")")

# Step 5: compare the smoothing settings on the 6 test subjects
K = mm.N_STATES
print(f"\n{'setting':>14} | {'biggest cov':>11} {'#subj>70%':>9} {'mean dur ms':>11} "
      f"{'seconds':>8} | mean coverage per state", flush=True)
for name, args in SETTINGS.items():
    t0 = time.perf_counter()
    covs_all, durs = [], []
    for f, raw in test_raws.items():
        par = model.predict(raw, verbose="ERROR", **args).compute_parameters()
        covs_all.append([par.get(f"{s}_timecov", np.nan) for s in range(K)])
        durs.append(np.nanmean([par.get(f"{s}_meandurs", np.nan) for s in range(K)]))
    covs_all = np.array(covs_all)
    biggest = np.nanmax(covs_all, axis=1)
    per_state = " ".join(f"{c:.2f}" for c in np.nanmean(covs_all, axis=0))
    print(f"{name:>14} | {biggest.mean():>11.2f} {int((biggest > 0.7).sum()):>9d} "
          f"{np.mean(durs) * 1000:>11.0f} {time.perf_counter() - t0:>8.0f} | {per_state}", flush=True)
