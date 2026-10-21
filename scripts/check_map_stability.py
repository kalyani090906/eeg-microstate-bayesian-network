"""
check_map_stability.py   (PROJECT script - put it in MLproject/scripts/)

Step 3: are the 4 microstate maps an accident of one particular fit?

It re-fits the shared model several times and compares each new set of maps with
your saved model (shared_microstate_model_v3.pkl):
   * 4 fits with DIFFERENT random seeds (all subjects)
   * 4 fits with 4 randomly chosen subjects LEFT OUT
For each fit it matches every new map to its closest saved map and prints the
correlation (absolute, polarity ignored).

Rule of thumb: lowest matched correlation >= 0.9 -> stable;
0.8-0.9 -> fairly stable; < 0.8 -> that map is not reliable.

Peak selection is identical to the tuned pipeline (2-20 Hz, loudest 5% of peaks
dropped, 1500 peaks per subject). Takes roughly as long as tune_microstates_v3.py.

Run:  python check_map_stability.py
"""
import os
import pickle
import time
import numpy as np
import mne
from scipy.optimize import linear_sum_assignment
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_FILE = "shared_microstate_model_v3.pkl"

RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BADS = {"s07.edf": ["C4"], "h14.edf": ["F7"], "s11.edf": ["F7"], "s12.edf": ["P7"]}
LO, HI = 2, 20
DROP_TOP_PERCENT = 5
PEAKS_PER_SUBJECT = 1500
N_INIT = 30                   # a bit lower than the final model, to save time
SEEDS = [1, 2, 3, 4]
N_LEAVE_OUT, N_SUBSET_RUNS = 4, 4

with open(os.path.join(OUTPUT_DIR, MODEL_FILE), "rb") as fh:
    ref = pickle.load(fh)
K = ref.cluster_centers_.shape[0]

rng = np.random.default_rng(42)
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".edf"))
peak_arrays, info = [], None
for f in files:
    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, f), preload=True, verbose="ERROR")
    raw.rename_channels(RENAME)
    raw.set_montage("standard_1020", on_missing="warn")
    if f in BADS:
        raw.info["bads"] = BADS[f]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.filter(LO, HI, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    p = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
    d = p.get_data()
    info = info or p.info
    gfp = d.std(axis=0)
    d = d[:, gfp <= np.percentile(gfp, 100 - DROP_TOP_PERCENT)]
    n = min(PEAKS_PER_SUBJECT, d.shape[1])
    peak_arrays.append(d[:, rng.choice(d.shape[1], size=n, replace=False)])
    print(f"{f}: loaded", flush=True)


def unit(c):
    c = c - c.mean(axis=1, keepdims=True)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)


R = unit(np.asarray(ref.cluster_centers_))


def compare(model):
    corr = np.abs(unit(np.asarray(model.cluster_centers_)) @ R.T)   # new x ref
    r, c = linear_sum_assignment(-corr)
    matched = np.zeros(K)
    for i, j in zip(r, c):
        matched[j] = corr[i, j]          # correlation for each REFERENCE state
    return matched


print(f"\n{'run':>22} {'GEV':>6} | matched correlation for reference state 0..{K - 1} | min   mean")
results = []


def run(name, arrays, seed):
    t0 = time.perf_counter()
    pooled = ChData(np.concatenate(arrays, axis=1), info)
    m = ModKMeans(n_clusters=K, n_init=N_INIT, random_state=seed)
    m.fit(pooled, n_jobs=1, verbose="ERROR")
    mt = compare(m)
    results.append(mt)
    print(f"{name:>22} {m.GEV_:>6.3f} | " + " ".join(f"{v:5.2f}" for v in mt)
          + f" | {mt.min():.2f}  {mt.mean():.2f}   ({time.perf_counter() - t0:.0f}s)", flush=True)


for s in SEEDS:
    run(f"seed {s}, all subjects", peak_arrays, s)

for r in range(N_SUBSET_RUNS):
    drop = rng.choice(len(peak_arrays), size=N_LEAVE_OUT, replace=False)
    keep = [a for i, a in enumerate(peak_arrays) if i not in drop]
    run(f"subset {r + 1} (-{N_LEAVE_OUT} subj)", keep, 42)
    print("      left out:", ", ".join(files[i].replace(".edf", "") for i in sorted(drop)))

res = np.array(results)
print("\nWorst matched correlation per reference state across ALL runs:")
for s in range(K):
    v = res[:, s].min()
    verdict = "stable" if v >= 0.9 else ("fairly stable" if v >= 0.8 else "NOT reliable")
    print(f"  state {s}: {v:.2f}  -> {verdict}")