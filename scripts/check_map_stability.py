"""
check_map_stability.py - are the 4 microstate maps an accident of one particular fit?

It re-fits the shared model several times and compares each new set of maps with the saved
model (shared_microstate_model_v3.pkl):
   * 4 fits with DIFFERENT random seeds (all subjects)
   * 4 fits with 4 randomly chosen subjects LEFT OUT
For each fit every new map is matched to its closest saved map (one-to-one, Hungarian method)
and the absolute correlation is printed (polarity ignored).

Rule of thumb: lowest matched correlation >= 0.9 -> stable; 0.8-0.9 -> fairly stable;
< 0.8 -> that map is not reliable.

Peak selection is identical to tune_microstates_v3.py (same module, same seed). Does not change
any file. Takes roughly as long as tune_microstates_v3.py.

Run:  eeg_env\\Scripts\\python.exe scripts\\check_map_stability.py
"""

# ---------------- settings (change only here) ----------------
N_INIT = 30                   # a bit lower than the final model (50), to save time
SEEDS = [1, 2, 3, 4]
N_LEAVE_OUT, N_SUBSET_RUNS = 4, 4
SEED = 42
# --------------------------------------------------------------

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import eeg_preprocessing as pre
import microstate_model as mm

ref = mm.load_model()
K = ref.cluster_centers_.shape[0]

# Step 1: the same peak sample as the saved model (same generator, same file order)
rng = np.random.default_rng(SEED)
files = pre.recording_files()
peak_arrays, info = [], None
for f in files:
    peaks, peak_info = mm.sample_gfp_peaks(pre.load_clean(f), rng)
    peak_arrays.append(peaks)
    info = info or peak_info
    print(f"{f}: loaded", flush=True)


def unit(c):
    # centre each map and scale it to length 1, so a dot product is a correlation
    c = c - c.mean(axis=1, keepdims=True)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)


R = unit(np.asarray(ref.cluster_centers_))


def compare(model):
    # Step 2: match new maps to saved maps one-to-one by |correlation|
    corr = np.abs(unit(np.asarray(model.cluster_centers_)) @ R.T)   # new x ref
    r, c = linear_sum_assignment(-corr)
    matched = np.zeros(K)
    for i, j in zip(r, c):
        matched[j] = corr[i, j]          # correlation for each REFERENCE state
    return matched


print(f"\n{'run':>22} {'GEV':>6} | matched correlation for reference state 0..{K - 1} | min   mean")
results = []


def run(name, arrays, seed):
    # Step 3: one re-fit, printed as one line
    t0 = time.perf_counter()
    m = mm.fit_shared_model(arrays, info, n_init=N_INIT, seed=seed)
    mt = compare(m)
    results.append(mt)
    print(f"{name:>22} {m.GEV_:>6.3f} | " + " ".join(f"{v:5.2f}" for v in mt)
          + f" | {mt.min():.2f}  {mt.mean():.2f}   ({time.perf_counter() - t0:.0f}s)", flush=True)


for s in SEEDS:
    run(f"seed {s}, all subjects", peak_arrays, s)

for r in range(N_SUBSET_RUNS):
    drop = rng.choice(len(peak_arrays), size=N_LEAVE_OUT, replace=False)
    keep = [a for i, a in enumerate(peak_arrays) if i not in drop]
    run(f"subset {r + 1} (-{N_LEAVE_OUT} subj)", keep, SEED)
    print("      left out:", ", ".join(files[i].replace(".edf", "") for i in sorted(drop)))

# Step 4: the verdict per state
res = np.array(results)
print("\nWorst matched correlation per reference state across ALL runs:")
for s in range(K):
    v = res[:, s].min()
    verdict = "stable" if v >= 0.9 else ("fairly stable" if v >= 0.8 else "NOT reliable")
    print(f"  state {s} ({mm.STATE_NAMES[s]}): {v:.2f}  -> {verdict}")
