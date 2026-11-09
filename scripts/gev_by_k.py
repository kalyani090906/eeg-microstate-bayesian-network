"""
gev_by_k.py - how much of the signal do k microstate maps explain, for k = 2 ... 8?

Uses exactly the peaks of the final model (same cleaning, same 1,500 peaks per subject, same
seed 42, same 50 restarts; microstate_model), so the k = 4 row must reproduce GEV = 0.674.
GEV always rises with k (more templates always fit better); the question is where the gain
per extra map becomes small. Does NOT change the saved model.

Output: outputs/gev_by_k.csv and outputs/gev_by_k.png

Run:  eeg_env\\Scripts\\python.exe scripts\\gev_by_k.py      (about 10-20 minutes)
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
K_VALUES = range(2, 9)
SEED = 42
# --------------------------------------------------------------

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

import eeg_preprocessing as pre  # noqa: E402
import microstate_model as mm    # noqa: E402

# Step 1: the same peak sample as the final model (same generator, same file order)
rng = np.random.default_rng(SEED)
peak_arrays, info = [], None
for f in pre.recording_files():
    peaks, peak_info = mm.sample_gfp_peaks(pre.load_clean(f), rng)
    peak_arrays.append(peaks)
    info = info or peak_info
    print(f"{f}: loaded", flush=True)

# Step 2: fit modified k-means for every k and record its GEV
rows = []
for k in K_VALUES:
    t0 = time.perf_counter()
    model = mm.fit_shared_model(peak_arrays, info, n_states=k, seed=SEED)
    rows.append({"k": k, "GEV": model.GEV_})
    gain = model.GEV_ - rows[-2]["GEV"] if len(rows) > 1 else float("nan")
    print(f"k = {k}: GEV = {model.GEV_:.3f}   gain over k-1: {gain:+.3f}   "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

# Step 3: table (with the gain from each extra map) and the curve
res = pd.DataFrame(rows)
res["gain_over_previous_k"] = res["GEV"].diff()
res.round(6).to_csv(OUTPUT_DIR / "gev_by_k.csv", index=False)   # 6 decimals: no double rounding in reports

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(res["k"], res["GEV"] * 100, marker="o")
ax.axvline(4, color="grey", linestyle=":", label="k = 4 (our model)")
for k, g in zip(res["k"], res["GEV"]):
    ax.annotate(f"{g * 100:.1f}%", (k, g * 100), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set_xlabel("number of microstate maps k")
ax.set_ylabel("GEV (% of signal explained)")
ax.set_title("GEV by number of maps\n(28 subjects, 42,000 GFP peaks)", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "gev_by_k.png", dpi=160)
print(f"\n{res.round(3).to_string(index=False)}\nSaved: outputs/gev_by_k.csv, outputs/gev_by_k.png")
