"""
plot_labelled_maps.py - draw the 4 shared microstate maps with their A / B / C / D names.

How a map is drawn: each map is 19 numbers (one per electrode). MNE places the electrodes at
their standard 10-20 positions on a flat head and fills the space between them by interpolation,
then colours the values (red / blue = opposite signs). Because microstates ignore polarity, red
and blue could be swapped without changing the state.

IMPORTANT: this only changes the LABELS ON THE FIGURE. The state numbers 0-3 in all the feature
tables stay as they are. Each panel shows both, e.g. "C  (state 3)".

Run:  eeg_env\\Scripts\\python.exe scripts\\plot_labelled_maps.py
Output: outputs/maps_labelled_ABCD.png
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_DIR / "outputs" / "maps_labelled_ABCD.png"
NAMES = None   # None = the automatic suggestion; or e.g. {0: "A", 1: "D", 2: "B", 3: "C"} to force names
# --------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne                       # noqa: E402
import numpy as np               # noqa: E402

import microstate_model as mm    # noqa: E402

model = mm.load_model()
centers = np.asarray(model.cluster_centers_)           # (K, n_channels)
K = centers.shape[0]

# Step 1: the names (automatic shape matching, unless NAMES is set)
if NAMES is None:
    names, _, _, margins = mm.suggest_state_names(model)
    print("Automatic names (confirm by eye against [9]/[10] before using in the report):")
    for s in range(K):
        flag = "" if margins[s] > 0.1 else "   <-- UNCERTAIN"
        print(f"   state {s} -> {names[s]}   (margin {margins[s]:.2f}){flag}")
else:
    names = dict(NAMES)
    print("Using the names you set in NAMES:", names)

# Step 2: one scalp map per state, A to D from left to right
order = sorted(range(K), key=lambda s: names[s])
fig, axes = plt.subplots(1, K, figsize=(3 * K, 3.4))
for ax, s in zip(np.atleast_1d(axes), order):
    mne.viz.plot_topomap(centers[s], model.info, axes=ax, show=False, contours=0)
    ax.set_title(f"{names[s]}  (state {s})", fontsize=13)
fig.suptitle("Shared microstate maps (polarity ignored)", y=1.02)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
