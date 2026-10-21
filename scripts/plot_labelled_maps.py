"""
plot_labelled_maps.py   (PROJECT script - put it in MLproject/scripts/)

Draws the 4 shared microstate maps with their A / B / C / D names, for the report and slides.

IMPORTANT: this only changes the LABELS ON THE FIGURE. The state numbers 0-3 in all the
feature tables stay exactly as they are. The title of each panel shows both, e.g.
"C  (state 3)", so the figure can be matched to the columns state3_duration etc.

Where do the names come from?
  * By default the same idealised-shape matching as name_states.py (best one-to-one
    assignment, absolute correlation because microstates ignore polarity).
  * If you have confirmed the names by eye against the published figures of [9] and [10]
    (as we did for C and D), put them in NAMES below - these then win over the automatic ones.
    Example:   NAMES = {0: "A", 1: "D", 2: "B", 3: "C"}      (state number -> class name)

Run:  python plot_labelled_maps.py
Output: outputs/maps_labelled_ABCD.png
"""
import os
import pickle
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ---------------- settings (change here only) ----------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_FILE = "shared_microstate_model_v3.pkl"
NAMES = None            # e.g. {0: "A", 1: "D", 2: "B", 3: "C"}; None = automatic suggestion
OUT_FILE = "maps_labelled_ABCD.png"
# --------------------------------------------------------------

with open(os.path.join(OUTPUT_DIR, MODEL_FILE), "rb") as fh:
    model = pickle.load(fh)
centers = np.asarray(model.cluster_centers_)           # (K, n_channels)
K = centers.shape[0]
info = model.info


def suggest_names(centers, info):
    """Same idea as name_states.py: compare each map with idealised A/B/C/D shapes."""
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in info["ch_names"]])
    x, y = xyz[:, 0] / np.abs(xyz[:, 0]).max(), xyz[:, 1] / np.abs(xyz[:, 1]).max()
    templates = {"A": x + y, "B": -x + y, "C": y, "D": -(x ** 2 + (y - 0.2) ** 2)}
    cls = list(templates)

    def zs(v):
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-12)

    T = np.array([zs(templates[n]) for n in cls])
    C = np.array([zs(c) for c in centers])
    score = np.abs(C @ T.T)
    r, c = linear_sum_assignment(-score)
    names, margins = {}, {}
    for s, j in zip(r, c):
        names[int(s)] = cls[j]
        margins[int(s)] = score[s, j] - np.sort(score[s])[-2]
    return names, margins


if NAMES is None:
    names, margins = suggest_names(centers, info)
    print("Automatic names (confirm by eye against [9]/[10] before using in the report):")
    for s in range(K):
        flag = "" if margins[s] > 0.1 else "   <-- UNCERTAIN"
        print(f"   state {s} -> {names[s]}   (margin {margins[s]:.2f}){flag}")
else:
    names = dict(NAMES)
    print("Using the names you set in NAMES:", names)

order = sorted(range(K), key=lambda s: names[s])        # A, B, C, D from left to right
fig, axes = plt.subplots(1, K, figsize=(3 * K, 3.4))
for ax, s in zip(np.atleast_1d(axes), order):
    mne.viz.plot_topomap(centers[s], info, axes=ax, show=False, contours=0)
    ax.set_title(f"{names[s]}  (state {s})", fontsize=13)
fig.suptitle("Shared microstate maps (polarity ignored)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, OUT_FILE), dpi=160, bbox_inches="tight")
print(f"Saved: outputs/{OUT_FILE}")