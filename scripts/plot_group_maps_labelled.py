"""
plot_group_maps_labelled.py - the 4 microstate maps of HC and of SC (fitted separately by
group_specific_analysis.py), with every electrode NAME written at its place, so the electrode
placement can be checked by eye. Also prints how similar the HC and SC map of each letter are.

Electrode placement: standard 10-20 positions (MNE "standard_1020"); old names renamed
T3->T7, T4->T8, T5->P7, T6->P8. Odd numbers = left, even = right, z = midline. Nose at the top.

Input:  outputs/group_specific/model_HC.pkl, model_SZ.pkl, outputs/shared_microstate_model_v3.pkl
        (the shared maps are used only to give each map its letter A-D)
Output: outputs/group_specific/group_maps_HC_SC_labelled.png

Run:  eeg_env\\Scripts\\python.exe scripts\\plot_group_maps_labelled.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "outputs" / "group_specific"
GROUPS = {"HC": "HC (healthy controls, 14)", "SZ": "SC (schizophrenia, 14)"}
# --------------------------------------------------------------

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib import patheffects                # noqa: E402
import mne                                        # noqa: E402
import numpy as np                                # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import microstate_model as mm                     # noqa: E402

shared = mm.load_model()
ref = np.asarray(shared.cluster_centers_)
info = shared.info


def unit(a):
    a = a - a.mean(axis=-1, keepdims=True)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# Step 1: each group's maps, put in order A B C D (matched to the shared maps by |correlation|)
maps = {}
for g in GROUPS:
    with open(OUT / f"model_{g}.pkl", "rb") as fh:
        centers = np.asarray(pickle.load(fh).cluster_centers_)
    r, c = linear_sum_assignment(-np.abs(unit(centers) @ unit(ref).T))
    maps[g] = {mm.STATE_NAMES[int(j)]: centers[i] for i, j in zip(r, c)}

# Step 2: same sign for HC and SC (polarity is ignored in microstates; this only matches colours)
letters = sorted(maps["HC"])
for L in letters:
    if np.corrcoef(maps["HC"][L], maps["SZ"][L])[0, 1] < 0:
        maps["SZ"][L] = -maps["SZ"][L]
sim = {L: abs(np.corrcoef(maps["HC"][L], maps["SZ"][L])[0, 1]) for L in letters}
print("HC vs SC similarity per map:", {L: round(v, 3) for L, v in sim.items()})

# Step 3: draw, with electrode names
fig, axes = plt.subplots(2, 4, figsize=(13, 7))
for row, (g, title) in enumerate(GROUPS.items()):
    for col, L in enumerate(letters):
        ax = axes[row, col]
        mne.viz.plot_topomap(unit(maps[g][L]), info, axes=ax, show=False, contours=0, cmap="RdBu_r",
                             names=info["ch_names"], sensors="k.")
        for line in ax.lines:                          # hide the sensor dots; the names mark the places
            if line.get_marker() not in ("None", None, ""):
                line.set_visible(False)
        for t in ax.texts:
            t.set(fontsize=8, fontweight="bold", ha="center", va="center",
                  path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
        ax.set_title(L if row else f"{L}   (HC vs SC: r = {sim[L]:.2f})", fontsize=11)
    axes[row, 0].text(-0.25, 0.5, title.replace(" (", "\n("), transform=axes[row, 0].transAxes,
                      ha="right", va="center", fontsize=11, fontweight="bold")
fig.suptitle("Microstate maps fitted separately for HC and SC (2-20 Hz, modified k-means, k = 4)\n"
             "19 electrodes, standard 10-20 positions, nose at the top; red / blue = opposite polarity",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "group_maps_HC_SC_labelled.png", dpi=160, bbox_inches="tight")
print("Saved", OUT / "group_maps_HC_SC_labelled.png")
