"""
plot_transition_matrix.py - the 12 transition probabilities drawn as 4 x 4 heatmaps.

Row i, column j = P(i -> j): the probability that, when the brain LEAVES state i, the next
state is j. The diagonal is empty (staying in a state is not a transition) and each row sums to 1.
Three panels: the mean over the 14 controls, the mean over the 14 patients, and patients minus
controls. Descriptive only (no test, no correction): the group statistics give the p-values.

Input:  outputs/eeg_features_epoch_mean_15min.csv (the 24-feature table, 28 subjects)
Output: outputs/transition_matrix_heatmap.png

Run:  eeg_env\\Scripts\\python.exe scripts\\plot_transition_matrix.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLE = PROJECT_DIR / "outputs" / "eeg_features_epoch_mean_15min.csv"
OUT_PATH = PROJECT_DIR / "outputs" / "transition_matrix_heatmap.png"
# --------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from microstate_model import N_STATES, STATE_NAMES  # noqa: E402

df = pd.read_csv(TABLE)
# show the states in the order A, B, C, D (state numbers 0, 2, 3, 1)
order = sorted(range(N_STATES), key=lambda s: STATE_NAMES[s])
letters = [STATE_NAMES[s] for s in order]


def group_matrix(rows):
    # Step 1: 4 x 4 matrix of mean P(i -> j) over the subjects of one group (diagonal = NaN)
    m = np.full((N_STATES, N_STATES), np.nan)
    for a, i in enumerate(order):
        for b, j in enumerate(order):
            if i != j:
                m[a, b] = rows[f"trans_{i}to{j}"].mean()
    return m


hc, sz = group_matrix(df[df.label == 0]), group_matrix(df[df.label == 1])
panels = [("Controls (n=14)", hc, "viridis", (0.2, 0.45)),
          ("Patients (n=14)", sz, "viridis", (0.2, 0.45)),
          ("Patients minus controls", sz - hc, "coolwarm", (-0.06, 0.06))]

# Step 2: draw the three heatmaps with the value written in every cell
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (title, m, cmap, (lo, hi)) in zip(axes, panels):
    im = ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi)
    for a in range(N_STATES):
        for b in range(N_STATES):
            ax.text(b, a, "-" if a == b else f"{m[a, b]:+.3f}" if "minus" in title else f"{m[a, b]:.2f}",
                    ha="center", va="center", fontsize=10, color="black" if "minus" in title else "white")
    ax.set_xticks(range(N_STATES), [f"to {x}" for x in letters])
    ax.set_yticks(range(N_STATES), [f"from {x}" for x in letters])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Microstate transition probabilities P(from -> to); each row sums to 1", y=1.02)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
for name, m in [("controls", hc), ("patients", sz)]:
    print(f"\n{name} (rows = from, columns = to, order {letters}):\n{np.round(m, 3)}")
