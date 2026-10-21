"""
name_states.py   (PROJECT script - put it in MLproject/scripts/)

Step 2: suggest which of your states (0..3) is the classic microstate A, B, C or D.

HOW: each class has a typical SHAPE across the scalp:
   A  left-back  <->  right-front diagonal
   B  left-front <->  right-back  diagonal
   C  front <-> back axis
   D  centre-front peak vs the edges (roughly radial)
We build these idealised shapes from the electrode positions and compare each of
your maps with them (absolute correlation, because microstates ignore polarity).
Then we pick the best one-to-one assignment.

CAUTION: these are idealised shapes, NOT the published template maps (which this
workspace cannot download). Treat the output as a suggestion. Look at the printed
scores: if two classes score almost the same for a map, the name is uncertain.
Always confirm by eye against a published figure of A-D before putting names
in the report.

Run:  python name_states.py
"""
import os
import pickle
import numpy as np
import mne
from scipy.optimize import linear_sum_assignment

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_FILE = "shared_microstate_model_v3.pkl"

with open(os.path.join(OUTPUT_DIR, MODEL_FILE), "rb") as fh:
    model = pickle.load(fh)

centers = np.asarray(model.cluster_centers_)           # (K, n_channels)
K = centers.shape[0]
ch_names = model.info["ch_names"]

mont = mne.channels.make_standard_montage("standard_1020")
pos = mont.get_positions()["ch_pos"]
xyz = np.array([pos[c] for c in ch_names])
x, y = xyz[:, 0], xyz[:, 1]            # x: left(-) -> right(+),  y: back(-) -> front(+)
x, y = x / np.abs(x).max(), y / np.abs(y).max()

templates = {
    "A": x + y,
    "B": -x + y,
    "C": y,
    "D": -(x ** 2 + (y - 0.2) ** 2),
}
names = list(templates)


def zs(v):
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


T = np.array([zs(templates[n]) for n in names])         # (4, n_ch)
C = np.array([zs(c) for c in centers])                  # (K, n_ch)
score = np.abs(C @ T.T)                                 # (K states, 4 classes)

print("Absolute correlation of each of YOUR states with the idealised classes:\n")
print("state " + " ".join(f"{n:>6}" for n in names))
for s in range(K):
    print(f"{s:>5} " + " ".join(f"{score[s, j]:6.2f}" for j in range(len(names))))

row, col = linear_sum_assignment(-score)
print("\nSuggested one-to-one naming:")
for s, j in zip(row, col):
    runner_up = np.sort(score[s])[-2]
    margin = score[s, j] - runner_up
    note = "" if margin > 0.1 else "   <-- UNCERTAIN (another class scores almost as high)"
    print(f"  state {s}  ->  {names[j]}   (score {score[s, j]:.2f}, margin {margin:.2f}){note}")

print("\nConfirm by eye against a published A-D figure before using these names.")