"""
plot_group_comparison.py - one picture of the HC vs SC group comparison (easier to read than the CSV).

16 small charts, one per feature. In each: every dot is one subject (14 HC, 14 SC), the box shows
the middle half of the group and the line inside it is the median. The title gives the
Mann-Whitney p-value after Holm correction (from group_comparison.csv).

Input (made by group_specific_analysis.py):
  outputs/group_specific/features_group_specific_16.csv   (MAPS = "group-specific maps")
  outputs/features_final_16.csv                           (MAPS = "shared maps")
  outputs/group_specific/group_comparison.csv
Output: outputs/group_specific/group_comparison_HC_SC.png  (or ..._shared_maps.png)

Run:  eeg_env\\Scripts\\python.exe scripts\\plot_group_comparison.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "outputs" / "group_specific"
MAPS = "group-specific maps"          # or "shared maps"
SEED = 42
COLOURS = {"HC": "#2a78d6", "SC": "#eb6834"}
# --------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

table_file = OUT / "features_group_specific_16.csv" if MAPS == "group-specific maps" \
    else PROJECT_DIR / "outputs" / "features_final_16.csv"
features = pd.read_csv(table_file)
comp = pd.read_csv(OUT / "group_comparison.csv")
comp = comp[comp["maps"] == MAPS].set_index("feature")
rng = np.random.default_rng(SEED)

UNITS = {"duration": "ms", "occurrence": "per second"}
fig, axes = plt.subplots(4, 4, figsize=(13, 12))
for ax, col in zip(axes.flat, comp.index):
    scale = 1000 if col.endswith("_duration") else 1
    groups = [features.loc[features.label == 0, col] * scale, features.loc[features.label == 1, col] * scale]
    box = ax.boxplot(groups, widths=0.5, showfliers=False, patch_artist=True,
                     medianprops=dict(color="#222222", lw=2), whiskerprops=dict(color="#888888"),
                     capprops=dict(color="#888888"))
    for patch, c in zip(box["boxes"], COLOURS.values()):
        patch.set(facecolor=c, alpha=0.18, edgecolor=c, lw=1.2)
    for x, (vals, c) in enumerate(zip(groups, COLOURS.values()), start=1):
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=26, color=c,
                   edgecolor="white", lw=0.8, zorder=3)
    ax.set_xticks([1, 2], list(COLOURS))
    name = comp.loc[col, "name"]
    unit = UNITS.get(col.split("_")[-1], "probability")
    p = comp.loc[col, "p_holm"]
    ax.set_title(f"{name}\np (corrected) = {p:.2f}", fontsize=10,
                 color="#222222" if p >= 0.05 else "#b00020")
    ax.set_ylabel(unit, fontsize=8, color="#555555")
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)

fig.suptitle(f"Healthy controls (HC, n = 14) vs schizophrenia (SC, n = 14): the 16 microstate features "
             f"({MAPS})\nEach dot = one subject. Mann-Whitney U test, Holm-corrected over 16 features: "
             f"no feature differs significantly (all p > 0.05).", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.5)
name = "group_comparison_HC_SC.png" if MAPS == "group-specific maps" else "group_comparison_HC_SC_shared_maps.png"
fig.savefig(OUT / name, dpi=150)
print("Saved", OUT / name)
