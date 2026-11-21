"""
ranksum_group_test.py - HC vs SC on each of the 16 features with the Wilcoxon rank-sum test
(as asked by the faculty), feature by feature, with a picture.

Wilcoxon rank-sum test: put all 28 values of one feature in order, give them ranks 1..28, and ask
whether the SC ranks are higher or lower than chance would give. No bell-curve assumption, so it
suits 14 vs 14. (It is the same test as Mann-Whitney U, written differently.)

For every feature this reports:
  p_individual  the p-value of that feature on its own (what the faculty asked for)
  p_holm        the same p after Holm correction for testing 16 features (shown for honesty:
                with 16 tests, about 1 in 20 features can reach p < 0.05 by luck alone)
  r             effect size (rank-biserial): + = higher in SC, - = higher in HC, from -1 to 1

Done twice: shared maps (the main analysis, outputs/features_final_16.csv) and group-specific maps
(outputs/group_specific/features_group_specific_16.csv, made by group_specific_analysis.py).

Outputs (outputs/group_specific/):
  ranksum_results.csv                   one row per map set x feature
  ranksum_HC_SC_shared_maps.png         16 small charts, one per feature
  ranksum_HC_SC_group_maps.png

Run:  eeg_env\\Scripts\\python.exe scripts\\ranksum_group_test.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUT = OUTPUT_DIR / "group_specific"
TABLES = {"shared maps": OUTPUT_DIR / "features_final_16.csv",
          "group maps": OUT / "features_group_specific_16.csv"}
ALPHA = 0.05
SEED = 42
COLOURS = {"HC": "#2a78d6", "SC": "#eb6834"}
# --------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy.stats import mannwhitneyu, ranksums  # noqa: E402

import microstate_model as mm            # noqa: E402


def pretty(col):
    # state2_duration -> "B duration";  trans_3to1 -> "C->D"
    if col.startswith("trans_"):
        i, j = col[6:].split("to")
        return f"{mm.STATE_NAMES[int(i)]}->{mm.STATE_NAMES[int(j)]}"
    k, what = col[5:].split("_")
    return f"{mm.STATE_NAMES[int(k)]} {what}"


def holm(p):
    # Holm correction: sort p, multiply the i-th smallest by (m - i), keep it increasing, cap at 1
    p = np.asarray(p, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj


rows = []
for maps, path in TABLES.items():
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in ("subject", "label")]
    block = []
    for c in features:
        scale = 1000 if c.endswith("_duration") else 1                      # durations in ms
        hc, sz = df.loc[df.label == 0, c].to_numpy() * scale, df.loc[df.label == 1, c].to_numpy() * scale
        stat, p = ranksums(sz, hc)                                         # Wilcoxon rank-sum
        u = mannwhitneyu(sz, hc).statistic
        block.append({"maps": maps, "feature": c, "name": pretty(c),
                      "HC_median": round(float(np.median(hc)), 3), "SC_median": round(float(np.median(sz)), 3),
                      "higher_in": "SC" if np.median(sz) > np.median(hc) else "HC",
                      "ranksum_z": round(float(stat), 3), "p_individual": round(float(p), 4),
                      "r": round(2 * u / (len(sz) * len(hc)) - 1, 3)})
    for row, p_adj in zip(block, holm([b["p_individual"] for b in block])):
        row["p_holm"] = round(float(p_adj), 4)
        row["significant_individual"] = row["p_individual"] < ALPHA
        row["significant_after_correction"] = p_adj < ALPHA
    rows += block
res = pd.DataFrame(rows)
res.to_csv(OUT / "ranksum_results.csv", index=False)

for maps in TABLES:
    r = res[res["maps"] == maps]
    print(f"\n=== {maps}: Wilcoxon rank-sum, HC vs SC ===")
    print(r[["name", "HC_median", "SC_median", "higher_in", "p_individual", "p_holm"]].to_string(index=False))

# Pictures: one per map set
UNITS = {"duration": "ms", "occurrence": "per second"}
rng = np.random.default_rng(SEED)
for maps, path in TABLES.items():
    df = pd.read_csv(path)
    r = res[res["maps"] == maps].set_index("feature")
    fig, axes = plt.subplots(4, 4, figsize=(13, 12))
    for ax, col in zip(axes.flat, r.index):
        scale = 1000 if col.endswith("_duration") else 1
        groups = [df.loc[df.label == 0, col] * scale, df.loc[df.label == 1, col] * scale]
        box = ax.boxplot(groups, widths=0.5, showfliers=False, patch_artist=True,
                         medianprops=dict(color="#222222", lw=2), whiskerprops=dict(color="#888888"),
                         capprops=dict(color="#888888"))
        for patch, c in zip(box["boxes"], COLOURS.values()):
            patch.set(facecolor=c, alpha=0.18, edgecolor=c, lw=1.2)
        for x, (vals, c) in enumerate(zip(groups, COLOURS.values()), start=1):
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=26, color=c,
                       edgecolor="white", lw=0.8, zorder=3)
        ax.set_xticks([1, 2], list(COLOURS))
        p = r.loc[col, "p_individual"]
        star = "  *" if p < ALPHA else ""
        ax.set_title(f"{r.loc[col, 'name']}\np = {p:.3f}{star}   (corrected {r.loc[col, 'p_holm']:.2f})",
                     fontsize=10, color="#b00020" if p < ALPHA else "#222222",
                     fontweight="bold" if p < ALPHA else "normal")
        ax.set_ylabel(UNITS.get(col.split("_")[-1], "probability"), fontsize=8, color="#555555")
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    n_sig = int(r["significant_individual"].sum())
    fig.suptitle(f"HC (n = 14) vs SC (n = 14), 16 microstate features, {maps}: Wilcoxon rank-sum test per feature\n"
                 f"Each dot = one subject. * = p < 0.05 on its own ({n_sig} of 16 features); "
                 f"corrected = Holm correction for 16 tests", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.5)
    name = "ranksum_HC_SC_shared_maps.png" if maps == "shared maps" else "ranksum_HC_SC_group_maps.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("Saved", OUT / name)
