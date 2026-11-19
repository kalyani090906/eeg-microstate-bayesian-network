"""
group_specific_analysis.py - the whole microstate analysis done SEPARATELY for the two groups
(asked for by the faculty), and a group comparison of the features.

Main analysis (the rest of the project): ONE set of 4 maps fitted on all 28 subjects (shared maps),
every subject labelled with the same maps.
This script (group-specific maps):
  1. HC maps: GFP peaks of the 14 healthy controls only (1,500 per subject, loudest 5% dropped)
     -> modified k-means, k = 4, 50 starts, seed 42. Same again for the 14 SZ patients.
  2. Give each group map the letter of the most similar SHARED map (one-to-one, |correlation|),
     so "A" means the A-like map in both groups and the feature columns line up
     (state0 = A, state1 = D, state2 = B, state3 = C, exactly as in the main tables).
  3. Every subject: clean, cut into 5-s epochs (first 15 min), reject bad epochs, label every
     sample with ITS OWN GROUP's maps, compute the 24 features per epoch, one row per subject.
  4. Keep the same 16 features as the main table (features_final_16_list.txt).
  5. Group comparison, HC vs SZ, for every feature, done twice: with the shared maps (the main
     tables) and with the group-specific maps (this script). Mann-Whitney U test (no normality
     assumption, 14 vs 14), effect size = rank-biserial r (+ means SZ higher), Holm correction
     over the 16 features.

Read the result carefully:
  * With group-specific maps, "A" in SZ is not exactly the same map as "A" in HC, so a difference
    in a feature mixes a difference in the maps with a difference in the dynamics. With shared
    maps both groups are measured with the same ruler. That is why the main analysis uses shared
    maps; this script is the check the faculty asked for.
  * These features must NOT be used to train a classifier: the maps that made them were chosen
    knowing each subject's group (label leakage). Use them for the group comparison only.

Outputs (outputs/group_specific/):
  model_HC.pkl, model_SZ.pkl                 the two group models
  group_maps_HC_SZ_shared.png                the 4 maps of HC, SZ and the shared model
  map_similarity.csv                         |correlation| of each group map with the shared map
  features_group_specific_24.csv / _16.csv   one row per subject (same columns as the main tables)
  group_comparison.csv                       HC vs SZ per feature, shared maps and group maps
  epoch_rejection_report.csv                 epochs kept per subject

Run:  eeg_env\\Scripts\\python.exe scripts\\group_specific_analysis.py      (about 20-30 minutes)
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUT = OUTPUT_DIR / "group_specific"
SEED = 42
GROUPS = {0: "HC", 1: "SZ"}           # label 0 = healthy control, 1 = schizophrenia
# --------------------------------------------------------------

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import mne                                        # noqa: E402
import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import mannwhitneyu              # noqa: E402

import eeg_preprocessing as pre                   # noqa: E402
import microstate_features as mf                  # noqa: E402
import microstate_model as mm                     # noqa: E402

OUT.mkdir(exist_ok=True)
shared = mm.load_model()
K = mm.N_STATES
shared_maps = np.asarray(shared.cluster_centers_)
names24 = mf.feature_names(K)
names16 = [ln.strip() for ln in (OUTPUT_DIR / "features_final_16_list.txt").read_text().splitlines() if ln.strip()]


def pretty(col):
    # state2_duration -> "B duration";  trans_3to1 -> "C->D"
    if col.startswith("trans_"):
        i, j = col[6:].split("to")
        return f"{mm.STATE_NAMES[int(i)]}->{mm.STATE_NAMES[int(j)]}"
    k, what = col[5:].split("_")
    return f"{mm.STATE_NAMES[int(k)]} {what}"


def abs_corr(a, b):
    # |spatial correlation| between every row of a and every row of b
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return np.abs(a @ b.T) / a.shape[1]


def holm(p):
    # Holm correction written out: sort p, multiply the i-th smallest by (m - i), keep it increasing
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj


# Step 1: fit one model per group on that group's GFP peaks (2-20 Hz, as in the main model)
files = {g: [f for f in pre.recording_files() if pre.subject_and_label(f)[1] == lab] for lab, g in GROUPS.items()}
models, to_shared, sim_rows = {}, {}, []
for g, group_files in files.items():
    rng = np.random.default_rng(SEED)
    peak_arrays, info = [], None
    for f in group_files:
        p, info = mm.sample_gfp_peaks(pre.load_clean(f), rng)
        peak_arrays.append(p)
    models[g] = mm.fit_shared_model(peak_arrays, info, seed=SEED)
    with open(OUT / f"model_{g}.pkl", "wb") as fh:
        pickle.dump(models[g], fh)

    # Step 2: group state s -> the shared state number it looks like (so the letters line up)
    sim = abs_corr(np.asarray(models[g].cluster_centers_), shared_maps)
    r, c = linear_sum_assignment(-sim)
    to_shared[g] = {int(s): int(j) for s, j in zip(r, c)}
    for s, j in zip(r, c):
        sim_rows.append({"group": g, "letter": mm.STATE_NAMES[int(j)], "abs_corr_with_shared_map": round(sim[s, j], 3),
                         "group_GEV": round(float(models[g].GEV_), 3)})
    print(f"{g}: maps fitted on {sum(p.shape[1] for p in peak_arrays)} peaks of {len(group_files)} subjects, "
          f"GEV {models[g].GEV_:.3f};  |corr| with shared maps: "
          + ", ".join(f"{mm.STATE_NAMES[int(j)]} {sim[s, j]:.2f}" for s, j in sorted(zip(r, c), key=lambda t: mm.STATE_NAMES[int(t[1])])),
          flush=True)
pd.DataFrame(sim_rows).to_csv(OUT / "map_similarity.csv", index=False)

# Figure: HC, SZ and shared maps side by side (sign flipped to match the shared map's colours)
letters = sorted(mm.STATE_NAMES.values())
by_letter = {v: k for k, v in mm.STATE_NAMES.items()}
rows_to_draw = [("shared (all 28)", shared_maps, {k: k for k in range(K)}, shared.GEV_)] + \
               [(f"{g} only (14)", np.asarray(models[g].cluster_centers_), to_shared[g], models[g].GEV_) for g in files]
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for row, (title, maps, mapping, gev) in enumerate(rows_to_draw):
    inverse = {j: s for s, j in mapping.items()}
    for col, L in enumerate(letters):
        j = by_letter[L]
        m = maps[inverse[j]]
        m = m if np.corrcoef(m, shared_maps[j])[0, 1] >= 0 else -m
        mne.viz.plot_topomap(m, shared.info, axes=axes[row, col], show=False, contours=0)
        axes[row, col].set_title(L, fontsize=12)
    axes[row, 0].text(-0.3, 0.5, f"{title}\nGEV {gev:.2f}", transform=axes[row, 0].transAxes,
                      ha="right", va="center", fontsize=10, fontweight="bold")
fig.suptitle("Microstate maps (2-20 Hz, modified k-means, k = 4): shared vs fitted per group", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "group_maps_HC_SZ_shared.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Step 3: features of every subject, labelled with its own group's maps
mean_rows, report_rows = [], []
for f in pre.recording_files():
    subject, label = pre.subject_and_label(f)
    g = GROUPS[label]
    relabel = np.array([to_shared[g][s] for s in range(K)])
    raw = pre.load_clean(f)
    epochs = pre.cut_epochs(raw)
    keep = pre.kept_epochs(epochs)
    rows, pooled = [], np.zeros((K, K))
    for i in keep:
        _, labels = mm.label_epoch(models[g], epochs[i])
        labels = np.where(labels >= 0, relabel[np.clip(labels, 0, None)], -1)   # group state -> shared number
        row, counts = mf.epoch_features(labels, raw.info["sfreq"], K)
        pooled += counts
        rows.append(row)
    mean_rows.append({"subject": subject, "label": label, **mf.subject_features(rows, pooled, K, "mean")})
    report_rows.append({"subject": subject, "group": g, "epochs_total": len(epochs), "epochs_kept": len(keep)})
    print(f"{subject} ({g}): {len(keep)} epochs labelled with the {g} maps", flush=True)

# Step 4: the 24- and 16-feature tables
table = pd.DataFrame(mean_rows)[["subject", "label"] + names24]
table.to_csv(OUT / "features_group_specific_24.csv", index=False)
table[["subject", "label"] + names16].to_csv(OUT / "features_group_specific_16.csv", index=False)
for lab, g in GROUPS.items():                                   # the same 16 features, one file per group
    table.loc[table.label == lab, ["subject", "label"] + names16].to_csv(OUT / f"features_16_{g}.csv", index=False)
pd.DataFrame(report_rows).to_csv(OUT / "epoch_rejection_report.csv", index=False)

# Step 5: HC vs SZ for every one of the 16 features, shared maps and group-specific maps
main = pd.read_csv(OUTPUT_DIR / "features_final_16.csv")
comp = []
for maps_name, df in [("shared maps", main), ("group-specific maps", table)]:
    p_values, rows = [], []
    for c in names16:
        hc, sz = df.loc[df.label == 0, c].to_numpy(), df.loc[df.label == 1, c].to_numpy()
        u, p = mannwhitneyu(sz, hc, alternative="two-sided")
        scale = 1000 if c.endswith("_duration") else 1                       # durations shown in ms
        rows.append({"maps": maps_name, "feature": c, "name": pretty(c),
                     "HC_median": round(np.median(hc) * scale, 3), "SZ_median": round(np.median(sz) * scale, 3),
                     "rank_biserial_r": 2 * u / (len(sz) * len(hc)) - 1, "p": p})
        p_values.append(p)
    for row, p_adj in zip(rows, holm(p_values)):
        row["p_holm"] = p_adj
    comp += rows
comp = pd.DataFrame(comp)
comp[["rank_biserial_r", "p", "p_holm"]] = comp[["rank_biserial_r", "p", "p_holm"]].round(3)
comp.to_csv(OUT / "group_comparison.csv", index=False)

wide = comp.pivot(index="name", columns="maps", values=["rank_biserial_r", "p", "p_holm"])
wide = wide.reindex([pretty(c) for c in names16])
print("\nHC vs SZ (Mann-Whitney U; r > 0 = SZ higher; Holm over 16 features)")
print(wide.to_string())
print(f"\nSaved in {OUT}")
