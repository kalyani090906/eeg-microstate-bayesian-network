"""
make_pipeline_figures.py - pictures and real numbers that explain EEG -> GFP -> peaks -> maps -> labels.
(Explanation material for the project walkthrough. Reads the project data and model, changes nothing there.)

Figures in figures/ (subject h01, a short stretch of its recording):
  pipeline_fig1_filter.png   one electrode before and after the 2-20 Hz filter
  pipeline_fig2_gfp.png      GFP before and after filtering, with the GFP peaks marked
  pipeline_fig3_one_map.png  one GFP peak = 19 numbers = one "map" (bar chart and the same numbers on a head)
  pipeline_fig4_labels.png   1 second of a 5-s epoch: every sample labelled A/B/C/D -> segments
Table:
  numbers_per_subject.csv   samples, minutes, GFP peaks, peaks per second, time between peaks,
                            for every subject (filtered and unfiltered)

Run (from MLproject):  eeg_env\\Scripts\\python.exe project_walkthrough\\make_pipeline_figures.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                         # the MLproject folder
FIG = HERE / "figures"
SUBJECT = "h01.edf"
START_SEC = 60.0            # where the shown stretch starts
CHANNEL = "Cz"
# --------------------------------------------------------------

sys.path.insert(0, str(PROJECT / "scripts"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import mne                        # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import eeg_preprocessing as pre   # noqa: E402
import microstate_model as mm     # noqa: E402
import microstate_features as mf  # noqa: E402

COLOURS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

# ---------- numbers for every subject ----------
rows = []
for f in pre.recording_files():
    raw_unf = pre.filter_and_reference(pre.load_raw(f), band=None)   # average reference, NO filter
    raw = pre.load_clean(f)                                         # 2-20 Hz + average reference
    sf = raw.info["sfreq"]
    gfp_unf = mm.global_field_power(raw_unf.get_data())
    gfp = mm.global_field_power(raw.get_data())
    pk_unf, _ = find_peaks(gfp_unf, distance=2)
    pk, _ = find_peaks(gfp, distance=2)
    n15 = int(15 * 60 * sf)
    rows.append({"subject": f[:-4], "samples_whole": gfp.size, "minutes_whole": round(gfp.size / sf / 60, 1),
                 "gfp_values_first_15min": min(n15, gfp.size),
                 "peaks_whole_filtered": len(pk), "peaks_per_sec_filtered": round(len(pk) / (gfp.size / sf), 1),
                 "ms_between_peaks_filtered": round(1000 * np.median(np.diff(pk)) / sf, 1),
                 "peaks_per_sec_unfiltered": round(len(pk_unf) / (gfp.size / sf), 1),
                 "ms_between_peaks_unfiltered": round(1000 * np.median(np.diff(pk_unf)) / sf, 1),
                 "peaks_after_dropping_top5pct": int(np.sum(gfp[pk] <= np.percentile(gfp[pk], 95))),
                 "peaks_used_for_maps": min(1500, int(np.sum(gfp[pk] <= np.percentile(gfp[pk], 95))))})
    print(rows[-1], flush=True)
table = pd.DataFrame(rows)
table.to_csv(HERE / "numbers_per_subject.csv", index=False)
print(table.describe().round(1).to_string())

# ---------- figures for one subject ----------
raw_orig = pre.load_raw(SUBJECT)
raw_unf = pre.filter_and_reference(raw_orig, band=None)
raw = pre.load_clean(SUBJECT)
sf = raw.info["sfreq"]
a, b = int(START_SEC * sf), int((START_SEC + 2) * sf)          # a 2-second stretch
t_ms = (np.arange(a, b) - a) / sf * 1000
ch = raw.ch_names.index(CHANNEL)

# Figure 1: one electrode before / after the filter
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(t_ms, raw_unf.get_data()[ch, a:b] * 1e6, color="grey", lw=0.8, label="before filter (average reference)")
ax.plot(t_ms, raw.get_data()[ch, a:b] * 1e6, color="black", lw=1.4, label="after 2-20 Hz filter")
ax.set_xlabel("time (ms)")
ax.set_ylabel(f"voltage at {CHANNEL} (µV)")
ax.set_title(f"{SUBJECT[:-4]}, electrode {CHANNEL}, 2 seconds = 500 samples (one sample every 4 ms)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "pipeline_fig1_filter.png", dpi=150)

# Figure 2: GFP before / after the filter, peaks marked (first 0.5 s so the dots are visible)
c = a + int(0.5 * sf)
g_unf = mm.global_field_power(raw_unf.get_data()[:, a:c]) * 1e6
g = mm.global_field_power(raw.get_data()[:, a:c]) * 1e6
p_unf, _ = find_peaks(g_unf, distance=2)
p, _ = find_peaks(g, distance=2)
tt = (np.arange(c - a)) / sf * 1000
fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
m = table.mean(numeric_only=True)
for ax, gg, pp, title in [
        (axes[0], g_unf, p_unf, f"GFP BEFORE filtering: more jagged (all subjects: {m.peaks_per_sec_unfiltered:.0f} "
                                f"peaks/s, about {m.ms_between_peaks_unfiltered:.0f} ms apart)"),
        (axes[1], g, p, f"GFP AFTER 2-20 Hz filtering: smooth hills (all subjects: {m.peaks_per_sec_filtered:.0f} "
                        f"peaks/s, about {m.ms_between_peaks_filtered:.0f} ms apart)")]:
    ax.plot(tt, gg, color="black", lw=1)
    ax.plot(tt, gg, ".", color="grey", ms=3)
    ax.plot(tt[pp], gg[pp], "o", color="red", ms=6)
    ax.set_ylabel("GFP (µV)")
    ax.set_title(f"{title}; here {len(pp)} red-dot peaks in 0.5 s", fontsize=9)
axes[1].set_xlabel("time (ms); every grey dot is one sample (4 ms apart)")
fig.tight_layout()
fig.savefig(FIG / "pipeline_fig2_gfp.png", dpi=150)

# Figure 3: one GFP peak = 19 numbers = one map
k = a + p[np.argmax(g[p])]                                       # the highest peak in that stretch
values = raw.get_data()[:, k] * 1e6
fig = plt.figure(figsize=(11, 3.8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.bar(range(len(values)), values, color=["tab:red" if v > 0 else "tab:blue" for v in values])
ax1.set_xticks(range(len(values)), raw.ch_names, rotation=90, fontsize=8)
ax1.set_ylabel("µV")
ax1.set_title("one GFP peak = 19 numbers (one per electrode)", fontsize=10)
ax2 = fig.add_subplot(1, 2, 2)
mne.viz.plot_topomap(values, raw.info, axes=ax2, show=False, cmap="RdBu_r")
ax2.set_title("the SAME 19 numbers drawn on the head = a 'map'", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "pipeline_fig3_one_map.png", dpi=150)

# Figure 4: labels of every sample in 1 second of one kept epoch
model = mm.load_model()
epochs = pre.cut_epochs(raw)
e = pre.kept_epochs(epochs)[12]
_, labels = mm.label_epoch(model, epochs[e])
data = epochs[e].get_data()[0] * 1e6
gfp_e = mm.global_field_power(data)
s0, s1 = int(1 * sf), int(2 * sf)                                # second 1-2 of the epoch
tt = np.arange(s1 - s0) / sf * 1000
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(tt, gfp_e[s0:s1], color="black", lw=1)
for i in range(s1 - s0):
    lab = labels[s0 + i]
    if lab >= 0:
        ax.axvspan(tt[i] - 2, tt[i] + 2, color=COLOURS[mm.STATE_NAMES[lab]], alpha=0.35, lw=0)
segs = mf.segments(labels[s0:s1])
pos = 0
for state, length in segs:
    if state >= 0:
        ax.text((pos + length / 2) * 4, ax.get_ylim()[1] * 0.97, mm.STATE_NAMES[state],
                ha="center", va="top", fontsize=9, fontweight="bold")
    pos += length
ax.set_xlabel("time within the epoch (ms); colour = the map each sample was given")
ax.set_ylabel("GFP (µV)")
ax.set_title(f"1 second of one 5-s epoch: {len([s for s, _ in segs if s >= 0])} segments "
             f"-> these give duration, occurrence, coverage and transitions", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=v, alpha=0.5) for v in COLOURS.values()]
ax.legend(handles, COLOURS.keys(), loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "pipeline_fig4_labels.png", dpi=150)
print("Saved figures in", FIG)
