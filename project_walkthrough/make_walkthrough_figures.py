"""
make_walkthrough_figures.py - the extra explainer figures for the project walkthrough (not project results).

Uses the project's own modules (MLproject/scripts), so every figure shows exactly what the
pipeline does. Subject h01 is used as the example (any subject would do). Also copies the
8 pictures out of the executed toy notebook (notebooks/toy_gfp_demo.ipynb).

Writes: project_walkthrough/figures/*.png

Run (from MLproject):  eeg_env\\Scripts\\python.exe project_walkthrough\\make_walkthrough_figures.py
"""

# ---------------- settings (change only here) ----------------
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                         # the MLproject folder
FIG = HERE / "figures"
SUBJECT = "h01.edf"            # example subject
REJECT_SUBJECT = "h04.edf"     # a subject with several rejected epochs (159 of 180 kept)
EPOCH_INDEX = 12               # which 5-s epoch of h01 to label
SEED = 42
# --------------------------------------------------------------

sys.path.insert(0, str(PROJECT / "scripts"))
import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import find_peaks, welch

import eeg_preprocessing as pre
import microstate_model as mm

FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"A": "#2a6fdb", "B": "#e07b00", "C": "#2a9d55", "D": "#c03a6b", "-": "#bbbbbb"}
model = mm.load_model()
centers = np.asarray(model.cluster_centers_)                    # 4 x 19, order = state 0..3
letters = [mm.STATE_NAMES[k] for k in range(4)]                 # state number -> letter


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# Step 1: the 8 notebook pictures, in order
nb = json.loads((PROJECT / "notebooks" / "toy_gfp_demo.ipynb").read_text(encoding="utf-8"))
n = 0
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        png = out.get("data", {}).get("image/png")
        if png:
            n += 1
            (FIG / f"toy_{n}.png").write_bytes(base64.b64decode(png))
print("copied", n, "notebook pictures")

# Step 2: raw vs cleaned signal (same 3 s, 4 channels)
raw = pre.load_raw(SUBJECT)
clean = pre.filter_and_reference(raw)
fs = raw.info["sfreq"]
show = ["Fp1", "Cz", "O1", "T7"]
start, stop = int(60 * fs), int(63 * fs)
t = np.arange(stop - start) / fs
fig, axes = plt.subplots(len(show), 2, figsize=(11, 6), sharex=True)
for r, ch in enumerate(show):
    for c, (inst, title) in enumerate([(raw, "raw (as recorded)"), (clean, "cleaned: 2-20 Hz + average reference")]):
        y = inst.get_data(picks=[ch])[0, start:stop] * 1e6
        axes[r, c].plot(t, y, lw=0.8, color="#333333")
        axes[r, c].set_ylabel(f"{ch}\n(µV)")
        if r == 0:
            axes[r, c].set_title(title)
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
save(fig, "raw_vs_clean.png")

# Step 3: power spectrum before / after the 2-20 Hz filter (channel Cz)
fig, ax = plt.subplots(figsize=(8, 4))
for inst, label, col in [(raw, "raw", "#999999"), (clean, "after 2-20 Hz filter", "#2a6fdb")]:
    f, p = welch(inst.get_data(picks=["Cz"])[0] * 1e6, fs=fs, nperseg=int(4 * fs))
    ax.semilogy(f, p, label=label, color=col)
ax.axvspan(2, 20, color="#2a6fdb", alpha=0.08)
ax.set_xlim(0, 60)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power (µV²/Hz, log scale)")
ax.set_title("Cz, subject h01: what the band-pass filter keeps (shaded 2-20 Hz)")
ax.legend()
save(fig, "psd_before_after.png")

# Step 4: epoch rejection (robust z of log-variance) for a subject with rejections
r2 = pre.load_clean(REJECT_SUBJECT)
ep = pre.cut_epochs(r2)
logvar = np.log(ep.get_data().var(axis=(1, 2)) + 1e-30)
med = np.median(logvar)
mad = np.median(np.abs(logvar - med)) + 1e-12
z = 0.6745 * (logvar - med) / mad
bad = np.abs(z) > pre.REJECT_Z
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.scatter(np.arange(len(z))[~bad], z[~bad], s=14, color="#333333", label=f"kept ({(~bad).sum()})")
ax.scatter(np.arange(len(z))[bad], z[bad], s=30, color="#d62728", label=f"rejected ({bad.sum()})")
for v in (3.5, -3.5):
    ax.axhline(v, color="#d62728", ls="--", lw=1)
ax.set_xlabel("epoch number (each = 5 s)")
ax.set_ylabel("robust z of log-variance")
ax.set_title(f"{REJECT_SUBJECT[:3]}: epochs outside ±3.5 are removed")
ax.legend(loc="upper left")
save(fig, "epoch_rejection.png")

# Step 5: GFP peaks of h01 -> the 5% cut -> 1,500 random peaks
from pycrostates.preprocessing import extract_gfp_peaks
pk =extract_gfp_peaks(clean, min_peak_distance=2, verbose="ERROR")
gfp_all = pk.get_data().std(axis=0) * 1e6
cut = np.percentile(gfp_all, 95)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gfp_all, bins=80, color="#888888")
ax.axvline(cut, color="#d62728", ls="--")
ax.text(cut, ax.get_ylim()[1] * 0.9, "  loudest 5% dropped", color="#d62728")
ax.set_xlabel("GFP at the peak (µV)")
ax.set_ylabel("number of peaks")
ax.set_title(f"h01: {len(gfp_all):,} GFP peaks in the whole recording -> 1,500 picked at random")
save(fig, "peak_histogram.png")
print("h01 total peaks", len(gfp_all), "duration (s)", clean.n_times / fs)

# Step 6: labelling one epoch, without and with smoothing
eps = pre.cut_epochs(clean)
one = eps[EPOCH_INDEX]
x = one.get_data()[0]                                            # 19 x 1250
gfp = x.std(axis=0) * 1e6
seg0 = model.predict(one, factor=0, reject_edges=False, verbose="ERROR")
seg5 = model.predict(one, factor=mm.SMOOTH_FACTOR, half_window_size=mm.SMOOTH_HALF_WINDOW,
                     reject_edges=True, verbose="ERROR")
lab0 = np.asarray(seg0.labels).reshape(-1)
lab5 = np.asarray(seg5.labels).reshape(-1)
show_n = int(2 * fs)                                             # first 2 s
tt = np.arange(show_n) / fs * 1000
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})
axes[0].plot(tt, gfp[:show_n], color="#333333", lw=1)
axes[0].set_ylabel("GFP (µV)")
axes[0].set_title("one 5-s epoch of h01 (first 2 s shown): every sample gets the closest map")
for ax, lab, name in [(axes[1], lab0, "no smoothing"), (axes[2], lab5, "smoothed + edges removed (used)")]:
    for i in range(show_n):
        key = letters[lab[i]] if lab[i] >= 0 else "-"
        ax.axvspan(tt[i], tt[i] + 4, color=COLORS[key], lw=0)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center")
axes[2].set_xlabel("time (ms)")
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in "ABCD-"]
axes[0].legend(handles, ["A", "B", "C", "D", "unlabelled edge"], ncol=5, loc="upper right", fontsize=9)
save(fig, "backfitting_epoch.png")
n_seg0 = int((np.diff(lab0[:show_n]) != 0).sum() + 1)
n_seg5 = int((np.diff(lab5[:show_n]) != 0).sum() + 1)
print("segments in 2 s: unsmoothed", n_seg0, "smoothed", n_seg5)

# Step 7: how one sample is assigned: |correlation| with each of the 4 maps
s = int(np.argmax(gfp[200:400]) + 200)                          # a GFP peak inside the epoch
v = x[:, s]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


rs = [corr(v, c) for c in centers]
fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
mne.viz.plot_topomap(v, clean.info, axes=axes[0], show=False, contours=0)
axes[0].set_title(f"the sample\n(t = {s / fs * 1000:.0f} ms)")
order = [0, 2, 3, 1]                                             # A, B, C, D
for ax, k in zip(axes[1:], order):
    mne.viz.plot_topomap(centers[k], model.info, axes=ax, show=False, contours=0)
    best = " <- closest" if k == int(np.argmax(np.abs(rs))) else ""
    ax.set_title(f"map {letters[k]}\ncorr = {rs[k]:+.2f}{best}", fontsize=10)
save(fig, "assign_one_sample.png")
print("sample corrs", dict(zip(letters, np.round(rs, 3))))

# Step 8: polarity - a map and its sign-flipped copy are the same state
k = 3                                                            # state 3 = C
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
mne.viz.plot_topomap(centers[k], model.info, axes=axes[0], show=False, contours=0)
mne.viz.plot_topomap(-centers[k], model.info, axes=axes[1], show=False, contours=0)
axes[0].set_title("map C")
axes[1].set_title("map C × (-1)")
save(fig, "polarity.png")

# Step 9: the 4 maps with electrode names (shared model)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, k in zip(axes, order):
    mne.viz.plot_topomap(centers[k], model.info, axes=ax, show=False, contours=0,
                         names=model.info["ch_names"], sensors=True)
    ax.set_title(f"{letters[k]}  (state {k})")
save(fig, "maps_with_names.png")
print("done")
