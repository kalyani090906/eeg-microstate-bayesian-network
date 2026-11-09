"""
microstate_model.py - the shared 4-state microstate model: GFP peaks, fitting, labelling.

  GFP(t)   global field power = spread of the 19 channel values at sample t:
             GFP(t) = sqrt( (1/N) * sum_i (V_i(t) - mean_V(t))^2 ),  N = 19 channels.
           With average reference mean_V(t) = 0, so GFP(t) is the standard deviation across
           channels. At GFP PEAKS the map is strongest and most stable, so only peaks are clustered.
  Fitting  modified k-means on the pooled peak maps of all 28 subjects (1,500 per subject, the
           loudest 5% dropped first). "Modified" = polarity is ignored: a map and its sign-flipped
           copy belong to the same state, because the EEG oscillates and the same brain source
           shows up with both signs. Each map goes to the template with the largest |correlation|;
           each template is then recomputed as the main direction (first eigenvector) of its maps.
           Repeated n_init times from random starts; the run with the highest GEV is kept.
  GEV      global explained variance = how much of the signal the templates explain:
             GEV = sum_t ( GFP(t) * corr(V(t), template(t)) )^2 / sum_t GFP(t)^2
  Labelling (backfitting) every sample of every epoch gets the template with the largest
           |spatial correlation|, then temporal smoothing (Pascual-Marqui et al. 1995, factor 5,
           window 2*5+1 = 11 samples = 44 ms) removes one-sample flickers. The first and last
           segment of each epoch are set to -1 (cut by the epoch border, length unknown).

No labels (diagnosis) are used anywhere here.
Nothing to run here: tune_microstates_v3.py, extract_epoch_features.py etc. import it.
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "outputs" / "shared_microstate_model_v3.pkl"
N_STATES = 4
DROP_TOP_PERCENT = 5        # drop the loudest 5% of GFP peaks per subject (likely artefacts)
PEAKS_PER_SUBJECT = 1500    # same number from everybody, so no subject dominates the maps
N_INIT = 50                 # random restarts of modified k-means
SMOOTH_FACTOR = 5           # labelling: smoothing strength (0 = no smoothing)
SMOOTH_HALF_WINDOW = 5      # labelling: half window in samples (window = 2*5+1 = 11 samples)
STATE_NAMES = {0: "A", 1: "D", 2: "B", 3: "C"}   # matched by map shape (name_states.py)
# --------------------------------------------------------------

import pickle

import numpy as np
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks


def global_field_power(data):
    # GFP of every sample; data = (channels, samples), average-referenced
    return data.std(axis=0)


def sample_gfp_peaks(raw, rng):
    # Step 1: the GFP peaks of one cleaned recording -> drop the loudest 5% -> pick 1,500 at random
    peaks = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
    data = peaks.get_data()
    gfp = global_field_power(data)
    data = data[:, gfp <= np.percentile(gfp, 100 - DROP_TOP_PERCENT)]
    n = min(PEAKS_PER_SUBJECT, data.shape[1])
    return data[:, rng.choice(data.shape[1], size=n, replace=False)], peaks.info


def fit_shared_model(peak_arrays, info, n_states=N_STATES, n_init=N_INIT, seed=42):
    # Step 2: modified k-means on the pooled peaks of all subjects
    pooled = ChData(np.concatenate(peak_arrays, axis=1), info)
    model = ModKMeans(n_clusters=n_states, n_init=n_init, random_state=seed)
    model.fit(pooled, n_jobs=1, verbose="ERROR")
    return model


def save_model(model, path=MODEL_PATH):
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_model(path=MODEL_PATH):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def suggest_state_names(model):
    # Which state is the classic A, B, C or D? Each class has a typical SHAPE over the scalp:
    #   A left-back <-> right-front diagonal     B left-front <-> right-back diagonal
    #   C front <-> back axis                    D centre-front peak vs the edges
    # Build these idealised shapes from the electrode positions, correlate each of our maps with
    # them (absolute value: polarity is ignored) and pick the best one-to-one assignment.
    # Returns names {state: letter}, the score table (states x classes) and each state's margin
    # over its second-best class (a small margin = an uncertain name).
    import mne
    from scipy.optimize import linear_sum_assignment

    centers = np.asarray(model.cluster_centers_)
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in model.info["ch_names"]])
    x, y = xyz[:, 0] / np.abs(xyz[:, 0]).max(), xyz[:, 1] / np.abs(xyz[:, 1]).max()
    shapes = {"A": x + y, "B": -x + y, "C": y, "D": -(x ** 2 + (y - 0.2) ** 2)}
    classes = list(shapes)

    def unit(v):
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-12)

    score = np.abs(np.array([unit(c) for c in centers]) @ np.array([unit(shapes[n]) for n in classes]).T)
    rows, cols = linear_sum_assignment(-score)
    names = {int(s): classes[j] for s, j in zip(rows, cols)}
    margins = {int(s): score[s, j] - np.sort(score[s])[-2] for s, j in zip(rows, cols)}
    return names, score, classes, margins


def label_epoch(model, epoch):
    # Step 3: backfitting - one state per sample of a ONE-epoch Epochs object (-1 = unlabeled edge)
    seg = model.predict(epoch, factor=SMOOTH_FACTOR, half_window_size=SMOOTH_HALF_WINDOW,
                        reject_edges=True, verbose="ERROR")
    return seg, np.asarray(seg.labels).reshape(-1)
