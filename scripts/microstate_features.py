"""
microstate_features.py - the microstate features, written out as plain formulas.

Input is the LABEL SEQUENCE of one 5-second epoch: one state number per sample
(0, 1, 2, 3; -1 = unlabeled). With reject_edges the first and last segment of every epoch are
-1, because they are cut by the epoch border and their true length is unknown.

A SEGMENT is a run of consecutive samples with the same state, e.g. 0 0 0 2 2 1 1 1 1 has the
segments (0, 3 samples), (2, 2 samples), (1, 4 samples).

For state k in one epoch (fs = sampling rate = 250 Hz, L = number of labeled samples):
    n_k          = number of segments of state k
    len_k,j      = length of the j-th segment of state k, in samples
    duration_k   = (1 / n_k) * sum_j len_k,j / fs          mean length of one visit, seconds
    occurrence_k = n_k / (L / fs)                            visits per second
    coverage_k   = sum_j len_k,j / L                         share of time in state k (0..1)
  so coverage_k = occurrence_k * duration_k exactly (within an epoch).

Transitions (repetitions removed first, so A A A B B C -> A B C; staying in a state is not a
transition):
    c_ij       = number of times a segment of state i is directly followed by a segment of state j
    P(i -> j)  = c_ij / sum_{m != i} c_im                   (i != j; the 3 values out of i sum to 1)
That is 4 x 3 = 12 transition probabilities. For one SUBJECT the counts c_ij of all kept epochs
are added up first and then divided (pooled), which is more accurate than averaging 170 small
per-epoch probabilities.

These are the same definitions as pycrostates 0.6.1 (Segmentation.compute_parameters and
compute_transition_matrix(stat="count")); check_features_match.py confirms the numbers agree.
Nothing to run here: extract_epoch_features.py imports it.
"""

from itertools import groupby

import numpy as np


def segments(labels):
    # Step 1: cut the label sequence into runs of the same state: [(state, length in samples), ...]
    return [(int(state), len(list(run))) for state, run in groupby(labels)]


def state_parameters(labels, sfreq, n_states):
    # Step 2: duration, occurrence and coverage of every state in ONE epoch (formulas above)
    segs = segments(labels)
    n_labeled = sum(length for state, length in segs if state != -1)        # L
    params = {}
    for k in range(n_states):
        lengths = np.array([length for state, length in segs if state == k])
        if len(lengths) == 0:                     # state never visited in this epoch
            params[k] = {"duration": np.nan, "occurrence": 0.0, "coverage": 0.0}
            continue
        params[k] = {
            "duration": lengths.mean() / sfreq,                  # seconds per visit
            "occurrence": len(lengths) / n_labeled * sfreq,      # visits per second
            "coverage": lengths.sum() / n_labeled,               # share of labeled time
        }
    return params


def transition_counts(labels, n_states):
    # Step 3: c_ij = how often a segment of i is directly followed by a segment of j (i != j)
    sequence = [state for state, _ in segments(labels)]      # repetitions removed
    counts = np.zeros((n_states, n_states))
    for i, j in zip(sequence, sequence[1:]):
        if i != -1 and j != -1:                              # skip the unlabeled edges
            counts[i, j] += 1
    return counts


def transition_probabilities(counts):
    # Step 4: P(i -> j) = c_ij / (all transitions out of i). The diagonal is always 0.
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals > 0, counts / totals, np.nan)


def feature_names(n_states):
    # The 24 column names, in the order of the feature tables
    names = []
    for k in range(n_states):
        names += [f"state{k}_duration", f"state{k}_occurrence", f"state{k}_coverage"]
    names += [f"trans_{i}to{j}" for i in range(n_states) for j in range(n_states) if i != j]
    return names


def epoch_features(labels, sfreq, n_states):
    # Step 5: one row of 24 features for one epoch, plus its transition counts (for pooling)
    params = state_parameters(labels, sfreq, n_states)
    counts = transition_counts(labels, n_states)
    probs = transition_probabilities(counts)
    row = {}
    for k in range(n_states):
        row[f"state{k}_duration"] = params[k]["duration"]
        row[f"state{k}_occurrence"] = params[k]["occurrence"]
        row[f"state{k}_coverage"] = params[k]["coverage"]
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                row[f"trans_{i}to{j}"] = probs[i, j]
    return row, counts


def subject_features(epoch_rows, pooled_counts, n_states, how="mean"):
    # Step 6: one row per subject. Durations, occurrences, coverages: mean (or SD) over the kept
    # epochs, ignoring epochs where a state never appeared. Transitions: from the POOLED counts
    # (how="mean") or the SD of the per-epoch probabilities (how="sd").
    names = feature_names(n_states)
    state_cols = [c for c in names if c.startswith("state")]
    out = {}
    for c in state_cols:
        values = np.array([r[c] for r in epoch_rows], dtype=float)
        out[c] = float(np.nanmean(values) if how == "mean" else np.nanstd(values))
    pooled = transition_probabilities(pooled_counts)
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                c = f"trans_{i}to{j}"
                if how == "mean":
                    out[c] = float(pooled[i, j])
                else:
                    out[c] = float(np.nanstd(np.array([r[c] for r in epoch_rows], dtype=float)))
    return out
