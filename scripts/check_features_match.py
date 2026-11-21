"""
check_features_match.py - do our written-out feature formulas give the same numbers as pycrostates?

microstate_features.py writes duration, occurrence, coverage and the transition counts out line by
line, so every formula can be shown and explained. This script checks them against the library:
for every kept epoch of the chosen subjects it labels the epoch once (microstate_model.label_epoch)
and computes the features twice:
  ours        microstate_features.state_parameters / transition_counts
  pycrostates Segmentation.compute_parameters (meandurs, occurrences, timecov)
              Segmentation.compute_transition_matrix(stat="count")
and prints the largest absolute difference over all epochs. Prints only; writes nothing.

Run:  eeg_env\\Scripts\\python.exe scripts\\check_features_match.py      (about 1 minute)
"""

# ---------------- settings (change only here) ----------------
SUBJECTS = ["h01.edf", "s01.edf"]   # one control, one patient
MAX_EPOCHS_PER_SUBJECT = 180        # 180 + 180 = 360 epochs (fewer if a subject has fewer kept)
# --------------------------------------------------------------

import warnings

import numpy as np

import eeg_preprocessing as pre
import microstate_features as mf
import microstate_model as mm

warnings.filterwarnings("ignore")
model = mm.load_model()
K = model.cluster_centers_.shape[0]
worst = {"duration": 0.0, "occurrence": 0.0, "coverage": 0.0, "transition counts": 0.0}
n_epochs = 0

for f in SUBJECTS:
    # Step 1: the same cleaning, epochs and rejection as extract_epoch_features.py
    raw = pre.load_clean(f)
    epochs = pre.cut_epochs(raw)
    keep = pre.kept_epochs(epochs)[:MAX_EPOCHS_PER_SUBJECT]
    sfreq = raw.info["sfreq"]

    for i in keep:
        # Step 2: label the epoch once; both methods use this same labelling
        seg, labels = mm.label_epoch(model, epochs[i])

        # Step 3: our formulas
        ours = mf.state_parameters(labels, sfreq, K)
        our_counts = mf.transition_counts(labels, K)

        # Step 4: the library (meandurs = duration, occurrences = occurrence, timecov = coverage)
        lib = seg.compute_parameters(norm_gfp=True)
        lib_counts = np.asarray(seg.compute_transition_matrix(stat="count"))

        # Step 5: largest difference so far (a state never visited: our duration is NaN, the
        # library leaves the key out or gives 0, so durations are compared only where both exist)
        for k in range(K):
            pairs = [("duration", f"{k}_meandurs"), ("occurrence", f"{k}_occurrences"), ("coverage", f"{k}_timecov")]
            for name, key in pairs:
                a, b = ours[k][name], lib.get(key, np.nan)
                if np.isnan(a) or np.isnan(b):
                    continue
                worst[name] = max(worst[name], abs(float(a) - float(b)))
        worst["transition counts"] = max(worst["transition counts"], float(np.abs(our_counts - lib_counts).max()))
        n_epochs += 1
    print(f"{f}: {len(keep)} epochs compared", flush=True)

print(f"\nLargest absolute difference over {n_epochs} epochs (0 or about 1e-17 = identical):")
for name, d in worst.items():
    print(f"  {name:18s} {d:.2e}")
