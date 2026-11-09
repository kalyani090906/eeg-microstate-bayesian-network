"""
check_bad_channels.py - find broken electrodes in each recording (prints a table only).

For every recording, filtered 1-40 Hz, three label-free checks per channel:
  1. flat:            variance below 5% of the median channel variance (dead electrode)
  2. noisy:           robust z-score of the variance above 5 (median / MAD), far louder than the rest
  3. low correlation: its best |correlation| with any other channel is below 0.3
                      (neighbouring electrodes always share signal, so this means a bad contact)
The channels found here became the BADS list in eeg_preprocessing.py
(s07 C4, h14 F7, s11 F7, s12 P7), which every later script rebuilds from its neighbours.

Run:  eeg_env\\Scripts\\python.exe scripts\\check_bad_channels.py
"""

# ---------------- settings (change only here) ----------------
CHECK_BAND = (1, 40)          # Hz
FLAT_FRACTION = 0.05          # flat if variance < 5% of the median
NOISY_Z = 5                   # noisy if robust z of variance > 5
MIN_CORR = 0.3                # bad contact if best |correlation| with any channel < 0.3
# --------------------------------------------------------------

import numpy as np

import eeg_preprocessing as pre

print(f"{'Subject':10} {'Flat channels':30} {'Noisy channels':30} {'Low-corr channels'}")
print("-" * 110)

for fname in pre.recording_files():
    # Step 1: read and rename, WITHOUT rebuilding channels (we are looking for the bad ones)
    raw = pre.load_raw(fname, interpolate=False)
    raw.filter(*CHECK_BAND, verbose="ERROR")
    data = raw.get_data()  # shape: (n_channels, n_samples)
    ch_names = raw.ch_names

    # Step 2: flat / dead channels: very low variance compared to the rest
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    flat_channels = [ch_names[i] for i, v in enumerate(variances) if v < FLAT_FRACTION * median_var]

    # Step 3: noisy channels: variance is a huge outlier vs the rest (robust z-score via MAD)
    mad = np.median(np.abs(variances - median_var)) + 1e-12
    robust_z = 0.6745 * (variances - median_var) / mad
    noisy_channels = [ch_names[i] for i, z in enumerate(robust_z) if z > NOISY_Z]

    # Step 4: low-correlation channels: barely related to any other channel = bad connection
    corr_matrix = np.corrcoef(data)
    np.fill_diagonal(corr_matrix, np.nan)
    max_corr_per_channel = np.nanmax(np.abs(corr_matrix), axis=1)
    low_corr_channels = [ch_names[i] for i, c in enumerate(max_corr_per_channel) if c < MIN_CORR]

    subject, _ = pre.subject_and_label(fname)
    print(f"{subject:10} {str(flat_channels):30} {str(noisy_channels):30} {str(low_corr_channels)}")
