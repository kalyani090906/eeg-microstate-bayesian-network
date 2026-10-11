import os
import numpy as np
import mne

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
subjects = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".edf")])

rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

print(f"{'Subject':10} {'Flat channels':30} {'Noisy channels':30} {'Low-corr channels'}")
print("-" * 110)

for fname in subjects:
    path = os.path.join(DATA_DIR, fname)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels(rename_map)
    raw.set_montage("standard_1020", on_missing="warn")
    raw.filter(1, 40, verbose="ERROR")

    data = raw.get_data()  # shape: (n_channels, n_samples)
    ch_names = raw.ch_names

    # 1. Flat/dead channels: very low variance compared to the rest
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    flat_channels = [ch_names[i] for i, v in enumerate(variances) if v < 0.05 * median_var]

    # 2. Noisy channels: variance is a huge outlier vs the rest (robust z-score via MAD)
    mad = np.median(np.abs(variances - median_var)) + 1e-12
    robust_z = 0.6745 * (variances - median_var) / mad
    noisy_channels = [ch_names[i] for i, z in enumerate(robust_z) if z > 5]

    # 3. Low-correlation channels: a channel that barely correlates with
    # any other channel is likely a bad connection, not real brain signal
    corr_matrix = np.corrcoef(data)
    np.fill_diagonal(corr_matrix, np.nan)
    max_corr_per_channel = np.nanmax(np.abs(corr_matrix), axis=1)
    low_corr_channels = [ch_names[i] for i, c in enumerate(max_corr_per_channel) if c < 0.3]

    subj_id = fname.replace(".edf", "")
    print(f"{subj_id:10} {str(flat_channels):30} {str(noisy_channels):30} {str(low_corr_channels)}")