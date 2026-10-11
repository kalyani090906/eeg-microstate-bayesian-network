import os
import pickle
import numpy as np
import mne
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

subjects = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".edf")])
print(f"Found {len(subjects)} subjects: {subjects}")

all_peak_arrays = []
shared_info = None

for fname in subjects:
    path = os.path.join(DATA_DIR, fname)

    raw = mne.io.read_raw_edf(
        path,
        preload=True,
        verbose="ERROR"
    )

    # Rename old electrode names to modern names
    rename_map = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8"
    }

    raw.rename_channels(rename_map)

    # Tell MNE where the electrodes are located
    raw.set_montage("standard_1020", on_missing="warn")

    # Mark and interpolate known bad channels for specific subjects
    bads_by_subject = {
        "s07.edf": ["C4"],
        "h14.edf": ["F7"],
        "s11.edf": ["F7"],
        "s12.edf": ["P7"],
    }
    if fname in bads_by_subject:
        raw.info["bads"] = bads_by_subject[fname]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        
    # Continue with preprocessing
    raw.filter(1, 40, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")

    peaks = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
    data = peaks.get_data()  # shape: (n_channels, n_peaks)

    all_peak_arrays.append(data)
    if shared_info is None:
        shared_info = peaks.info  # same 19-channel layout for every subject

    print(f"{fname}: {data.shape[1]} GFP peaks extracted")

# Pool every subject's GFP peaks into one shared dataset
pooled_data = np.concatenate(all_peak_arrays, axis=1)
pooled_peaks = ChData(pooled_data, shared_info)
print(f"\nTotal pooled peaks across all subjects: {pooled_data.shape[1]}")

# Fit ONE shared 4-microstate model across the whole group
model = ModKMeans(n_clusters=4, n_init=100, random_state=42)
model.fit(pooled_peaks, n_jobs=1, verbose="ERROR")

print("\nShared microstate model fitted.")
print("Cluster centers shape:", model.cluster_centers_.shape)

# Save the fitted model so every later script can reuse it
model_path = os.path.join(OUTPUT_DIR, "shared_microstate_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to: {model_path}")