import os
import pickle
import numpy as np
import pandas as pd
import mne

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# Load the shared microstate model fit across all 28 subjects
model_path = os.path.join(OUTPUT_DIR, "shared_microstate_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
bads_by_subject = {
    "s07.edf": ["C4"],
    "h14.edf": ["F7"],
    "s11.edf": ["F7"],
    "s12.edf": ["P7"],
}

subjects = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".edf")])
rows = []

for fname in subjects:
    path = os.path.join(DATA_DIR, fname)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels(rename_map)
    raw.set_montage("standard_1020", on_missing="warn")

    if fname in bads_by_subject:
        raw.info["bads"] = bads_by_subject[fname]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")

    raw.filter(1, 40, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")

    # Back-fit: label every timepoint against the SHARED 4-state model
    segmentation = model.predict(raw, factor=10, half_window_size=10, verbose="ERROR")

    # Built-in duration / occurrence / coverage per state
    params = segmentation.compute_parameters()

    # Built-in transition probability matrix (4x4)
    trans_matrix = segmentation.compute_transition_matrix(stat="probability")

    subj_id = fname.replace(".edf", "")
    label = 1 if subj_id.startswith("s") else 0  # 1 = schizophrenia, 0 = healthy

    row = {"subject": subj_id, "label": label}

    # Add duration, occurrence, coverage for each of the 4 states
    for state in range(4):
        row[f"state{state}_duration"] = params.get(f"{state}_meandurs", np.nan)
        row[f"state{state}_occurrence"] = params.get(f"{state}_occurrences", np.nan)
        row[f"state{state}_coverage"] = params.get(f"{state}_timecov", np.nan)

    # Add all 12 off-diagonal transition probabilities
    for i in range(4):
        for j in range(4):
            if i != j:
                row[f"trans_{i}to{j}"] = trans_matrix[i, j]

    rows.append(row)
    print(f"{subj_id}: features extracted ({len(row)-2} features)")

df = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, "eeg_features.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved feature table to: {out_path}")
print(f"Shape: {df.shape}")
print(df.head())