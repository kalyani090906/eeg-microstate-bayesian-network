"""
Project item 2: deep-learning embeddings + ML classifiers.

A pretrained EEG model (LaBraM, Jiang et al., ICLR 2024; weights from Hugging Face
"braindecode/labram-pretrained") turns each 5-second epoch into 200 numbers (an embedding).
The model is FROZEN: it is not trained on our data at all. Ordinary classifiers then learn
patient vs control from those numbers, scored with the same leave-one-subject-out evaluate().

Two ways to classify:
  (a) epoch level  - train on every epoch of the training subjects; the held-out subject's epoch
                     probabilities are averaged into one prediction (never report epoch accuracy)
  (b) subject level - average each subject's embeddings into one row (28 rows), then classify

Input:  outputs/epochs_15min_labram.npz   (make it with: make_epoch_array.py --preset labram)
Output: outputs/embeddings_labram.npz      (4786 x 200, with subject / label / epoch; not for git)
        outputs/items/item2_embeddings.csv (the results table)

Run:  eeg_env\\Scripts\\python.exe scripts\\item2_embeddings.py
Needs: pip install torch --index-url https://download.pytorch.org/whl/cpu
       pip install braindecode huggingface_hub safetensors
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "items"
EPOCHS_FILE = OUTPUT_DIR / "epochs_15min_labram.npz"
EMB_FILE = OUTPUT_DIR / "embeddings_labram.npz"
HF_MODEL = "braindecode/labram-pretrained"
MODEL_SFREQ = 200          # LaBraM works on 200 Hz, 1-second patches of 200 samples
UNIT_SCALE = 1e4           # volts -> units of 0.1 mV (LaBraM divides microvolts by 100)
BATCH = 64
SEED = 42
# --------------------------------------------------------------

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "labs"))
from evaluate import evaluate          # noqa: E402
from lab_common import all_metrics     # noqa: E402

warnings.filterwarnings("ignore")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_labram(n_times):
    """
    Load the pretrained LaBraM for inputs of n_times samples.
    The checkpoint was saved for 15-second inputs (16 time positions). As in the original LaBraM
    code, a shorter input uses the first time positions, so we keep the first ones and copy every
    other weight unchanged (strict loading: every tensor must match).
    """
    import torch
    from braindecode.models import Labram
    full = Labram.from_pretrained(HF_MODEL)
    model = Labram(n_chans=full.n_chans, n_times=n_times, n_outputs=0, chs_info=full.chs_info)
    state = full.state_dict()
    state["temporal_embedding"] = state["temporal_embedding"][:, : model.temporal_embedding.shape[1]]
    model.load_state_dict(state, strict=True)
    torch.manual_seed(SEED)
    return model.eval()


def extract():
    import torch
    from scipy.signal import resample_poly

    # Step 1: load the epochs and bring them to LaBraM's sampling rate and units
    d = np.load(EPOCHS_FILE, allow_pickle=True)
    X, ch_names, sfreq = d["X"], [str(c) for c in d["ch_names"]], float(d["sfreq"])
    print(f"Epochs: {X.shape} at {sfreq:.0f} Hz, channels {ch_names}")
    if sfreq != MODEL_SFREQ:
        print(f"  resampling {sfreq:.0f} -> {MODEL_SFREQ} Hz")
        X = resample_poly(X, MODEL_SFREQ, int(sfreq), axis=-1)
    X = (X * UNIT_SCALE).astype("float32")

    # Step 2: run the frozen model; embedding = mean of the patch tokens (LaBraM's own pooling)
    model = load_labram(X.shape[-1])
    out, start = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            feats = model(torch.from_numpy(X[i:i + BATCH]), return_features=True, ch_names=ch_names)
            out.append(feats["features"].mean(dim=1).numpy())
            print(f"\r  {min(i + BATCH, len(X))}/{len(X)} epochs", end="", flush=True)
    emb = np.concatenate(out)
    print(f"\n  done in {time.perf_counter() - start:.0f} s; embedding size {emb.shape[1]}")

    # Step 3: save with subject / label / epoch so every row can be traced
    np.savez_compressed(EMB_FILE, emb=emb, subject=d["subject"], label=d["label"], epoch=d["epoch"])
    print(f"Saved: {EMB_FILE}")


def classify():
    d = np.load(EMB_FILE, allow_pickle=True)
    cols = [f"e{i}" for i in range(d["emb"].shape[1])]
    epochs = pd.DataFrame(d["emb"], columns=cols)
    epochs.insert(0, "epoch", d["epoch"])
    epochs.insert(0, "label", d["label"].astype(int))
    epochs.insert(0, "subject", d["subject"].astype(str))
    subjects = epochs.groupby(["subject", "label"], as_index=False)[cols].mean()

    # Step 4: classifiers (scaling is added inside every training fold by evaluate)
    epoch_models = {
        "Logistic regression": lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=SEED),
        "Random forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                        n_jobs=-1, random_state=SEED),
    }
    subject_models = {
        "Logistic regression": lambda: LogisticRegression(C=0.1, max_iter=5000, random_state=SEED),
        "SVM (linear)": lambda: CalibratedClassifierCV(SVC(kernel="linear", C=0.1, random_state=SEED),
                                                       ensemble=False),
        "kNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
        "Random forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),
    }

    rows = []
    for table_name, keep in [("with s14", lambda t: t), ("without s14", lambda t: t[t.subject != "s14"])]:
        print(f"\n=== {table_name} ===")
        for level, table, models in [("epoch level", keep(epochs), epoch_models),
                                     ("subject mean", keep(subjects), subject_models)]:
            for name, factory in models.items():
                # Step 5: leave-one-subject-out, one prediction per subject
                start = time.perf_counter()
                result = evaluate(factory, table, cols)
                row = {"table": table_name, "level": level, "model": name, **all_metrics(result),
                       "seconds": time.perf_counter() - start}
                rows.append(row)
                print(f"  {level:13s} {name:20s} acc {row['accuracy']:.1%} "
                      f"[{row['ci_low']:.0%}-{row['ci_high']:.0%}]  AUC {row['auc']:.2f}  "
                      f"({row['seconds']:.0f}s)")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS_DIR / "item2_embeddings.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'item2_embeddings.csv'}")
    print("Compare with the 16 microstate features (outputs/labs/lab09_a3_classifier_table.csv).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-extract even if embeddings exist")
    args = parser.parse_args()
    if args.force or not EMB_FILE.exists():
        extract()
    else:
        print(f"Using saved embeddings: {EMB_FILE} (add --force to re-extract)")
    classify()
