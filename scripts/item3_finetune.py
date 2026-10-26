"""
Project item 3: deep model with transfer learning (fine-tuning LaBraM).

Item 2 used LaBraM frozen. Here it is ADAPTED to our task: the last TRAINABLE_BLOCKS of its 12
transformer layers and a small classification head are trained on our epochs; the earlier
layers keep their pretrained weights (frozen). This is transfer learning, not a model trained
from scratch.

Protocol (different from the leave-one-subject-out used elsewhere - label it in the report):
  - 7-fold subject-grouped split (StratifiedGroupKFold): all epochs of a subject are on the same
    side; each test fold holds about 4 subjects (balanced between the groups where possible).
  - Training settings are FIXED in advance (below). Nothing is tuned on the test subjects.
  - No augmentation, no overlapping windows.
  - Each test subject's epoch probabilities are averaged into one prediction (never epoch accuracy).
Each finished fold is saved immediately; if the run stops, start it again and it carries on.

Run (about 1-2 hours on CPU for both tables):
    eeg_env\\Scripts\\python.exe scripts\\item3_finetune.py
Quick check (1 fold, 1 pass, few epochs - only to see that it runs; results meaningless):
    eeg_env\\Scripts\\python.exe scripts\\item3_finetune.py --smoke
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "items" / "item3_finetune"
EPOCHS_FILE = OUTPUT_DIR / "epochs_15min_labram.npz"
N_FOLDS = 7
TRAINABLE_BLOCKS = 2        # last 2 of 12 transformer layers are trained, the rest stay frozen
PASSES = 5                  # training passes over the training epochs (fixed, not tuned)
BATCH = 32
LR_BACKBONE = 1e-4          # learning rate of the unfrozen LaBraM layers
LR_HEAD = 1e-3              # learning rate of the new classification head
WEIGHT_DECAY = 0.05
DROPOUT = 0.5
SEED = 42
# --------------------------------------------------------------

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from evaluate import wilson_ci                                   # noqa: E402
from item2_embeddings import MODEL_SFREQ, UNIT_SCALE, load_labram  # noqa: E402

warnings.filterwarnings("ignore")


class FineTuner(nn.Module):
    # LaBraM backbone -> mean of its patch tokens -> norm -> dropout -> 1 output (patient score)
    def __init__(self, n_times, ch_names):
        super().__init__()
        self.backbone = load_labram(n_times)
        self.ch_names = ch_names
        self.head = nn.Sequential(nn.LayerNorm(200), nn.Dropout(DROPOUT), nn.Linear(200, 1))
        # freeze everything, then unfreeze the last TRAINABLE_BLOCKS layers and the final norm
        last = {f"blocks.{i}." for i in range(12 - TRAINABLE_BLOCKS, 12)}
        for name, p in self.backbone.named_parameters():
            p.requires_grad = any(name.startswith(b) for b in last) or name.startswith("norm.")

    def forward(self, x):
        feats = self.backbone(x, return_features=True, ch_names=self.ch_names)["features"]
        return self.head(feats.mean(dim=1)).squeeze(1)


def train_and_predict(X, y, train_idx, test_idx, ch_names, fold_seed, passes):
    # Step 3: build a fresh model for this fold and train it on the training subjects only
    torch.manual_seed(fold_seed)
    model = FineTuner(X.shape[-1], ch_names)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW([{"params": backbone_params, "lr": LR_BACKBONE},
                                   {"params": model.head.parameters(), "lr": LR_HEAD}],
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(fold_seed)
    losses = []
    for p in range(passes):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b].astype("float32"))
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(b)
        losses.append(total / len(order))
        print(f"    pass {p + 1}/{passes}: training loss {losses[-1]:.4f}", flush=True)

    # Step 4: predict every epoch of the test subjects
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(test_idx), 64):
            probs.append(torch.sigmoid(model(torch.from_numpy(X[test_idx[i:i + 64]]))).numpy())
    return np.concatenate(probs), losses


def run_table(table_name, X, y, subjects, ch_names, args):
    keep = np.ones(len(y), bool) if table_name == "with s14" else subjects != "s14"
    idx_all = np.where(keep)[0]
    out_file = RESULTS_DIR / f"subject_predictions_{table_name.replace(' ', '_')}.csv"
    done = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame(columns=["fold"])

    # Step 2: 7 subject-grouped folds, stratified by diagnosis (fixed by SEED)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(idx_all, y[idx_all], groups=subjects[idx_all]))
    print(f"\n=== {table_name}: {len(np.unique(subjects[idx_all]))} subjects, {len(idx_all)} epochs, "
          f"{N_FOLDS} folds ===")
    for f, (tr, te) in enumerate(folds):
        if f in set(done["fold"]):
            print(f"  fold {f + 1}: already saved, skipped")
            continue
        if args.smoke and f > 0:
            break
        train_idx, test_idx = idx_all[tr], idx_all[te]
        if args.smoke:                                         # tiny run: only checks the code path
            train_idx = np.random.default_rng(0).choice(train_idx, 256, replace=False)
        test_subjects = sorted(str(s) for s in set(subjects[test_idx]))
        print(f"  fold {f + 1}/{N_FOLDS}: test subjects {test_subjects}, "
              f"{len(train_idx)} training epochs", flush=True)
        start = time.perf_counter()
        probs, losses = train_and_predict(X, y, train_idx, test_idx, ch_names, SEED + f,
                                          1 if args.smoke else PASSES)

        # Step 5: average each test subject's epoch probabilities into one prediction
        per_epoch = pd.DataFrame({"subject": subjects[test_idx], "label": y[test_idx], "prob": probs})
        per_subject = per_epoch.groupby(["subject", "label"], as_index=False)["prob"].mean()
        per_subject["predicted"] = (per_subject["prob"] >= 0.5).astype(int)
        per_subject["fold"] = f
        per_subject["final_train_loss"] = losses[-1]
        if not args.smoke:
            per_subject.to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
        print(f"    {per_subject[['subject', 'label', 'prob']].round(3).values.tolist()}  "
              f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

    if args.smoke or not out_file.exists():
        return None
    # Step 6: summary over all saved folds
    res = pd.read_csv(out_file)
    n, k = len(res), int((res["label"] == res["predicted"]).sum())
    lo, hi = wilson_ci(k, n)
    bal = np.mean([np.mean(res[res.label == c].predicted == c) for c in (0, 1)])
    return {"table": table_name, "protocol": f"{N_FOLDS}-fold subject-grouped",
            "folds_done": res["fold"].nunique(), "n_subjects": n, "n_correct": k,
            "accuracy": k / n, "ci_low": lo, "ci_high": hi, "balanced_accuracy": bal,
            "auc": roc_auc_score(res["label"], res["prob"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 fold, 1 pass, 256 epochs: code check only")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # Step 1: load the LaBraM-ready epochs (same 4786 epochs as the feature table)
    d = np.load(EPOCHS_FILE, allow_pickle=True)
    assert float(d["sfreq"]) == MODEL_SFREQ, "run make_epoch_array.py --preset labram first"
    X = (d["X"] * UNIT_SCALE).astype("float32")
    y, subjects = d["label"].astype(int), d["subject"].astype(str)
    ch_names = [str(c) for c in d["ch_names"]]
    (RESULTS_DIR / "settings.json").write_text(json.dumps(
        {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str))},
        indent=2))

    rows = [r for r in (run_table(t, X, y, subjects, ch_names, args)
                        for t in ["with s14", "without s14"]) if r]
    if rows:
        summary = pd.DataFrame(rows)
        summary.round(4).to_csv(RESULTS_DIR / "item3_summary.csv", index=False)
        print("\n" + summary.round(3).to_string(index=False))
        print(f"\nSaved in {RESULTS_DIR}")
        print("Protocol differs from leave-one-subject-out (7 folds of ~4 subjects): say so in the report.")
