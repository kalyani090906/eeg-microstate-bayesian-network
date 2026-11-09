"""
Baseline check: the same 11 classifiers on ALL 24 microstate features first, then on the
reduced 16, side by side.

  24 features = 4 durations + 4 occurrences + 4 coverages + 12 transition probabilities
                (outputs/eeg_features_epoch_mean_15min.csv, straight from extract_epoch_features.py)
  16 features = the 24 minus the 4 coverages and minus 1 transition per starting state
                (outputs/features_final_16.csv, from make_final_features.py; chosen without labels)

Everything is scored with the shared leave-one-subject-out evaluate(), with and without s14,
and every row has both test accuracy (the result) and training accuracy (shows overfitting).
The classifiers are exactly the ones in labs/lab09_a3_classifiers.py (project item 1).

Output: outputs/items/baseline_24_vs_16.csv

Run:  eeg_env\\Scripts\\python.exe scripts\\baseline_24_vs_16.py      (about 5 minutes)
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_24 = OUTPUT_DIR / "eeg_features_epoch_mean_15min.csv"
TABLE_16 = OUTPUT_DIR / "features_final_16.csv"
FEATURE_LIST_16 = OUTPUT_DIR / "features_final_16_list.txt"
RESULT_FILE = OUTPUT_DIR / "items" / "baseline_24_vs_16.csv"
SEED = 42
# --------------------------------------------------------------

import time
import warnings

import pandas as pd

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "labs"))
from evaluate import evaluate, load_feature_names  # noqa: E402
from lab_common import all_metrics                  # noqa: E402
from lab09_a3_classifiers import MODELS             # noqa: E402  (the 11 classifiers of item 1)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Step 1: the two feature sets. The 24 are every column except subject and label.
    t24, t16 = pd.read_csv(TABLE_24), pd.read_csv(TABLE_16)
    feature_sets = {
        "24 (all)": (t24, [c for c in t24.columns if c not in ("subject", "label")]),
        "16 (reduced)": (t16, load_feature_names(FEATURE_LIST_16)),
    }
    assert len(feature_sets["24 (all)"][1]) == 24 and len(feature_sets["16 (reduced)"][1]) == 16

    rows = []
    for table_name in ["with s14", "without s14"]:
        print(f"\n=== {table_name} ===")
        for set_name, (table, cols) in feature_sets.items():
            # Step 2: the same people in both sets (drop s14 for the sensitivity table)
            df = table if table_name == "with s14" else table[table["subject"] != "s14"]
            for name, factory in MODELS.items():
                # Step 3: leave-one-subject-out score (scaling learned inside each fold)
                start = time.perf_counter()
                result = evaluate(factory, df, cols)
                row = {"table": table_name, "feature_set": set_name, "model": name,
                       **all_metrics(result), "seconds": time.perf_counter() - start}
                rows.append(row)
                print(f"  {set_name:13s} {name:20s} acc {row['accuracy']:.1%} "
                      f"[{row['ci_low']:.0%}-{row['ci_high']:.0%}]  train {row['train_accuracy']:.1%}  "
                      f"AUC {row['auc']:.2f}")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULT_FILE, index=False)

    # Step 4: side-by-side view of test accuracy (24 vs 16)
    for table_name in ["with s14", "without s14"]:
        part = res[res.table == table_name].pivot(index="model", columns="feature_set", values="accuracy")
        part["16 minus 24"] = part["16 (reduced)"] - part["24 (all)"]
        print(f"\nTest accuracy, {table_name}:\n{(part * 100).round(1).to_string()}")
    print(f"\nSaved: {RESULT_FILE}")
    print("Exploratory: 28 subjects, one subject = about 3.6 points; differences of 1-2 subjects are noise.")
