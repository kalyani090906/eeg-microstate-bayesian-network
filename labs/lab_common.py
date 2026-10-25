"""
lab_common.py  -  settings and helpers shared by every lab script in this folder.

Nothing to run here. The lab scripts do:  from lab_common import ...
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path
import sys

LABS_DIR = Path(__file__).resolve().parent            # MLproject/labs
PROJECT_DIR = LABS_DIR.parent                         # MLproject
OUTPUT_DIR = PROJECT_DIR / "outputs"
LAB_OUT = OUTPUT_DIR / "labs"                         # every lab result goes here
TABLES = {                                            # main table and the sensitivity table
    "with s14": OUTPUT_DIR / "features_final_16.csv",
    "without s14": OUTPUT_DIR / "features_final_16_no_s14.csv",
}
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
SEED = 42
# --------------------------------------------------------------

import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Make scripts/evaluate.py importable from this folder
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from evaluate import evaluate, load_feature_names, load_table  # noqa: E402

LAB_OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # small data: MLP/LogReg warnings are noise
warnings.filterwarnings("ignore", category=UserWarning)


def all_metrics(result):
    # Turn one evaluate() result into one row of metrics (patient = positive class)
    ps = result["per_subject"]
    y, pred, prob = ps["true_label"], ps["predicted"], ps["prob_patient"]
    return {
        "n_subjects": result["n_subjects"],
        "accuracy": result["accuracy"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "balanced_accuracy": result["balanced_accuracy"],
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
    }


def run_models(models, scale=True, features=None, tables=TABLES):
    """
    models:   dict  name -> function that returns a NEW untrained model
    scale:    True puts a StandardScaler in front (set False if the model is already a full pipeline)
    features: list of columns; default = the 16 names from the list file
    Runs every model on both tables with leave-one-subject-out and returns one results table.
    """
    features = features or load_feature_names(FEATURE_LIST)
    rows = []
    for table_name, path in tables.items():
        df = load_table(path)
        print(f"\n=== {table_name}: {len(df)} subjects, {len(features)} features ===")
        for name, factory in models.items():
            # Step 1: score the model with leave-one-subject-out, and time it
            start = time.perf_counter()
            result = evaluate(factory, df, features, scale=scale)
            seconds = time.perf_counter() - start
            # Step 2: collect every metric in one row
            row = {"table": table_name, "model": name, **all_metrics(result), "seconds": seconds}
            rows.append(row)
            print(f"  {name:26s} acc {row['accuracy']:.1%}  "
                  f"[{row['ci_low']:.0%}-{row['ci_high']:.0%}]  "
                  f"bal {row['balanced_accuracy']:.1%}  F1 {row['f1']:.2f}  "
                  f"AUC {row['auc']:.2f}  ({seconds:.1f}s)")
    return pd.DataFrame(rows)


def save(df, filename):
    # Save a results table to outputs/labs/ with 3 decimals
    path = LAB_OUT / filename
    df.round(3).to_csv(path, index=False)
    print(f"\nSaved: {path}")
    print("Exploratory: 28 subjects, one subject = about 3.6 points; chance = 50%.")
