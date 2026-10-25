"""
Project item 4: feature reduction and selection.

Compares, with leave-one-subject-out and with/without s14:
  (a) all 24 features   (eeg_features_epoch_mean_15min.csv)
  (b) the 16 features   (24 minus coverages and the 4 dependent transitions; chosen WITHOUT labels)
  (c) top-k of the 16, k = 4, 8, 12, chosen INSIDE each training fold by ANOVA F-score
  (d) L1 logistic regression selection inside each training fold (keeps features with weight != 0)
for two classifiers (logistic regression, and AdaBoost - the best candidate from Lab 9 A3),
reporting accuracy, confidence interval, other metrics and run time.
It also counts how often each feature is chosen across the folds (top-4), to see which are stable.

Run:  eeg_env\\Scripts\\python.exe scripts\\item4_feature_reduction.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "items"
TABLE_16 = OUTPUT_DIR / "features_final_16.csv"
TABLE_24 = OUTPUT_DIR / "eeg_features_epoch_mean_15min.csv"
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
TOP_K = [4, 8, 12]
SEED = 42
# --------------------------------------------------------------

import time
import warnings

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "labs"))
from evaluate import evaluate, load_feature_names  # noqa: E402
from lab_common import all_metrics                  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIERS = {
    "Logistic regression": lambda: LogisticRegression(max_iter=5000, random_state=SEED),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED),
}


def plain(make_clf):
    # scale -> classify (all given features)
    return lambda: Pipeline([("scale", StandardScaler()), ("clf", make_clf())])


def top_k(make_clf, k):
    # scale -> keep the k best features by ANOVA F (learned on the training fold) -> classify
    return lambda: Pipeline([("scale", StandardScaler()),
                             ("select", SelectKBest(f_classif, k=k)),
                             ("clf", make_clf())])


def l1_select(make_clf):
    # scale -> keep features an L1 logistic regression gives non-zero weight -> classify
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=SEED)
    return lambda: Pipeline([("scale", StandardScaler()),
                             ("select", SelectFromModel(l1)),
                             ("clf", make_clf())])


if __name__ == "__main__":
    # Step 1: load both tables; the 24-feature list is every column except subject/label
    df16_all = pd.read_csv(TABLE_16)
    df24_all = pd.read_csv(TABLE_24)
    f16 = load_feature_names(FEATURE_LIST)
    f24 = [c for c in df24_all.columns if c not in ("subject", "label")]
    print(f"{len(f24)} features in the full table, {len(f16)} in the reduced table")

    rows = []
    for table_name, keep in [("with s14", lambda d: d),
                             ("without s14", lambda d: d[d["subject"] != "s14"])]:
        df16, df24 = keep(df16_all), keep(df24_all)
        print(f"\n=== {table_name}: {len(df16)} subjects ===")
        for clf_name, make_clf in CLASSIFIERS.items():
            # Step 2: the feature sets to compare (name, table, columns, model)
            setups = [("all 24", df24, f24, plain(make_clf)),
                      ("16 (structure-based)", df16, f16, plain(make_clf))]
            setups += [(f"top-{k} of 16 (ANOVA, in fold)", df16, f16, top_k(make_clf, k)) for k in TOP_K]
            setups += [("L1 selection of 16 (in fold)", df16, f16, l1_select(make_clf))]

            for set_name, df, cols, factory in setups:
                # Step 3: leave-one-subject-out score, timed (pipelines scale themselves)
                start = time.perf_counter()
                result = evaluate(factory, df, cols, scale=False)
                seconds = time.perf_counter() - start
                row = {"table": table_name, "classifier": clf_name, "feature_set": set_name,
                       **all_metrics(result), "seconds": seconds}
                rows.append(row)
                print(f"  {clf_name:20s} {set_name:30s} acc {row['accuracy']:.1%} "
                      f"[{row['ci_low']:.0%}-{row['ci_high']:.0%}]  AUC {row['auc']:.2f}  ({seconds:.1f}s)")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS_DIR / "item4_feature_reduction.csv", index=False)

    # Step 4: which features does top-4 pick in each training fold? (main table)
    counts = pd.Series(0, index=f16)
    for subject in df16_all["subject"]:
        train = df16_all[df16_all["subject"] != subject]
        selector = SelectKBest(f_classif, k=4).fit(
            StandardScaler().fit_transform(train[f16]), train["label"])
        counts[[f for f, keep in zip(f16, selector.get_support()) if keep]] += 1
    counts = counts.sort_values(ascending=False)
    counts.to_csv(RESULTS_DIR / "item4_top4_selection_counts.csv", header=["folds_selected"])
    print(f"\nTop-4 selection across {len(df16_all)} training folds (main table):")
    for f, c in counts[counts > 0].items():
        print(f"  {f:20s} chosen in {c:2d} / {len(df16_all)} folds")

    print(f"\nSaved in {RESULTS_DIR}")
    print("24 -> 16 used no labels; top-k and L1 use labels, so they are fitted inside each fold only.")
