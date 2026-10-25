"""
evaluate.py  -  one shared way to score ANY classifier on our data.

What it does, in plain words:
  - Leave-one-subject-out: train on all subjects except one, test on that one,
    repeat until every subject has been the test subject once.
  - Scaling is learned inside each training fold only (no leakage).
  - Works on the 28-row subject table AND on tables with many rows per subject
    (epoch features, embeddings): rows of one subject always stay together, and
    the epoch probabilities are averaged into one prediction per subject.
  - Reports accuracy, balanced accuracy and a 95% confidence interval.

Run it directly to get the first baseline (logistic regression, with and without s14):
    eeg_env\\Scripts\\python.exe scripts\\evaluate.py
Other scripts can import it:
    from evaluate import evaluate, load_table, load_feature_names
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent      # MLproject folder, found automatically
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_WITH_S14 = OUTPUT_DIR / "features_final_16.csv"
TABLE_WITHOUT_S14 = OUTPUT_DIR / "features_final_16_no_s14.csv"
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
RESULTS_FILE = OUTPUT_DIR / "baseline_logreg_results.csv"
SEED = 42
# --------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_feature_names(path=FEATURE_LIST):
    # Read the 16 feature names, one per line (never guess them from the table columns)
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_table(path):
    # Read a table that has 'subject', 'label' and the feature columns
    return pd.read_csv(path)


def wilson_ci(n_correct, n_total, level=0.95):
    # 95% confidence interval for an accuracy (Wilson method; behaves well for small n)
    z = norm.ppf(1 - (1 - level) / 2)
    p = n_correct / n_total
    centre = (p + z**2 / (2 * n_total)) / (1 + z**2 / n_total)
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / (1 + z**2 / n_total)
    return centre - half, centre + half


def evaluate(model_factory, df, features, scale=True):
    """
    model_factory: a function with no arguments that returns a NEW untrained model
                   (anything with fit and predict/predict_proba, e.g. lambda: LogisticRegression()).
    df:            table with 'subject', 'label' and the feature columns.
    features:      list of column names to use as inputs.
    scale:         put a StandardScaler in front of the model (fitted on training rows only).
    Returns a dict with per-subject predictions and the summary numbers.
    """
    subjects = df["subject"].unique()
    rows = []

    for test_subject in subjects:
        # Step 1: split by subject - all rows of the test subject go to the test side
        is_test = df["subject"] == test_subject
        train, test = df[~is_test], df[is_test]

        # Step 2: build a fresh model (with scaling inside it, so scaling only sees training rows)
        model = model_factory()
        if scale:
            model = make_pipeline(StandardScaler(), model)

        # Step 3: train on the other subjects
        model.fit(train[features].values, train["label"].values)

        # Step 4: predict the held-out subject; average over its rows if it has many (epochs).
        # Models without probabilities (e.g. Perceptron) give 0/1 votes, averaged the same way.
        if hasattr(model, "predict_proba"):
            prob_patient = model.predict_proba(test[features].values)[:, 1].mean()
        else:
            prob_patient = model.predict(test[features].values).mean()
        rows.append({
            "subject": test_subject,
            "true_label": int(test["label"].iloc[0]),
            "prob_patient": prob_patient,
            "predicted": int(prob_patient >= 0.5),
        })

    # Step 5: one row per subject -> accuracy, balanced accuracy, confidence interval
    per_subject = pd.DataFrame(rows)
    y_true, y_pred = per_subject["true_label"], per_subject["predicted"]
    n_correct = int((y_true == y_pred).sum())
    ci_low, ci_high = wilson_ci(n_correct, len(per_subject))
    return {
        "per_subject": per_subject,
        "n_subjects": len(per_subject),
        "n_correct": n_correct,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


if __name__ == "__main__":
    # First baseline: logistic regression on the 16 features, with and without s14
    features = load_feature_names()
    print(f"Using {len(features)} features from {FEATURE_LIST.name}")

    def logreg():
        return LogisticRegression(max_iter=5000, random_state=SEED)

    summary = []
    for name, path in [("with s14", TABLE_WITH_S14), ("without s14", TABLE_WITHOUT_S14)]:
        df = load_table(path)
        result = evaluate(logreg, df, features)
        print(f"\n{name}: {result['n_subjects']} subjects "
              f"({(df['label'] == 0).sum()} healthy, {(df['label'] == 1).sum()} patients)")
        print(f"  correct           : {result['n_correct']} / {result['n_subjects']}")
        print(f"  accuracy          : {result['accuracy']:.1%}  "
              f"(95% CI {result['ci_low']:.1%} to {result['ci_high']:.1%})")
        print(f"  balanced accuracy : {result['balanced_accuracy']:.1%}")
        wrong = result["per_subject"].query("true_label != predicted")["subject"].tolist()
        print(f"  misclassified     : {', '.join(wrong) if wrong else 'none'}")
        summary.append({"table": name, "model": "logistic regression",
                        **{k: result[k] for k in ["n_subjects", "n_correct", "accuracy",
                                                  "balanced_accuracy", "ci_low", "ci_high"]}})

    pd.DataFrame(summary).to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved: {RESULTS_FILE}")
    print("Chance is 50%. Exploratory result: 28 subjects, one subject = about 3.6 points.")
