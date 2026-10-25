"""
Lab 9 A5 (also project item 5): LIME explanations of the pipeline's predictions.

For every leave-one-subject-out fold: fit the pipeline on the 27 training subjects, build the
LIME explainer from the training subjects only, and explain the held-out subject. LIME is
unstable with so few subjects, so this is repeated with several seeds and averaged.

Output: one bar chart per table (mean |LIME weight| per feature, coloured by direction).

Run:  eeg_env\\Scripts\\python.exe labs\\lab09_a5_lime.py
Needs: pip install lime
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- settings ----------------
SEEDS = [42, 43, 44, 45, 46]       # LIME repeats (it is random)
NUM_SAMPLES = 1000                 # perturbed samples LIME draws per explanation
CLASSIFIERS = {                    # explain the best simple model and the best model from Lab 9 A3
    "Logistic regression": lambda: LogisticRegression(max_iter=5000, random_state=SEED),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED),
}
# ------------------------------------------


def explain_table(df, features, make_clf):
    # Returns a (subjects*seeds) x features table of LIME weights towards "patient"
    rows = []
    for test_subject in df["subject"]:
        # Step 1: split by subject and fit the pipeline on the training subjects
        train, test = df[df["subject"] != test_subject], df[df["subject"] == test_subject]
        pipe = Pipeline([("scale", StandardScaler()), ("clf", make_clf())])
        pipe.fit(train[features].values, train["label"].values)

        for seed in SEEDS:
            # Step 2: LIME learns what "typical" values are from the training subjects only
            explainer = LimeTabularExplainer(
                train[features].values, feature_names=features,
                class_names=["control", "patient"], discretize_continuous=True,
                random_state=seed)
            # Step 3: explain the held-out subject's prediction
            exp = explainer.explain_instance(test[features].values[0], pipe.predict_proba,
                                             num_features=len(features), num_samples=NUM_SAMPLES)
            weights = dict(exp.as_map()[1])                 # feature index -> weight
            rows.append([weights.get(i, 0.0) for i in range(len(features))])
    return pd.DataFrame(rows, columns=features)


def plot(summary, title, path):
    # Bar chart: length = average importance, colour = which class it usually pushes towards
    summary = summary.sort_values("mean_abs_weight")
    colours = ["#c0392b" if w > 0 else "#2471a3" for w in summary["mean_weight"]]
    plt.figure(figsize=(8, 6))
    plt.barh(summary.index, summary["mean_abs_weight"], color=colours)
    plt.xlabel("Mean |LIME weight| (across held-out subjects and seeds)")
    plt.title(title + "\nred = pushes towards patient, blue = towards control")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    features = load_feature_names(FEATURE_LIST)
    for clf_name, make_clf in CLASSIFIERS.items():
        for table_name, path in TABLES.items():
            df = load_table(path)
            print(f"LIME: {clf_name}, {table_name} ({len(df)} subjects x {len(SEEDS)} seeds) ...")
            weights = explain_table(df, features, make_clf)

            # Step 4: average across all held-out subjects and seeds
            summary = pd.DataFrame({
                "mean_abs_weight": weights.abs().mean(),
                "mean_weight": weights.mean(),
                "sd_abs_weight": weights.abs().std(),
            })
            tag = f"{clf_name.split()[0].lower()}_{table_name.replace(' ', '_')}"
            summary.sort_values("mean_abs_weight", ascending=False).round(4).to_csv(
                LAB_OUT / f"lab09_a5_lime_{tag}.csv")
            plot(summary, f"LIME - {clf_name}, {table_name}",
                 LAB_OUT / f"lab09_a5_lime_{tag}.png")
            top = summary["mean_abs_weight"].sort_values(ascending=False).head(3)
            print("   top 3:", ", ".join(f"{f} ({w:.3f})" for f, w in top.items()))
    print(f"\nSaved charts and tables in {LAB_OUT}")
    print("LIME shows what the model uses, not what causes schizophrenia.")
