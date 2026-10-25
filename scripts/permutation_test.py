"""
Permutation test: are our best leave-one-subject-out results better than chance?

What it does, in plain words:
  1. Score each candidate model with the real labels (leave-one-subject-out, as everywhere else).
  2. Shuffle the labels across subjects, re-run the WHOLE leave-one-subject-out evaluation, and
     record the score. Repeat N_PERMUTATIONS times. This shows what scores pure chance produces
     with this exact data, model and evaluation (with 28 subjects that is NOT simply 50%).
  3. p-value = (1 + number of shuffles scoring at least as well as the real labels) / (1 + N).
  4. Family-wise p-value: the candidates were picked after seeing many results, so we also ask
     how often the BEST of all candidates on shuffled labels beats each real score
     (max-statistic correction). Report this one as the main p-value.

The statistic is balanced accuracy (the no-s14 table has 14 vs 13 subjects).
Each shuffle uses seed SEED + permutation number, so any single shuffle can be reproduced.
Results are appended to a CSV as they finish: if the run stops, start it again and it carries
on. The file name contains a fingerprint of the data and settings, so changed data or settings
start a fresh file instead of mixing old and new results.

Run (about 1 hour with AdaBoost on 22 cores; logistic regression alone takes minutes):
    eeg_env\\Scripts\\python.exe scripts\\permutation_test.py
Quick check first (20 shuffles):
    eeg_env\\Scripts\\python.exe scripts\\permutation_test.py --n 20
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "permutation"
TABLES = {
    "with s14": OUTPUT_DIR / "features_final_16.csv",
    "without s14": OUTPUT_DIR / "features_final_16_no_s14.csv",
}
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
N_PERMUTATIONS = 1000
CHUNK = 50          # shuffles computed (in parallel) between saves
N_JOBS = -2         # all CPU cores but one
SEED = 42
# --------------------------------------------------------------

import argparse
import hashlib
import json
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from evaluate import evaluate, load_feature_names  # noqa: E402


# Step 0: the candidates, fixed BEFORE running (name -> (model factory, add a scaler?))
def logreg():
    return LogisticRegression(max_iter=5000, random_state=SEED)


class L1ThenLogReg:
    """
    Same as item 4's pipeline (scale -> L1 selection -> logistic regression), with one addition:
    on shuffled labels the L1 step can keep ZERO features. Then there is nothing to learn from,
    so the model predicts the training share of patients for everyone (a "found nothing" guess)
    instead of crashing. On the real labels it behaves exactly like item 4.
    """

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=SEED)
        self.select = SelectFromModel(l1).fit(Xs, y)
        self.prior = float(np.mean(y))
        self.empty = not self.select.get_support().any()
        if not self.empty:
            self.clf = LogisticRegression(max_iter=5000, random_state=SEED).fit(
                self.select.transform(Xs), y)
        return self

    def predict_proba(self, X):
        if self.empty:
            return np.tile([1 - self.prior, self.prior], (len(X), 1))
        return self.clf.predict_proba(self.select.transform(self.scaler.transform(X)))


def l1_logreg():
    return L1ThenLogReg()


def adaboost():
    return AdaBoostClassifier(n_estimators=100, random_state=SEED)


CANDIDATES = {
    "Logistic regression": (logreg, True),
    "L1 selection + logistic regression": (l1_logreg, False),
    "AdaBoost": (adaboost, True),
}


def score(name, df, features):
    # One full leave-one-subject-out run -> accuracy, balanced accuracy, AUC
    factory, scale = CANDIDATES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = evaluate(factory, df, features, scale=scale)
    ps = r["per_subject"]
    return {"accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
            "auc": roc_auc_score(ps["true_label"], ps["prob_patient"])}


def one_permutation(perm_id, df, features, names):
    # perm_id 0 = real labels; perm_id >= 1 = labels shuffled with seed SEED + perm_id
    data = df.copy()
    seed = SEED + perm_id
    if perm_id > 0:
        data["label"] = np.random.default_rng(seed).permutation(data["label"].values)
    return [{"perm_id": perm_id, "seed": seed if perm_id > 0 else None, "candidate": n,
             **score(n, data, features)} for n in names]


def fingerprint(path, names):
    # Short hash of the table contents + the settings that change the results
    h = hashlib.sha256(Path(path).read_bytes())
    h.update(json.dumps({"seed": SEED, "candidates": names,
                         "features": load_feature_names(FEATURE_LIST)}).encode())
    return h.hexdigest()[:10]


def run_table(table_name, path, names, n_perm):
    df = pd.read_csv(path)
    features = load_feature_names(FEATURE_LIST)
    out = RESULTS_DIR / f"perm_{table_name.replace(' ', '_')}_{fingerprint(path, names)}.csv"

    # Step 1: resume - which shuffles are already in the results file?
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["perm_id"])
    todo = [i for i in range(0, n_perm + 1) if i not in set(done["perm_id"])]
    print(f"\n=== {table_name}: {len(df)} subjects -> {out.name}")
    print(f"  {len(set(done['perm_id']))} runs already saved, {len(todo)} to go")

    # Step 2: compute the remaining shuffles in parallel, saving after every chunk
    start = time.perf_counter()
    for c in range(0, len(todo), CHUNK):
        ids = todo[c:c + CHUNK]
        batches = Parallel(n_jobs=N_JOBS)(delayed(one_permutation)(i, df, features, names) for i in ids)
        rows = pd.DataFrame([row for batch in batches for row in batch])
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        finished = c + len(ids)
        rate = (time.perf_counter() - start) / finished
        print(f"  {finished}/{len(todo)} done, about {rate * (len(todo) - finished) / 60:.1f} min left",
              flush=True)

    # Step 3: p-values from everything saved (only the first n_perm shuffles are used)
    res = pd.read_csv(out)
    res = res[res["perm_id"] <= n_perm]
    real = res[res["perm_id"] == 0].set_index("candidate")
    null = res[res["perm_id"] > 0]
    best_null = null.groupby("perm_id")["balanced_accuracy"].max()   # best candidate per shuffle
    summary = []
    for n in names:
        obs = real.loc[n, "balanced_accuracy"]
        nul = null[null["candidate"] == n]["balanced_accuracy"]
        summary.append({
            "table": table_name, "candidate": n, "n_permutations": len(nul),
            "real_accuracy": real.loc[n, "accuracy"], "real_balanced_accuracy": obs,
            "real_auc": real.loc[n, "auc"],
            "null_mean": nul.mean(), "null_95th_percentile": nul.quantile(0.95),
            "p_value": (1 + (nul >= obs).sum()) / (1 + len(nul)),
            "p_family": (1 + (best_null >= obs).sum()) / (1 + len(best_null)),
        })
    return pd.DataFrame(summary), null, real


def plot(null, real, table_name, names):
    # One histogram per candidate: chance scores, with the real score as a red line
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.8), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        ax.hist(null[null["candidate"] == n]["balanced_accuracy"], bins=np.linspace(0, 1, 29),
                color="#95a5a6", edgecolor="white")
        ax.axvline(real.loc[n, "balanced_accuracy"], color="#c0392b", linewidth=2, label="real labels")
        ax.axvline(0.5, color="black", linestyle=":", label="50%")
        ax.set_title(n, fontsize=10), ax.set_xlabel("balanced accuracy")
    np.atleast_1d(axes)[0].set_ylabel("number of shuffles"), np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Permutation test ({table_name}): chance scores vs the real score")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"permutation_{table_name.replace(' ', '_')}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PERMUTATIONS, help="number of shuffles")
    parser.add_argument("--only", nargs="*", help="run only these candidates (names as in CANDIDATES)")
    args = parser.parse_args()
    names = args.only or list(CANDIDATES)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for table_name, path in TABLES.items():
        summary, null, real = run_table(table_name, path, names, args.n)
        plot(null, real, table_name, names)
        summaries.append(summary)
        print(summary[["candidate", "real_balanced_accuracy", "null_mean", "null_95th_percentile",
                       "p_value", "p_family"]].round(3).to_string(index=False))

    final = pd.concat(summaries)
    final.round(4).to_csv(RESULTS_DIR / "permutation_summary.csv", index=False)
    print(f"\nSaved in {RESULTS_DIR}")
    print("Read p_family as the main p-value (it allows for having picked the best of several models).")
    print(f"Smallest possible p with {args.n} shuffles: {1 / (1 + args.n):.4f}.")
