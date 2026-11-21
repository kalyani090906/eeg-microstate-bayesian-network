"""
permutation_test_nested.py - are the results with the maps REFITTED IN EACH FOLD better than chance?

Same test as permutation_test.py, on the features from nested_loso_refit_maps.py:
  1. Score each candidate with the real labels (leave-one-subject-out; in the fold where subject t
     is tested, every subject is described with the maps fitted without t).
  2. Shuffle the labels across SUBJECTS, run the whole leave-one-subject-out again, record the
     balanced accuracy. 1,000 shuffles. Shuffle i uses seed 42 + i, exactly as permutation_test.py.
     The fold maps were fitted WITHOUT labels, so a shuffle does not change them: only the
     classifier is retrained. That is why no map has to be refitted here.
  3. p-value = (1 + shuffles scoring at least as well as the real labels) / (1 + 1,000).
  4. p_family: how often the BEST of the 3 candidates on shuffled labels reaches each real score
     (allows for reporting the best of 3). Report this one as the main p-value.

Candidates (fixed before running, the same as permutation_test.py minus the LaBraM one):
  logistic regression, L1 selection + logistic regression, AdaBoost.

Needs: outputs/nested_loso/folds/*.npz  (nested_loso_refit_maps.py)
Outputs (outputs/nested_loso/):
  fold_features_16.csv                the 16 features of every subject in every fold (one table)
  permutation_nested_summary.csv      real score, chance scores, p-values
  permutation_nested_<run>.png        the chance scores (histogram) and the real score (line)
Run:  eeg_env\\Scripts\\python.exe scripts\\permutation_test_nested.py            (about 10-20 minutes)
      eeg_env\\Scripts\\python.exe scripts\\permutation_test_nested.py --n 20     (quick check)
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUT = OUTPUT_DIR / "nested_loso"
FOLD_DIR = OUT / "folds"
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
RUNS = ["with s14", "without s14"]
N_PERMUTATIONS = 1000
N_JOBS = -2                    # all CPU cores but one
SEED = 42
# --------------------------------------------------------------

import argparse                                   # noqa: E402
import sys                                        # noqa: E402
import warnings                                   # noqa: E402

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402
from joblib import Parallel, delayed              # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline        # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from permutation_test import adaboost, l1_logreg, logreg  # noqa: E402  (same model settings)

CANDIDATES = {"Logistic regression": (logreg, True),
              "L1 selection + logistic regression": (l1_logreg, False),     # scales inside itself
              "AdaBoost": (adaboost, True)}


def fold_table():
    # One long table: run, test_subject (the fold), subject, label, 16 features
    path = OUT / "fold_features_16.csv"
    if path.exists():
        return pd.read_csv(path)
    names16 = [ln.strip() for ln in FEATURE_LIST.read_text().splitlines() if ln.strip()]
    parts = []
    for f in sorted(FOLD_DIR.glob("*.npz")):
        run, t = f.stem.split("__")
        d = np.load(f, allow_pickle=True)
        tab = pd.DataFrame(d["table"], columns=d["columns"])[["subject", "label"] + names16]
        tab.insert(0, "test_subject", t)
        tab.insert(0, "run", run.replace("_", " "))
        parts.append(tab)
    table = pd.concat(parts, ignore_index=True)
    table["label"] = table["label"].astype(int)
    table[names16] = table[names16].astype(float)
    table.to_csv(path, index=False)
    return table


def loso(folds, factory, scale, labels, features):
    # Leave-one-subject-out over the fold tables; labels = subject -> label used (real or shuffled)
    y_true, y_pred = [], []
    for t, tab in folds.items():
        train, test = tab[tab.subject != t], tab[tab.subject == t]
        model = make_pipeline(StandardScaler(), factory()) if scale else factory()
        model.fit(train[features].values.astype(float), train.subject.map(labels).values)
        p = model.predict_proba(test[features].values.astype(float))[0, 1]
        y_true.append(labels[t])
        y_pred.append(int(p >= 0.5))
    return balanced_accuracy_score(y_true, y_pred)


def one_shuffle(perm_id, folds, real_labels, features):
    # Shuffle i (seed 42 + i) of the subject labels, every candidate scored on the same shuffle
    subjects = list(real_labels.index)
    labels = real_labels.copy()
    if perm_id > 0:
        labels = pd.Series(np.random.default_rng(SEED + perm_id).permutation(real_labels.values), index=subjects)
    return [{"perm_id": perm_id, "candidate": n, "balanced_accuracy": loso(folds, f, s, labels, features)}
            for n, (f, s) in CANDIDATES.items()]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PERMUTATIONS, help="number of shuffles")
    n_perm = parser.parse_args().n
    table = fold_table()
    FEATURES = [c for c in table.columns if c not in ("run", "test_subject", "subject", "label")]

    summary = []
    for run in RUNS:
        sub = table[table.run == run]
        folds = {t: g for t, g in sub.groupby("test_subject")}
        real_labels = sub.drop_duplicates("subject").set_index("subject")["label"].sort_index()
        print(f"\n=== {run}: {len(folds)} folds, {n_perm} shuffles ===", flush=True)
        rows = Parallel(n_jobs=N_JOBS)(delayed(one_shuffle)(i, folds, real_labels, FEATURES) for i in range(n_perm + 1))
        res = pd.DataFrame([r for block in rows for r in block])
        real = res[res.perm_id == 0].set_index("candidate")["balanced_accuracy"]
        null = res[res.perm_id > 0]
        best_null = null.groupby("perm_id")["balanced_accuracy"].max()
        for n in CANDIDATES:
            nul = null[null.candidate == n]["balanced_accuracy"]
            summary.append({"table": run, "candidate": n, "n_permutations": n_perm,
                            "real_balanced_accuracy": real[n], "null_mean": nul.mean(),
                            "null_95th_percentile": nul.quantile(0.95),
                            "p_value": (1 + (nul >= real[n]).sum()) / (1 + len(nul)),
                            "p_family": (1 + (best_null >= real[n]).sum()) / (1 + len(best_null))})

        # Picture: chance scores of each candidate and its real score
        fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(13, 3.6), sharey=True)
        for ax, n in zip(axes, CANDIDATES):
            ax.hist(null[null.candidate == n]["balanced_accuracy"], bins=np.linspace(0, 1, 29),
                    color="#9db9e0", edgecolor="white")
            ax.axvline(real[n], color="#b00020", lw=2, label=f"real labels: {real[n]:.1%}")
            p = summary[-len(CANDIDATES) + list(CANDIDATES).index(n)]["p_value"]
            ax.set(title=f"{n}\np = {p:.3f}", xlabel="balanced accuracy")
            ax.legend(fontsize=8, loc="upper left")
        axes[0].set_ylabel("number of shuffles")
        fig.suptitle(f"Permutation test, maps refitted in each fold ({run}): {n_perm} label shuffles",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"permutation_nested_{run.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    s = pd.DataFrame(summary)
    s.round(4).to_csv(OUT / "permutation_nested_summary.csv", index=False)
    print("\n" + s[["table", "candidate", "real_balanced_accuracy", "null_mean", "null_95th_percentile",
                    "p_value", "p_family"]].round(3).to_string(index=False))
    print(f"\nSaved in {OUT}")
    print("p_family is the main p-value: it allows for reporting the best of 3 candidates.")
