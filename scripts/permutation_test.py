"""
Permutation test: are our best leave-one-subject-out results better than chance?

What it does, in plain words:
  1. Score each candidate model with the real labels (leave-one-subject-out, as everywhere else).
  2. Shuffle the labels across SUBJECTS, re-run the WHOLE leave-one-subject-out evaluation, and
     record the score. Repeat N_PERMUTATIONS times. This shows what scores pure chance produces
     with this exact data, model and evaluation (with 28 subjects that is NOT simply 50%).
  3. p-value = (1 + number of shuffles scoring at least as well as the real labels) / (1 + N).
  4. Family-wise p-value: the candidates were picked after seeing many results, so we also ask
     how often the BEST of all candidates on shuffled labels beats each real score
     (max-statistic correction). Report this one as the main p-value.

The statistic is balanced accuracy (the no-s14 table has 14 vs 13 subjects).
Shuffle number i uses seed SEED + i and shuffles the SUBJECT labels; every candidate gets the
same shuffle (for the epoch-level candidate each subject's epochs all get that subject's
shuffled label). So any single shuffle can be reproduced, and the family-wise max is valid.

Results are appended to outputs/permutation/perm_<table>.csv as they finish, one row per
(shuffle, candidate). Each row stores a fingerprint of the data that candidate used: if the
data change, old rows are ignored and recomputed. If the run stops, start it again and it
carries on; adding a candidate only computes that candidate.

Run:  eeg_env\\Scripts\\python.exe scripts\\permutation_test.py
      eeg_env\\Scripts\\python.exe scripts\\permutation_test.py --n 20      (quick check)
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
EMB_FILE = OUTPUT_DIR / "embeddings_labram.npz"     # from item2_embeddings.py
N_PERMUTATIONS = 1000
CHUNK = 50          # shuffles computed (in parallel) between saves
N_JOBS = -2         # all CPU cores but one
SEED = 42
# --------------------------------------------------------------

import argparse
import hashlib
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


# Step 0: the candidates, fixed BEFORE running
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


def emb_logreg():
    # same as item 2: epoch-level logistic regression on frozen LaBraM embeddings
    return LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)


# name -> (model factory, add a scaler?, data: "features" = 16-feature table, "embeddings" = epochs)
CANDIDATES = {
    "Logistic regression": (logreg, True, "features"),
    "L1 selection + logistic regression": (l1_logreg, False, "features"),
    "AdaBoost": (adaboost, True, "features"),
    "LaBraM embeddings + logistic regression": (emb_logreg, True, "embeddings"),
}


def file_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h


def data_hash(table_path, data_kind):
    # fingerprint of everything a candidate's result depends on (data + seed + feature list)
    h = file_hash(table_path, *([EMB_FILE] if data_kind == "embeddings" else []))
    h.update(f"{SEED}|{load_feature_names(FEATURE_LIST)}".encode())
    return h.hexdigest()[:10]


_EMB_CACHE = {}


def load_embeddings():
    # read once per worker process: one row per epoch with subject, label and e0..e199
    if "emb" not in _EMB_CACHE:
        d = np.load(EMB_FILE, allow_pickle=True)
        cols = [f"e{i}" for i in range(d["emb"].shape[1])]
        df = pd.DataFrame(d["emb"], columns=cols)
        df.insert(0, "label", d["label"].astype(int))
        df.insert(0, "subject", d["subject"].astype(str))
        _EMB_CACHE["emb"] = (df, cols)
    return _EMB_CACHE["emb"]


def score(name, table, perm_id):
    # One full leave-one-subject-out run with (possibly shuffled) subject labels
    factory, scale, kind = CANDIDATES[name]
    subj_labels = table.set_index("subject")["label"]
    if perm_id > 0:
        subj_labels = pd.Series(np.random.default_rng(SEED + perm_id).permutation(subj_labels.values),
                                index=subj_labels.index)
    if kind == "features":
        df = table.assign(label=subj_labels.values)
        cols = load_feature_names(FEATURE_LIST)
    else:
        emb, cols = load_embeddings()
        df = emb[emb["subject"].isin(subj_labels.index)].copy()
        df["label"] = df["subject"].map(subj_labels).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = evaluate(factory, df, cols, scale=scale)
    ps = r["per_subject"]
    return {"accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
            "auc": roc_auc_score(ps["true_label"], ps["prob_patient"])}


def one_job(perm_id, name, table, dhash):
    return {"perm_id": perm_id, "seed": SEED + perm_id if perm_id > 0 else None,
            "candidate": name, "data_hash": dhash, **score(name, table, perm_id)}


def run_table(table_name, path, names, n_perm):
    table = pd.read_csv(path)
    out = RESULTS_DIR / f"perm_{table_name.replace(' ', '_')}.csv"
    hashes = {n: data_hash(path, CANDIDATES[n][2]) for n in names}

    # Step 1: resume - which (shuffle, candidate) pairs are saved with today's data?
    saved = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["perm_id", "candidate", "data_hash"])
    valid = saved[saved.apply(lambda r: hashes.get(r["candidate"]) == r["data_hash"], axis=1)] \
        if len(saved) else saved
    have = set(zip(valid["perm_id"], valid["candidate"]))
    todo = [(i, n) for n in names for i in range(n_perm + 1) if (i, n) not in have]
    print(f"\n=== {table_name}: {len(table)} subjects -> {out.name}")
    print(f"  {len(have)} runs already saved, {len(todo)} to go")

    # Step 2: compute the missing ones in parallel, saving after every chunk
    start = time.perf_counter()
    for c in range(0, len(todo), CHUNK):
        jobs = todo[c:c + CHUNK]
        rows = Parallel(n_jobs=N_JOBS)(delayed(one_job)(i, n, table, hashes[n]) for i, n in jobs)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        finished = c + len(jobs)
        rate = (time.perf_counter() - start) / finished
        print(f"  {finished}/{len(todo)} done, about {rate * (len(todo) - finished) / 60:.1f} min left",
              flush=True)

    # Step 3: p-values from the valid rows (first n_perm shuffles only)
    res = pd.read_csv(out)
    res = res[res.apply(lambda r: hashes.get(r["candidate"]) == r["data_hash"], axis=1)]
    res = res[(res["perm_id"] <= n_perm) & res["candidate"].isin(names)]
    real = res[res["perm_id"] == 0].drop_duplicates("candidate").set_index("candidate")
    null = res[res["perm_id"] > 0].drop_duplicates(["perm_id", "candidate"])
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
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        ax.hist(null[null["candidate"] == n]["balanced_accuracy"], bins=np.linspace(0, 1, 29),
                color="#95a5a6", edgecolor="white")
        ax.axvline(real.loc[n, "balanced_accuracy"], color="#c0392b", linewidth=2, label="real labels")
        ax.axvline(0.5, color="black", linestyle=":", label="50%")
        ax.set_title(n, fontsize=9), ax.set_xlabel("balanced accuracy")
    axes[0].set_ylabel("number of shuffles"), axes[0].legend(fontsize=8)
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
    print(f"p_family is the main p-value: it allows for picking the best of {len(names)} candidates.")
    print(f"Smallest possible p with {args.n} shuffles: {1 / (1 + args.n):.4f}.")
