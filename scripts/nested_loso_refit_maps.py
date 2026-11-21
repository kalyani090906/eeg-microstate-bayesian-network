"""
nested_loso_refit_maps.py - leave-one-subject-out with the microstate MAPS refitted inside every fold.

Why (asked by the faculty: "all the pre-processing inside the cross-validation"):
  In the main analysis the 4 maps were fitted ONCE on the GFP peaks of all 28 subjects. No labels
  were used, but the test subject's own EEG helped to make the maps that later describe it. Here
  that step is moved inside the cross-validation, so the test subject is never seen before testing.

What one fold does (test subject = t):
  1. GFP peaks: the same 1,500 peaks per subject as the main model (same generator, seed 42, file
     order), so the only change from the main analysis is that t's peaks are left out.
  2. Fit 4 maps by modified k-means (k = 4, 50 starts, seed 42) on the peaks of the OTHER subjects
     only (27 subjects; 26 in the "without s14" run, where s14 is never used at all).
  3. Put the fold's maps in the order of the main maps (one-to-one, |correlation|). This only
     fixes the column order, so "state0" means the A-like map in every fold. It cannot carry
     the diagnosis: no label is used and nothing is fitted.
  4. Label every sample of every kept epoch of ALL subjects with the fold's maps (the training
     subjects and t), then the 24 features per epoch and one row per subject, exactly as in
     extract_epoch_features.py; then the same 16 features (make_final_features.py rule).
  5. Train each classifier on the training subjects (scaling and L1 selection learned on them
     only), predict t.
Every subject is t once. Accuracy = correct predictions / number of subjects.

Classifiers: the 11 of item 1 (labs/lab09_a3_classifiers.py) and L1 selection + logistic
regression (item 4), same settings. They are also run on the MAIN feature table (maps from all
28) in the same way, so the two columns of the result table can be compared directly.

Inputs (local, not in git):
  dataset/*.edf                 (for the GFP peaks; read once, then cached)
  outputs/epochs_15min.npz      (the kept 2-20 Hz epochs; made by make_epoch_array.py)
  outputs/shared_microstate_model_v3.pkl   (only to fix the column order, step 3)
Outputs (outputs/nested_loso/):
  gfp_peaks_per_subject.npz     cache of the peaks (not for git, 6 MB)
  folds/<run>__<subject>.npz    one file per fold: maps, GEV, 24 features of all subjects, and the
                                labels of every sample (for the LSTM script) (not for git)
  nested_loso_results.csv       accuracy etc. per classifier: main maps vs maps refitted per fold
  nested_loso_per_subject.csv   the prediction for every test subject
  fold_map_similarity.csv       how close each fold's maps are to the main maps
If stopped, run again: finished folds are skipped.

Run:  eeg_env\\Scripts\\python.exe scripts\\nested_loso_refit_maps.py      (about 30-45 minutes)
"""

# ---------------- settings (change only here) ----------------
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUT = OUTPUT_DIR / "nested_loso"
FOLD_DIR = OUT / "folds"
EPOCH_FILE = OUTPUT_DIR / "epochs_15min.npz"
PEAK_CACHE = OUT / "gfp_peaks_per_subject.npz"
FEATURE_LIST = OUTPUT_DIR / "features_final_16_list.txt"
MAIN_TABLES = {"with s14": OUTPUT_DIR / "features_final_16.csv",
               "without s14": OUTPUT_DIR / "features_final_16_no_s14.csv"}
EXCLUDED = {"with s14": [], "without s14": ["s14"]}
SEED = 42
N_WORKERS = 8                  # map fits run in parallel (one fit = about 2 minutes on one core)
THREADS_PER_WORKER = "2"
# --------------------------------------------------------------

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS_PER_WORKER)

import sys                                         # noqa: E402
import time                                        # noqa: E402
import warnings                                    # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import mne                                         # noqa: E402
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from scipy.optimize import linear_sum_assignment   # noqa: E402

import eeg_preprocessing as pre                    # noqa: E402
import microstate_features as mf                   # noqa: E402
import microstate_model as mm                      # noqa: E402

K = mm.N_STATES


def fit_fold(run, test_subject, train_subjects, peaks_file, info):
    # Worker: modified k-means on the pooled peaks of the training subjects only
    peaks = np.load(peaks_file)
    model = mm.fit_shared_model([peaks[s] for s in train_subjects], info, seed=SEED)
    return run, test_subject, model


def abs_corr(a, b):
    # |spatial correlation| between every row of a and every row of b
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return np.abs(a @ b.T) / a.shape[1]


def make_classifiers():
    # The 11 classifiers of item 1 and L1 selection + logistic regression of item 4 (same settings)
    sys.path.insert(0, str(PROJECT_DIR / "labs"))
    from lab09_a3_classifiers import MODELS
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    models = dict(MODELS)
    models["L1 selection + logistic regression"] = lambda: Pipeline([
        ("select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                                      random_state=SEED))),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED))])
    return models


def one_fold(make_model, train, test, features):
    # Scale -> model, learned on the training subjects only; returns P(patient) for test, train acc
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), make_model())
    model.fit(train[features].values, train["label"].values)
    proba = (lambda X: model.predict_proba(X)[:, 1]) if hasattr(model, "predict_proba") else model.predict
    train_acc = float(((proba(train[features].values) >= 0.5).astype(int) == train["label"].values).mean())
    return float(proba(test[features].values)[0]), train_acc


def summarise(rows, train_accs):
    # Same numbers as lab_common.all_metrics, from the per-subject predictions
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from evaluate import wilson_ci
    ps = pd.DataFrame(rows)
    y, pred, prob = ps["true_label"], ps["predicted"], ps["prob_patient"]
    n_correct = int((y == pred).sum())
    lo, hi = wilson_ci(n_correct, len(ps))
    acc, train = n_correct / len(ps), float(np.mean(train_accs))
    return {"n_subjects": len(ps), "n_correct": n_correct, "accuracy": acc, "ci_low": lo, "ci_high": hi,
            "train_accuracy": train, "overfit_gap": train - acc,
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "auc": roc_auc_score(y, prob)}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    shared = mm.load_model()
    shared_maps = np.asarray(shared.cluster_centers_)
    names24 = mf.feature_names(K)
    names16 = [ln.strip() for ln in FEATURE_LIST.read_text().splitlines() if ln.strip()]
    files = pre.recording_files()
    subjects = [pre.subject_and_label(f)[0] for f in files]
    label_of = {s: pre.subject_and_label(f)[1] for s, f in zip(subjects, files)}

    # Step 1: GFP peaks of every subject, sampled exactly as for the main model (cached)
    if not PEAK_CACHE.exists():
        rng = np.random.default_rng(SEED)
        peaks = {}
        for f, s in zip(files, subjects):
            peaks[s], _ = mm.sample_gfp_peaks(pre.load_clean(f), rng)
            print(f"{s}: {peaks[s].shape[1]} peaks", flush=True)
        np.savez(PEAK_CACHE, **peaks)
    info = shared.info

    # The kept epochs (2-20 Hz, average reference) of every subject, for labelling
    z = np.load(EPOCH_FILE)
    X_all, epoch_subject, epoch_number = z["X"], z["subject"], z["epoch"]
    epoch_info = mne.create_info(list(z["ch_names"]), float(z["sfreq"]), "eeg")
    sfreq = float(z["sfreq"])

    def label_subject(model, s):
        # labels of every sample of every kept epoch of subject s (-1 = unlabelled epoch edge)
        ep = mne.EpochsArray(X_all[epoch_subject == s].astype(np.float64), epoch_info, verbose="ERROR")
        ep.set_montage("standard_1020")
        seg = model.predict(ep, factor=mm.SMOOTH_FACTOR, half_window_size=mm.SMOOTH_HALF_WINDOW,
                            reject_edges=True, verbose="ERROR")
        return np.asarray(seg.labels)

    def finish_fold(run, t, model, path):
        # Steps 3-4: column order of the main maps, label everybody, 24 features per subject
        sim = abs_corr(np.asarray(model.cluster_centers_), shared_maps)
        r, c = linear_sum_assignment(-sim)
        relabel = np.full(K, -1)
        relabel[r] = c
        rows, label_arrays = [], {}
        for s in subjects:
            if s in EXCLUDED[run]:
                continue
            labels = label_subject(model, s)
            labels = np.where(labels >= 0, relabel[np.clip(labels, 0, None)], -1).astype(np.int8)
            label_arrays[s] = labels
            ep_rows, pooled = [], np.zeros((K, K))
            for lab in labels:
                row, counts = mf.epoch_features(lab, sfreq, K)
                pooled += counts
                ep_rows.append(row)
            rows.append({"subject": s, "label": label_of[s], **mf.subject_features(ep_rows, pooled, K, "mean")})
        table = pd.DataFrame(rows)[["subject", "label"] + names24]
        np.savez_compressed(path, centers=np.asarray(model.cluster_centers_)[np.argsort(c)],
                            gev=model.GEV_, match_corr=sim[r, c][np.argsort(c)],
                            table=table.to_numpy(dtype=object), columns=np.array(table.columns),
                            **{f"labels_{s}": a for s, a in label_arrays.items()},
                            **{f"epochs_{s}": epoch_number[epoch_subject == s] for s in label_arrays})

    # Step 2: one map fit per fold, in parallel; each finished fit is labelled and saved at once
    jobs = []
    for run, excluded in EXCLUDED.items():
        pool = [s for s in subjects if s not in excluded]
        for t in pool:
            path = FOLD_DIR / f"{run.replace(' ', '_')}__{t}.npz"
            if not path.exists():
                jobs.append((run, t, [s for s in pool if s != t], path))
    print(f"{len(jobs)} folds to fit (folds already saved are skipped)", flush=True)
    start = time.perf_counter()
    if jobs:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(fit_fold, run, t, train, str(PEAK_CACHE), info): path
                       for run, t, train, path in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                run, t, model = fut.result()
                finish_fold(run, t, model, futures[fut])
                print(f"[{i}/{len(jobs)}] {run}, test {t}: maps fitted on the other subjects, GEV "
                      f"{model.GEV_:.3f} ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

    # Step 5: classifiers, main maps vs maps refitted per fold
    models = make_classifiers()
    results, per_subject, sim_rows = [], [], []
    for run in EXCLUDED:
        main = pd.read_csv(MAIN_TABLES[run])
        pool = list(main["subject"])
        fold_tables = {}
        for t in pool:
            d = np.load(FOLD_DIR / f"{run.replace(' ', '_')}__{t}.npz", allow_pickle=True)
            ft = pd.DataFrame(d["table"], columns=d["columns"])
            ft[names24] = ft[names24].astype(float)
            ft["label"] = ft["label"].astype(int)
            fold_tables[t] = ft
            sim_rows.append({"run": run, "test_subject": t, "GEV": round(float(d["gev"]), 4),
                             **{f"corr_{mm.STATE_NAMES[j]}": round(float(d["match_corr"][j]), 4) for j in range(K)}})
        for name, make_model in models.items():
            for maps in ["main maps (all subjects)", "maps refitted in each fold"]:
                rows, train_accs = [], []
                for t in pool:
                    table = main if maps.startswith("main") else fold_tables[t]
                    test, train = table[table.subject == t], table[table.subject != t]
                    p, tr = one_fold(make_model, train, test, names16)
                    train_accs.append(tr)
                    rows.append({"subject": t, "true_label": int(test["label"].iloc[0]),
                                 "prob_patient": p, "predicted": int(p >= 0.5)})
                results.append({"table": run, "model": name, "maps": maps, **summarise(rows, train_accs)})
                per_subject += [{"table": run, "model": name, "maps": maps, **r} for r in rows]
            a, b = results[-2]["accuracy"], results[-1]["accuracy"]
            print(f"{run:12s} {name:36s} main maps {a:.1%}   refitted per fold {b:.1%}", flush=True)

    res = pd.DataFrame(results)
    res.round(4).to_csv(OUT / "nested_loso_results.csv", index=False)
    pd.DataFrame(per_subject).round(4).to_csv(OUT / "nested_loso_per_subject.csv", index=False)
    sims = pd.DataFrame(sim_rows)
    sims.to_csv(OUT / "fold_map_similarity.csv", index=False)
    corr_cols = [c for c in sims.columns if c.startswith("corr_")]
    print(f"\nFold maps vs main maps: lowest |correlation| over all folds = {sims[corr_cols].min().min():.3f}")
    print(f"Saved in {OUT}")
    print("Exploratory: 28 subjects, one subject = about 3.6 points; chance = 50%.")
