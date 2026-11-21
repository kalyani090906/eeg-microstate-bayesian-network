"""
item8_lstm_sequences.py - a sequence model (LSTM) on the microstate SEQUENCE itself, asked for by
the faculty ("the sequential thing").

Why: the 16 features are summaries (how long, how often, which state follows which). They throw
away the ORDER of the states beyond one step. An LSTM reads the whole sequence of one epoch,
step by step, and can use longer patterns (for example A -> D -> A -> D runs).

Input of one epoch (5 s): its sequence of microstate segments, one step per segment:
    A (64 ms) -> D (112 ms) -> B (48 ms) -> ...      about 60-80 steps per epoch
    each step = which state (one-hot, 4 numbers) + log of its duration (1 number, scaled on the
    training subjects only). The two edge segments of each epoch are left out (cut by the edge).
Model: LSTM (1 layer, 32 units) -> dropout 0.3 -> one output = P(schizophrenia) for that epoch.
    Adam, learning rate 0.001, weight decay 0.0001, batch 64, 20 passes over the training epochs.
    All settings fixed BEFORE seeing any result (no tuning on the test subjects).
Subject prediction: the mean of its epochs' probabilities (as evaluate.py), >= 0.5 = schizophrenia.

Validation: leave-one-subject-out with NO leakage at all:
    the sequences come from nested_loso_refit_maps.py, where the 4 maps of each fold were fitted
    without the test subject; the LSTM is trained on the training subjects' epochs only, and the
    duration scaling is learned on them only.

Control (to read the result honestly): the same LSTM on the same segments with their ORDER
shuffled inside each epoch. The shuffled input keeps every duration and every state count but
destroys the order. If the real order beats the shuffled order, the order carries information;
if they are about equal, the LSTM is only using counts and durations (what the 16 features
already have).

Needs: outputs/nested_loso/folds/*.npz  (run nested_loso_refit_maps.py first)
Outputs (outputs/lstm/):
    lstm_results.csv       accuracy etc. per run (with / without s14) and input (real / shuffled order)
    lstm_per_subject.csv   prediction for every test subject
Run:  eeg_env\\Scripts\\python.exe scripts\\item8_lstm_sequences.py      (about 10-20 minutes)
"""

# ---------------- settings (change only here) ----------------
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FOLD_DIR = OUTPUT_DIR / "nested_loso" / "folds"
OUT = OUTPUT_DIR / "lstm"
RUNS = {"with s14": [], "without s14": ["s14"]}
INPUTS = ["real order", "shuffled order (control)"]
SFREQ = 250.0
SEED = 42
HIDDEN = 32
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 64
PASSES = 20                    # training epochs (passes over all training sequences)
N_WORKERS = 8                  # folds trained in parallel
THREADS_PER_WORKER = 2
# --------------------------------------------------------------

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS_PER_WORKER))

import sys                                         # noqa: E402
import time                                        # noqa: E402
import warnings                                    # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
import torch                                       # noqa: E402
from torch import nn                               # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence  # noqa: E402

K = 4


def segments(labels):
    # One epoch's labels (one per sample) -> list of (state, duration in samples); edges (-1) dropped
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    lengths = np.diff(np.r_[starts, len(labels)])
    states = labels[starts]
    keep = states >= 0
    return states[keep], lengths[keep]


def fold_sequences(path, excluded):
    # All epochs of all subjects of one fold -> (subject, label, states, log-durations) per epoch
    d = np.load(path, allow_pickle=True)
    table = pd.DataFrame(d["table"], columns=d["columns"])
    out = []
    for subject, label in zip(table["subject"], table["label"].astype(int)):
        if subject in excluded:
            continue
        for ep in d[f"labels_{subject}"]:
            states, lengths = segments(ep.astype(int))
            out.append((subject, label, states, np.log(lengths / SFREQ * 1000)))
    return out


class SequenceNet(nn.Module):
    # LSTM over the segments of one epoch -> last hidden state -> P(schizophrenia)
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=K + 1, hidden_size=HIDDEN, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.drop(h[-1])).squeeze(-1)


def to_tensors(items, mu, sd):
    # one-hot state + scaled log-duration, padded to the longest epoch in the list
    xs = [torch.tensor(np.c_[np.eye(K)[s], (dur - mu) / sd], dtype=torch.float32) for _, _, s, dur in items]
    return pad_sequence(xs, batch_first=True), torch.tensor([len(s) for _, _, s, _ in items])


def run_fold(run, input_name, test_subject, path, excluded):
    # Worker: train on the training subjects' epochs of this fold, predict the test subject
    torch.set_num_threads(THREADS_PER_WORKER)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    items = fold_sequences(path, excluded)
    if input_name.startswith("shuffled"):            # control: same segments, order destroyed
        shuffled = []
        for subj, lab, s, dur in items:
            order = rng.permutation(len(s))
            shuffled.append((subj, lab, s[order], dur[order]))
        items = shuffled
    train = [it for it in items if it[0] != test_subject]
    test = [it for it in items if it[0] == test_subject]

    # Step 1: scale the log-durations with the training epochs only
    all_dur = np.concatenate([dur for _, _, _, dur in train])
    mu, sd = all_dur.mean(), all_dur.std()
    Xtr, Ltr = to_tensors(train, mu, sd)
    ytr = torch.tensor([lab for _, lab, _, _ in train], dtype=torch.float32)
    Xte, Lte = to_tensors(test, mu, sd)

    # Step 2: train the LSTM (fixed settings, no early stopping on the test subject)
    net = SequenceNet()
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(PASSES):
        net.train()
        for idx in torch.randperm(len(train), generator=gen).split(BATCH):
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx], Ltr[idx]), ytr[idx])
            loss.backward()
            opt.step()

    # Step 3: epoch probabilities -> one probability per subject (mean over its epochs)
    net.eval()
    with torch.no_grad():
        p_test = torch.sigmoid(net(Xte, Lte)).numpy()
        p_train = torch.sigmoid(net(Xtr, Ltr)).numpy()
    train_subj = pd.DataFrame({"s": [it[0] for it in train], "y": ytr.numpy(), "p": p_train}).groupby("s").mean()
    train_acc = float(((train_subj["p"] >= 0.5).astype(int) == train_subj["y"]).mean())
    return {"table": run, "input": input_name, "subject": test_subject, "true_label": test[0][1],
            "prob_patient": float(p_test.mean()), "predicted": int(p_test.mean() >= 0.5),
            "epoch_accuracy": float(((p_test >= 0.5).astype(int) == test[0][1]).mean()),
            "train_accuracy": train_acc}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    from nested_loso_refit_maps import summarise
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for run, excluded in RUNS.items():
        paths = sorted(FOLD_DIR.glob(f"{run.replace(' ', '_')}__*.npz"))
        expected = 28 - len(excluded)
        if len(paths) != expected:
            sys.exit(f"{run}: {len(paths)} fold files found, {expected} needed. Run nested_loso_refit_maps.py first.")
        for input_name in INPUTS:
            for path in paths:
                jobs.append((run, input_name, path.stem.split("__")[1], str(path), excluded))
    print(f"{len(jobs)} folds to train ({N_WORKERS} at a time)", flush=True)

    start, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(run_fold, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['table']}, {r['input']}, test {r['subject']}: "
                  f"P(SZ) = {r['prob_patient']:.2f} (true {r['true_label']})  "
                  f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

    per_subject = pd.DataFrame(rows).sort_values(["table", "input", "subject"])
    per_subject.round(4).to_csv(OUT / "lstm_per_subject.csv", index=False)
    results = []
    for (run, input_name), g in per_subject.groupby(["table", "input"], sort=False):
        results.append({"table": run, "model": "LSTM on microstate sequences", "input": input_name,
                        **summarise(g.to_dict("records"), g["train_accuracy"].tolist()),
                        "mean_epoch_accuracy": g["epoch_accuracy"].mean()})
    res = pd.DataFrame(results)
    res.round(4).to_csv(OUT / "lstm_results.csv", index=False)
    print("\n" + res[["table", "input", "n_correct", "n_subjects", "accuracy", "ci_low", "ci_high",
                      "train_accuracy", "auc"]].round(3).to_string(index=False))
    print(f"\nSaved in {OUT}")
    print("Exploratory: 28 subjects, one subject = about 3.6 points; chance = 50%.")
