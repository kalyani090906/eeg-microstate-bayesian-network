"""
Project item 6: comparison with published results on the SAME dataset
(RepOD, Olejarczyk & Jernajczyk 2017: 14 schizophrenia + 14 controls, 19 channels, 250 Hz).

The published numbers below were checked against the papers on 26 October 2026; the column
'checked' says how. Our own numbers are read from the output files, so re-running an item
updates this table. Writes:
  outputs/items/item6_comparison.csv   the table
  outputs/items/item6_comparison.md    the table + five points (what to copy, what to do differently)

Run:  eeg_env\\Scripts\\python.exe scripts\\item6_comparison.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
ITEMS = OUTPUT_DIR / "items"
# --------------------------------------------------------------

import pandas as pd
from scipy.stats import mannwhitneyu

# Step 1: published work on this dataset (verbatim numbers from the papers)
PUBLISHED = [
    {"study": "Kim et al. 2021, PLOS ONE [9]",
     "preprocessing": "2-20 Hz, average reference, 250 Hz",
     "epochs": "5 s non-overlapping, 15 min",
     "microstates": "4 (modified k-means)",
     "features": "19 microstate (duration, occurrence, coverage, mean GFP ...)",
     "classifier": "quadratic SVM (also LDA, NB, RF, kNN)",
     "validation": "10-fold, subject-wise (all epochs of a subject on one side); permutation test",
     "accuracy": "75.64% (76.62% after feature selection)",
     "C_D_direction": "C up (duration, coverage), D down (duration, occurrence, coverage)",
     "comparable": "yes - same preprocessing, subject-wise",
     "checked": "full text (PMC8121321)"},
    {"study": "Keihani et al. 2022, Brain Sciences [10]",
     "preprocessing": "2-20 Hz, average reference, 250 Hz",
     "epochs": "5 s non-overlapping, 15 min",
     "microstates": "4 (modified k-means)",
     "features": "20 microstate (+ mean GFP, map correlation); chi-square ranking, best 6",
     "classifier": "Bayesian-optimised AutoML (best: Gaussian SVM / ensemble)",
     "validation": "main: random 80/20 split of EPOCHS (same subjects in train and test); "
                   "leave-one-subject-out 'about 90%' (supplement, not checked)",
     "accuracy": "90.93% (19 features), 89.04% (6 features)",
     "C_D_direction": "C down, D up (opposite to Kim et al.)",
     "comparable": "no for 90.93% (subject-dependent split)",
     "checked": "full text (PMC9688063); supplement not seen"},
    {"study": "Raeisi et al. 2025, Cognitive Neurodynamics [5]",
     "preprocessing": "microstates in 5 frequency bands",
     "epochs": "not checked",
     "microstates": "4 per band",
     "features": "band-wise microstate + transition features",
     "classifier": "random forest (best); deep neural network 98.31%",
     "validation": "data AUGMENTED from 14 to 56 subjects per group; split details not checked (paywall)",
     "accuracy": "99.94% +/- 0.12%",
     "C_D_direction": "transitions: D->B more in SZ, D->A more in controls (gamma band)",
     "comparable": "no - augmentation",
     "checked": "abstract and secondary sources only"},
    {"study": "Racz & Csukly 2025, medRxiv preprint (context, not a classifier)",
     "preprocessing": "-", "epochs": "-", "microstates": "-", "features": "-", "classifier": "-",
     "validation": "re-ran published pipelines with and without information leakage",
     "accuracy": "on RepOD: leaky vs leakage-free gap 28.27 points; ~65% of reviewed studies leaky",
     "C_D_direction": "-",
     "comparable": "explains why epoch-split results look high",
     "checked": "abstract and search excerpts (preprint, not peer reviewed)"},
]


def ours():
    # Step 2: our results, read from the output files
    i4 = pd.read_csv(ITEMS / "item4_feature_reduction.csv")
    i2 = pd.read_csv(ITEMS / "item2_embeddings.csv")
    i3 = pd.read_csv(ITEMS / "item3_finetune" / "item3_summary.csv")
    perm = pd.read_csv(OUTPUT_DIR / "permutation" / "permutation_summary.csv")

    def acc(df, table, **match):
        row = df[(df["table"] == table)]
        for k, v in match.items():
            row = row[row[k] == v]
        r = row.iloc[0]
        return f"{r['accuracy']:.1%} [{r['ci_low']:.0%}-{r['ci_high']:.0%}]"

    def p(cand, table):
        return perm[(perm.candidate == cand) & (perm.table == table)]["p_family"].iloc[0]

    common = {"preprocessing": "2-20 Hz, average reference, 250 Hz (features)",
              "epochs": "5 s non-overlapping, first 15 min, outlier epochs rejected",
              "microstates": "4, one shared model for all subjects (GEV 67.4%)",
              "comparable": "-", "checked": "this project"}
    rows = [
        {**common, "study": "OURS: microstate features (items 1 + 4)",
         "features": "16 (24 minus redundant columns, no labels used); L1 selection inside each fold",
         "classifier": "L1 selection + logistic regression",
         "validation": "leave-one-subject-out; 1000-shuffle permutation test, corrected for 4 candidates",
         "accuracy": f"{acc(i4, 'with s14', classifier='Logistic regression', feature_set='L1 selection of 16 (in fold)')}"
                     f" with s14 (p={p('L1 selection + logistic regression', 'with s14'):.3f}); "
                     f"{acc(i4, 'without s14', classifier='Logistic regression', feature_set='L1 selection of 16 (in fold)')}"
                     f" without (p={p('L1 selection + logistic regression', 'without s14'):.3f})"},
        {**common, "study": "OURS: frozen LaBraM embeddings (item 2)",
         "preprocessing": "0.1-75 Hz, 50 Hz notch, 200 Hz (LaBraM's pretraining setting)",
         "microstates": "- (raw EEG)", "features": "200-number embedding per epoch (pretrained, frozen)",
         "classifier": "logistic regression, epoch probabilities averaged per subject",
         "validation": "leave-one-subject-out; permutation test, corrected for 4 candidates",
         "accuracy": f"{acc(i2, 'with s14', level='epoch level', model='Logistic regression')} with s14 "
                     f"(p={p('LaBraM embeddings + logistic regression', 'with s14'):.3f}); "
                     f"{acc(i2, 'without s14', level='epoch level', model='Logistic regression')} without "
                     f"(p={p('LaBraM embeddings + logistic regression', 'without s14'):.3f})"},
        {**common, "study": "OURS: fine-tuned LaBraM (item 3)",
         "preprocessing": "0.1-75 Hz, 50 Hz notch, 200 Hz", "microstates": "- (raw EEG)",
         "features": "last 2 of 12 layers + head trained", "classifier": "LaBraM + linear head",
         "validation": "7-fold subject-grouped (different protocol); no permutation test",
         "accuracy": f"{acc(i3, 'with s14')} with s14; {acc(i3, 'without s14')} without"},
    ]
    return rows


def directions():
    # Step 3: which way C and D differ in OUR data (descriptive, uncorrected Mann-Whitney)
    d = pd.read_csv(OUTPUT_DIR / "eeg_features_epoch_mean_15min.csv")
    out = []
    for name, s in [("C", 3), ("D", 1)]:
        for f in ["duration", "occurrence", "coverage"]:
            c = f"state{s}_{f}"
            hc, sz = d[d.label == 0][c], d[d.label == 1][c]
            out.append(f"{name} {f} {'up' if sz.mean() > hc.mean() else 'down'} "
                       f"(p={mannwhitneyu(sz, hc).pvalue:.3f})")
    return "; ".join(out)


if __name__ == "__main__":
    rows = PUBLISHED + ours()
    for r in rows:
        if r["study"].startswith("OURS: microstate"):
            r["C_D_direction"] = directions() + " - uncorrected, see group statistics"
        r.setdefault("C_D_direction", "-")
    cols = ["study", "preprocessing", "epochs", "microstates", "features", "classifier",
            "validation", "accuracy", "C_D_direction", "comparable", "checked"]
    table = pd.DataFrame(rows)[cols]
    table.to_csv(ITEMS / "item6_comparison.csv", index=False)

    # Step 4: a readable version: a short main table, a details table, and the five points
    def md_table(columns, headers):
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |"
                  for r in table[columns].values]
        return lines

    md = ["# Item 6: comparison with published results (same RepOD dataset, 14 + 14)", "",
          "## Main comparison", ""]
    md += md_table(["study", "validation", "accuracy", "comparable"],
                   ["Study", "Validation", "Accuracy", "Comparable with ours?"])
    md += ["", "## Methods in detail", ""]
    md += md_table(["study", "preprocessing", "epochs", "microstates", "features", "classifier"],
                   ["Study", "Preprocessing", "Epochs", "Microstates", "Features", "Classifier"])
    md += ["", "## Group differences and how each row was checked", ""]
    md += md_table(["study", "C_D_direction", "checked"],
                   ["Study", "States C and D in patients vs controls", "How checked"])
    md += ["", "## What to copy, what to do differently", "",
           "1. **Copy Kim et al.'s set-up.** Same 2-20 Hz band, average reference, 5-s epochs and four "
           "microstates, so our microstate results are directly comparable with their 75.64% (subject-wise). "
           "Ours is in the same range.",
           "2. **Copy subject-wise splitting; do not copy epoch splitting.** Keihani et al.'s 90.93% comes from "
           "a random split of epochs, so the same people are in training and test. A re-analysis of this very "
           "dataset (Racz & Csukly, preprint) found such leakage inflates accuracy by about 28 points.",
           "3. **Do not copy augmentation.** Raeisi et al.'s 99.94% uses data augmented from 14 to 56 subjects "
           "per group; augmented copies of one person can make a model recognise people rather than diagnosis. "
           "We use no augmentation and no overlapping windows.",
           "4. **Go further than all three on checking chance.** Keep all learning (scaling, feature "
           "selection, tuning) inside the training fold, report with and without s14, give confidence "
           "intervals, and use a permutation test corrected for the number of candidate models.",
           "5. **Treat the C/D direction with care.** Our trends (C longer, D lower in patients) match Kim "
           "et al. and the C-up / D-down pattern reported in reviews, but are opposite to Keihani et al. on "
           "the same data. Because which template is called C or D is a matching choice (ours are the least "
           "certain), state the direction as tentative until the group statistics are done.",
           "", "Not compared: results on other datasets (different recordings and patients). "
           "Oh et al. 2019 (CNN on this dataset) was not included because its numbers could not be checked."]
    (ITEMS / "item6_comparison.md").write_text("\n".join(md), encoding="utf-8")
    # Step 5: terminal view, one block per study (a wide table wraps in the terminal)
    for r in table.itertuples():
        print(f"\n{r.study}\n  validation : {r.validation}\n  accuracy   : {r.accuracy}"
              f"\n  comparable : {r.comparable}")
    print(f"\nSaved: {ITEMS / 'item6_comparison.csv'} and item6_comparison.md")
    print("To see it as a table: open outputs/items/item6_comparison.md in VS Code and press Ctrl+Shift+V.")
