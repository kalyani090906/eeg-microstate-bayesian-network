# Item 6: comparison with published results (same RepOD dataset, 14 + 14)

## Main comparison

| Study | Validation | Accuracy | Comparable with ours? |
|---|---|---|---|
| Kim et al. 2021, PLOS ONE [9] | 10-fold, subject-wise (all epochs of a subject on one side); permutation test | 75.64% (76.62% after feature selection) | yes - same preprocessing, subject-wise |
| Keihani et al. 2022, Brain Sciences [10] | main: random 80/20 split of EPOCHS (same subjects in train and test); leave-one-subject-out 'about 90%' (supplement, not checked) | 90.93% (19 features), 89.04% (6 features) | no for 90.93% (subject-dependent split) |
| Raeisi et al. 2025, Cognitive Neurodynamics [5] | data AUGMENTED from 14 to 56 subjects per group; split details not checked (paywall) | 99.94% +/- 0.12% | no - augmentation |
| Racz & Csukly 2025, medRxiv preprint (context, not a classifier) | re-ran published pipelines with and without information leakage | on RepOD: leaky vs leakage-free gap 28.27 points; ~65% of reviewed studies leaky | explains why epoch-split results look high |
| OURS: microstate features (items 1 + 4) | leave-one-subject-out; 1000-shuffle permutation test, corrected for 4 candidates | 78.6% [60%-90%] with s14 (p=0.033); 81.5% [63%-92%] without (p=0.014) | - |
| OURS: frozen LaBraM embeddings (item 2) | leave-one-subject-out; permutation test, corrected for 4 candidates | 75.0% [57%-87%] with s14 (p=0.059); 74.1% [55%-87%] without (p=0.050) | - |
| OURS: fine-tuned LaBraM (item 3) | 7-fold subject-grouped (different protocol); no permutation test | 71.4% [53%-85%] with s14; 77.8% [59%-89%] without | - |

## Methods in detail

| Study | Preprocessing | Epochs | Microstates | Features | Classifier |
|---|---|---|---|---|---|
| Kim et al. 2021, PLOS ONE [9] | 2-20 Hz, average reference, 250 Hz | 5 s non-overlapping, 15 min | 4 (modified k-means) | 19 microstate (duration, occurrence, coverage, mean GFP ...) | quadratic SVM (also LDA, NB, RF, kNN) |
| Keihani et al. 2022, Brain Sciences [10] | 2-20 Hz, average reference, 250 Hz | 5 s non-overlapping, 15 min | 4 (modified k-means) | 20 microstate (+ mean GFP, map correlation); chi-square ranking, best 6 | Bayesian-optimised AutoML (best: Gaussian SVM / ensemble) |
| Raeisi et al. 2025, Cognitive Neurodynamics [5] | microstates in 5 frequency bands | not checked | 4 per band | band-wise microstate + transition features | random forest (best); deep neural network 98.31% |
| Racz & Csukly 2025, medRxiv preprint (context, not a classifier) | - | - | - | - | - |
| OURS: microstate features (items 1 + 4) | 2-20 Hz, average reference, 250 Hz (features) | 5 s non-overlapping, first 15 min, outlier epochs rejected | 4, one shared model for all subjects (GEV 67.4%) | 16 (24 minus redundant columns, no labels used); L1 selection inside each fold | L1 selection + logistic regression |
| OURS: frozen LaBraM embeddings (item 2) | 0.1-75 Hz, 50 Hz notch, 200 Hz (LaBraM's pretraining setting) | 5 s non-overlapping, first 15 min, outlier epochs rejected | - (raw EEG) | 200-number embedding per epoch (pretrained, frozen) | logistic regression, epoch probabilities averaged per subject |
| OURS: fine-tuned LaBraM (item 3) | 0.1-75 Hz, 50 Hz notch, 200 Hz | 5 s non-overlapping, first 15 min, outlier epochs rejected | - (raw EEG) | last 2 of 12 layers + head trained | LaBraM + linear head |

## Group differences and how each row was checked

| Study | States C and D in patients vs controls | How checked |
|---|---|---|
| Kim et al. 2021, PLOS ONE [9] | C up (duration, coverage), D down (duration, occurrence, coverage) | full text (PMC8121321) |
| Keihani et al. 2022, Brain Sciences [10] | C down, D up (opposite to Kim et al.) | full text (PMC9688063); supplement not seen |
| Raeisi et al. 2025, Cognitive Neurodynamics [5] | transitions: D->B more in SZ, D->A more in controls (gamma band) | abstract and secondary sources only |
| Racz & Csukly 2025, medRxiv preprint (context, not a classifier) | - | abstract and search excerpts (preprint, not peer reviewed) |
| OURS: microstate features (items 1 + 4) | C duration up (p=0.029); C occurrence down (p=0.800); C coverage up (p=0.629); D duration down (p=0.535); D occurrence down (p=0.148); D coverage down (p=0.370) - uncorrected, see group statistics | this project |
| OURS: frozen LaBraM embeddings (item 2) | - | this project |
| OURS: fine-tuned LaBraM (item 3) | - | this project |

## What to copy, what to do differently

1. **Copy Kim et al.'s set-up.** Same 2-20 Hz band, average reference, 5-s epochs and four microstates, so our microstate results are directly comparable with their 75.64% (subject-wise). Ours is in the same range.
2. **Copy subject-wise splitting; do not copy epoch splitting.** Keihani et al.'s 90.93% comes from a random split of epochs, so the same people are in training and test. A re-analysis of this very dataset (Racz & Csukly, preprint) found such leakage inflates accuracy by about 28 points.
3. **Do not copy augmentation.** Raeisi et al.'s 99.94% uses data augmented from 14 to 56 subjects per group; augmented copies of one person can make a model recognise people rather than diagnosis. We use no augmentation and no overlapping windows.
4. **Go further than all three on checking chance.** Keep all learning (scaling, feature selection, tuning) inside the training fold, report with and without s14, give confidence intervals, and use a permutation test corrected for the number of candidate models.
5. **Treat the C/D direction with care.** Our trends (C longer, D lower in patients) match Kim et al. and the C-up / D-down pattern reported in reviews, but are opposite to Keihani et al. on the same data. Because which template is called C or D is a matching choice (ours are the least certain), state the direction as tentative until the group statistics are done.

Not compared: results on other datasets (different recordings and patients). Oh et al. 2019 (CNN on this dataset) was not included because its numbers could not be checked.