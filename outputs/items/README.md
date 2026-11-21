# Project items: where each result is

All results use leave-one-subject-out validation (except item 3) and are reported twice:
with s14 (28 subjects, main) and without s14 (27 subjects, sensitivity check).
With 28 subjects every result is exploratory; one subject is about 3.6 percentage points.

`accuracy` is the test accuracy (the result). `train_accuracy` is the same model scored on its own
training subjects, averaged over the folds, and `overfit_gap` = train minus test. A large gap means
the model memorised its training subjects; it is shown for that reason only.

| Item | What it is | Files here | Made by |
|---|---|---|---|
| 1 | 16 microstate features + 11 ML classifiers | `item1_classifiers.csv` | `labs/lab09_a3_classifiers.py` (copied by `scripts/collect_items.py`) |
| 2 | Frozen LaBraM embeddings + classifiers | `item2_embeddings.csv` | `scripts/item2_embeddings.py` |
| 3 | Fine-tuned LaBraM (7-fold subject-grouped, a different protocol) | `item3_finetune/` | `scripts/item3_finetune.py` |
| 4 | Feature reduction: 24 vs 16 vs top-k vs L1 | `item4_feature_reduction.csv`, `item4_top4_selection_counts.csv` | `scripts/item4_feature_reduction.py` |
| 5 | Pipeline (scale, select top-k, classify) + LIME | `item5_pipeline.csv`, `item5_lime_*.png` / `.csv` | `labs/lab09_a4_pipeline.py`, `labs/lab09_a5_lime.py` (copied by `scripts/collect_items.py`) |
| 6 | Comparison with published results on the same dataset, + what to copy / do differently | `item6_comparison.md` (read this), `item6_comparison.csv` | `scripts/item6_comparison.py` |
| 8 | LSTM on the microstate sequences (maps refitted per fold), with a shuffled-order control | `../lstm/lstm_results.csv`, `../lstm/lstm_per_subject.csv` | `scripts/item8_lstm_sequences.py` |
| 7 | Clustering vs diagnosis | `item7_clustering.csv`, `item7_contingency_tables.csv`, `item7_kmeans_pca.png` | `scripts/item7_clustering.py` |

Baseline check (24 vs 16 features, 11 classifiers): `baseline_24_vs_16.csv`, made by `scripts/baseline_24_vs_16.py`.
Transition heatmap: `outputs/transition_matrix_heatmap.png` (`scripts/plot_transition_matrix.py`).
Why 4 maps: GEV for k = 2–8 in `outputs/gev_by_k.csv` / `.png` (`scripts/gev_by_k.py`); k = 4 gives 67.4%.

Related: the permutation test (is a result better than chance?) is in `outputs/permutation/`.

**Update (November 2026): maps refitted inside every fold.** Items 1, 4 and 5 above used maps fitted
once on all 28 subjects (no labels, but the test subject's EEG was included). `outputs/nested_loso/`
repeats the 12 classifiers with the maps refitted without the test subject in every fold
(`scripts/nested_loso_refit_maps.py`), and the permutation test on those results
(`scripts/permutation_test_nested.py`). These are now the main classification results:
L1 selection + logistic regression 82.1% with s14 (p = 0.011, corrected), 66.7% without s14 (p = 0.153).
PCA (Lab 10 A3, part of item 4) is in `outputs/labs/lab10_a3_pca.csv`.

State names: state0 = A, state2 = B, state3 = C, state1 = D (C and D are the least certain).
