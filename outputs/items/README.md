# Project items: where each result is

All results use leave-one-subject-out validation (except item 3) and are reported twice:
with s14 (28 subjects, main) and without s14 (27 subjects, sensitivity check).
With 28 subjects every result is exploratory; one subject is about 3.6 percentage points.

| Item | What it is | Files here | Made by |
|---|---|---|---|
| 1 | 16 microstate features + 11 ML classifiers | `item1_classifiers.csv` | `labs/lab09_a3_classifiers.py` (copied by `scripts/collect_items.py`) |
| 2 | Frozen LaBraM embeddings + classifiers | `item2_embeddings.csv` | `scripts/item2_embeddings.py` |
| 3 | Fine-tuned LaBraM (7-fold subject-grouped, a different protocol) | `item3_finetune/` | `scripts/item3_finetune.py` |
| 4 | Feature reduction: 24 vs 16 vs top-k vs L1 | `item4_feature_reduction.csv`, `item4_top4_selection_counts.csv` | `scripts/item4_feature_reduction.py` |
| 5 | Pipeline (scale, select top-k, classify) + LIME | `item5_pipeline.csv`, `item5_lime_*.png` / `.csv` | `labs/lab09_a4_pipeline.py`, `labs/lab09_a5_lime.py` (copied by `scripts/collect_items.py`) |
| 6 | Comparison with published results | (to do) | |
| 7 | Clustering vs diagnosis | `item7_clustering.csv`, `item7_contingency_tables.csv`, `item7_kmeans_pca.png` | `scripts/item7_clustering.py` |

Related: the permutation test (is a result better than chance?) is in `outputs/permutation/`.
PCA (Lab 10 A3, part of item 4) is in `outputs/labs/lab10_a3_pca.csv`.

State names: state0 = A, state2 = B, state3 = C, state1 = D (C and D are the least certain).
