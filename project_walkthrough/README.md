# Project walkthrough: the whole project, step by step

A long deck for the team to **learn and explain** the project, from the raw EEG to the results.
It is not the presentation deck. The deck itself (`EEG_Microstates_Project_Walkthrough_v2_21Nov2026.pptx`,
131 slides) is **not in git**: it is shared separately. This folder holds what makes it, so it can be
rebuilt and checked.

| Module | Topic |
|---|---|
| 0 | Big picture: pipeline, vocabulary, which file does what, setup and run order |
| 1 | EEG and microstate theory (what a microstate is and is not, the A-D classes, schizophrenia literature) |
| 2 | The dataset and s14 |
| 3 | The data matrix: from 19 x 225,000 to 16 numbers (real counts for h01 and for all 28) |
| 4 | Cleaning the EEG (`scripts/eeg_preprocessing.py`) |
| 5 | GFP and GFP peaks (with the toy notebook) |
| 6 | Finding the 4 maps (modified k-means, GEV, why k = 4, stability, how to read a map) |
| 7 | Labelling and features (formulas, worked example, transitions, 24 -> 16) |
| 8 | Group statistics (rank-sum, Holm, group-specific maps) |
| 9 | How we evaluate (leave-one-subject-out, metrics, the rules) |
| 10 | Classical ML (items 1, 4, 5, 7; classifier settings) |
| 11 | Maps refitted in every fold, the train/test split fold by fold, permutation tests (main results) |
| 12 | Deep learning (LaBraM; LSTM as a limitation) |
| 13 | Comparison with published work |
| 14 | Lab exercises |
| 15 | Results summary, limitations, faculty question bank |

## What is in this folder

| File | What it is |
|---|---|
| `make_walkthrough_figures.py` | the explainer figures in `figures/` (raw vs clean, spectrum, epoch rejection, GFP peaks, labelling one sample / one epoch, polarity, maps with electrode names) and copies the toy-notebook pictures |
| `make_fold_and_matrix_numbers.py` | the train/test split fold by fold (`fold_by_fold_*.csv`), the matrix counts for h01 (`matrix_numbers_h01.csv`) and the matrix picture (`figures/data_matrix.png`) |
| `build_walkthrough_deck.js` | writes the .pptx from the figures and the project outputs, to `../../reports/project_walkthrough/` (outside the repo) |
| `figures/` | the pictures used in the deck (project figures are read from `../outputs/`) |

## Rebuild (only if something changes)

From the `MLproject` folder (the Python scripts need `dataset/` and the files in `outputs/`):

```powershell
eeg_env\Scripts\python.exe project_walkthrough\make_walkthrough_figures.py
eeg_env\Scripts\python.exe project_walkthrough\make_fold_and_matrix_numbers.py
```

The deck itself needs Node.js. From `MLproject\project_walkthrough`:

```powershell
npm install pptxgenjs@3.12.0
node build_walkthrough_deck.js
```

All numbers in the deck come from the project's output files or from these two scripts.
With 28 subjects every result is exploratory.
