
# Bayesian network modeling of EEG microstate transitions (schizophrenia)

ML course project. Pipeline: EEG (RepOD dataset) -> shared microstate model
-> per-subject features (eeg_features.csv) -> Bayesian network -> evaluation.

## Setup
1. python -m venv eeg_env
2. eeg_env\Scripts\activate
3. pip install -r requirements.txt
4. Download the EDF files from
   https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441
   and put them in a folder called dataset/ (h01-h14.edf, s01-s14.edf)

## Folder layout
- scripts/     preprocessing, microstate model, feature extraction
- outputs/     eeg_features.csv (current table) and figures
- network/     Bayesian network code
- evaluation/  cross-validation, permutation test, bootstrap
- stats/       group comparisons
- notes/       paper notes, data-quality notes

## Status
eeg_features.csv is provisional: the microstate settings are being tuned.
'@ | Set-Content README.md
