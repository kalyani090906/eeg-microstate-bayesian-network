"""
collect_items.py - copy the item results that are produced by LAB scripts into outputs/items/,
so every project item can be found in one folder.

  item 1 (features + ML classifiers) <- labs/lab09_a3_classifiers.py
  item 5 (pipeline + LIME)           <- labs/lab09_a4_pipeline.py, labs/lab09_a5_lime.py

The lab scripts stay the source: re-run them first, then this script refreshes the copies.

Run:  eeg_env\\Scripts\\python.exe scripts\\collect_items.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LAB_OUT = PROJECT_DIR / "outputs" / "labs"
ITEMS = PROJECT_DIR / "outputs" / "items"

# copy name in outputs/items  <-  file in outputs/labs
COPIES = {
    "item1_classifiers.csv": "lab09_a3_classifier_table.csv",
    "item5_pipeline.csv": "lab09_a4_pipeline.csv",
    "item5_lime_logistic_with_s14.png": "lab09_a5_lime_logistic_with_s14.png",
    "item5_lime_logistic_with_s14.csv": "lab09_a5_lime_logistic_with_s14.csv",
    "item5_lime_logistic_without_s14.png": "lab09_a5_lime_logistic_without_s14.png",
    "item5_lime_logistic_without_s14.csv": "lab09_a5_lime_logistic_without_s14.csv",
    "item5_lime_adaboost_with_s14.png": "lab09_a5_lime_adaboost_with_s14.png",
    "item5_lime_adaboost_with_s14.csv": "lab09_a5_lime_adaboost_with_s14.csv",
    "item5_lime_adaboost_without_s14.png": "lab09_a5_lime_adaboost_without_s14.png",
    "item5_lime_adaboost_without_s14.csv": "lab09_a5_lime_adaboost_without_s14.csv",
}
# --------------------------------------------------------------

import shutil

ITEMS.mkdir(parents=True, exist_ok=True)
missing = []
for copy_name, source in COPIES.items():
    src = LAB_OUT / source
    if not src.exists():
        missing.append(source)
        continue
    shutil.copy2(src, ITEMS / copy_name)          # copy2 keeps the original file date
    print(f"  {source:42s} -> items/{copy_name}")

if missing:
    print("\nNot found (run the lab script first):", ", ".join(missing))
else:
    print(f"\nAll {len(COPIES)} files copied to {ITEMS}")
