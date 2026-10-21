"""
make_final_features.py   (PROJECT script - put it in MLproject/scripts/)

Step 1 of the final work split: build the FINAL feature table that the whole team uses.

What it does (and why)
  1. Reads outputs/eeg_features_epoch_mean_15min.csv  (28 subjects, 24 features).
  2. Drops REDUNDANT columns, using only the structure of the features - it never looks
     at the label, so it cannot inflate accuracy:
       * coverage columns: the coverages of the 4 states add up to 1, and
         coverage is roughly duration x occurrence, so they carry no new information.
       * ONE transition per source state: the 3 outgoing transitions of a state add up
         to 1, so the third one is fully determined by the other two.
         Fixed rule: for each source state we drop the LAST of its three columns.
  3. Keeps 16 features:  4 durations + 4 occurrences + 8 transitions.
  4. Writes two files with identical columns (subject, label, 16 features):
       outputs/features_final_16.csv           all 28 subjects  (main analysis)
       outputs/features_final_16_no_s14.csv    27 subjects      (sensitivity analysis)
     plus outputs/features_final_16_list.txt   (the feature names, one per line)
  5. Prints a label-free check: pairs of kept features that are still very strongly
     correlated (|r| > 0.9) and how much each feature varies between subjects.

Rule agreed for the project: the first-15-minute window for everybody; every result is
reported with s14 (main) and without s14 (sensitivity). The rule was fixed because s14's
recording is abnormal, not because of anything about the labels.

Run:  python make_final_features.py
"""
import os
import re
import numpy as np
import pandas as pd

# ---------------- settings (change here only) ----------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
IN_FILE = "eeg_features_epoch_mean_15min.csv"
EXCLUDE_SUBJECT = "s14"          # sensitivity analysis leaves this subject out
INCLUDE_COVERAGE = False         # True = keep coverage too (then you have redundant columns)
CORR_FLAG = 0.9                  # report kept pairs with |correlation| above this
# --------------------------------------------------------------

df = pd.read_csv(os.path.join(OUTPUT_DIR, IN_FILE))
features_all = [c for c in df.columns if c not in ("subject", "label")]
print(f"Input: {len(df)} subjects, {len(features_all)} features "
      f"({int((df.label == 0).sum())} healthy, {int((df.label == 1).sum())} schizophrenia)")

# ---- 1) drop redundant columns (structure only, label never used) ----
drop = []
if not INCLUDE_COVERAGE:
    drop += [c for c in features_all if c.endswith("_coverage")]

# transitions grouped by source state; drop the last one of each group
by_source = {}
for c in features_all:
    m = re.fullmatch(r"trans_(\d+)to(\d+)", c)
    if m:
        by_source.setdefault(int(m.group(1)), []).append(c)
for src, cols in sorted(by_source.items()):
    drop.append(cols[-1])

keep = [c for c in features_all if c not in drop]
print(f"\nDropped {len(drop)} redundant columns:")
for c in drop:
    why = "coverages sum to 1 / = duration x occurrence" if c.endswith("_coverage") \
        else "outgoing transitions of one state sum to 1"
    print(f"   {c:18s} ({why})")
print(f"\nKept {len(keep)} features:")
print("   " + ", ".join(keep))

# ---- 2) write the files ----
final = df[["subject", "label"] + keep]
main_path = os.path.join(OUTPUT_DIR, "features_final_16.csv")
sens_path = os.path.join(OUTPUT_DIR, "features_final_16_no_s14.csv")
final.to_csv(main_path, index=False)
final[final["subject"] != EXCLUDE_SUBJECT].to_csv(sens_path, index=False)
with open(os.path.join(OUTPUT_DIR, "features_final_16_list.txt"), "w") as fh:
    fh.write("\n".join(keep) + "\n")
n_sens = int((final["subject"] != EXCLUDE_SUBJECT).sum())
print(f"\nSaved: features_final_16.csv ({len(final)} rows), "
      f"features_final_16_no_s14.csv ({n_sens} rows), features_final_16_list.txt")

# ---- 3) label-free sanity checks ----
X = final[keep]
assert X.notna().all().all(), "missing values found"
corr = X.corr().abs()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(keep) for b in keep[i + 1:]
         if corr.loc[a, b] > CORR_FLAG]
print(f"\nStill strongly related (|r| > {CORR_FLAG}) among the kept features:")
if pairs:
    for a, b, r in sorted(pairs, key=lambda p: -p[2]):
        print(f"   {a} ~ {b}   r = {r:.2f}")
else:
    print("   none")

print("\nHow much each feature varies between subjects (spread as % of the mean):")
cv = (X.std() / X.mean() * 100).sort_values()
for c, v in cv.items():
    print(f"   {c:18s} {v:5.1f}%")
print("\nNote: nothing above uses the label. Discretising / scaling is still done INSIDE each"
      " training fold by the modelling code.")