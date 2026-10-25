"""
Lab 10 A2: feature correlation analysis with a heatmap (the 16 features, all 28 subjects).
No labels are used, so using all subjects here is fine.

Run:  eeg_env\\Scripts\\python.exe labs\\lab10_a2_heatmap.py
"""

from lab_common import FEATURE_LIST, LAB_OUT, TABLES, load_feature_names, load_table

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

STRONG = 0.8   # |r| above this is reported as a strongly correlated pair

# Step 1: load the main table and the 16 feature names
df = load_table(TABLES["with s14"])
features = load_feature_names(FEATURE_LIST)
corr = df[features].corr()

# Step 2: draw the heatmap (lower triangle only, so each pair appears once)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
plt.figure(figsize=(11, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            annot_kws={"size": 7})
plt.title("Correlation (Pearson r) of the 16 microstate features, 28 subjects")
plt.tight_layout()
plt.savefig(LAB_OUT / "lab10_a2_heatmap.png", dpi=160)
plt.close()

# Step 3: list the strongly correlated pairs
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(features) for b in features[i + 1:]
         if abs(corr.loc[a, b]) > STRONG]
print(f"Pairs with |r| > {STRONG}:")
for a, b, r in sorted(pairs, key=lambda p: -abs(p[2])):
    print(f"  {a} - {b}: r = {r:.2f}")
print(f"\nSaved: {LAB_OUT / 'lab10_a2_heatmap.png'}")
