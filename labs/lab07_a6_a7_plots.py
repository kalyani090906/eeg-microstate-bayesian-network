"""
Lab 7 A6 + A7 on the project data:
  A6 - draw scikit-learn's decision tree (entropy, depth 3) fitted on all 28 subjects
  A7 - decision boundary of a tree that uses only 2 features

The two features for A7 are the two with the highest information gain (your own function).
Choosing them with the labels on all subjects is fine for a PICTURE; it is not a performance
number (the scores are in lab07_tree.py and lab07_a8_gridsearch.py).

Run:  eeg_env\\Scripts\\python.exe labs\\lab07_a6_a7_plots.py
"""

from lab_common import FEATURE_LIST, LAB_OUT, SEED, TABLES, load_feature_names, load_table
from lab07_tree import information_gain

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = load_table(TABLES["with s14"])
features = load_feature_names(FEATURE_LIST)
X, y = df[features], df["label"].values

# A6: draw the tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED).fit(X, y)
plt.figure(figsize=(16, 9))
plot_tree(clf, feature_names=features, class_names=["healthy", "schizophrenia"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision tree (entropy, max depth 3), 28 subjects, 16 features")
plt.savefig(LAB_OUT / "lab07_a6_tree.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"A6. Saved: {LAB_OUT / 'lab07_a6_tree.png'}")

# A7: pick the 2 most informative features, then draw the 2-feature tree's decision regions
gains = {f: information_gain(X[f].tolist(), y.tolist()) for f in features}
feat1, feat2 = sorted(gains, key=gains.get, reverse=True)[:2]
print(f"A7. Features: {feat1} (gain {gains[feat1]:.3f}), {feat2} (gain {gains[feat2]:.3f})")
X2 = X[[feat1, feat2]].values
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED).fit(X2, y)

pad1, pad2 = np.ptp(X2[:, 0]) * 0.05, np.ptp(X2[:, 1]) * 0.05
xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - pad1, X2[:, 0].max() + pad1, 300),
                     np.linspace(X2[:, 1].min() - pad2, X2[:, 1].max() + pad2, 300))
Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
for label, marker, name in [(0, "o", "healthy"), (1, "^", "schizophrenia")]:
    plt.scatter(X2[y == label, 0], X2[y == label, 1], marker=marker, edgecolors="k",
                c="#2471a3" if label == 0 else "#c0392b", label=name)
plt.xlabel(feat1), plt.ylabel(feat2), plt.legend()
plt.title(f"Decision boundary (tree, depth 3): {feat1} vs {feat2}")
plt.tight_layout()
plt.savefig(LAB_OUT / "lab07_a7_boundary.png", dpi=160)
plt.close()
print(f"    Saved: {LAB_OUT / 'lab07_a7_boundary.png'}")
print(f"    Training accuracy of the 2-feature tree: {clf2.score(X2, y):.3f} (on the same subjects,"
      " so optimistic)")
