"""
Lab 5 A3-A7 on the project data:
  A3 train/test split (70/30, stratified)   A4 sklearn kNN k=3   A5 score   A6 predict
  A7 the same with your own MyKNN (fit / predict / score)

Note for the report: with 28 subjects the test set is only 9 subjects, so one split is very
noisy (one subject = 11 points). Lab 5 A8-A9 also gives the leave-one-subject-out result.

Run:  eeg_env\\Scripts\\python.exe labs\\lab05_a3_a7_split.py
"""

from lab_common import FEATURE_LIST, SEED, TABLES, load_feature_names, load_table
from lab05_knn import MyKNN, min_max_apply, min_max_fit

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

K = 3

# Step 1 (A3): split the subjects 70/30, keeping the healthy/patient ratio in both parts
df = load_table(TABLES["with s14"])
features = load_feature_names(FEATURE_LIST)
X, y, subjects = df[features].values.tolist(), df["label"].tolist(), df["subject"].tolist()
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, subjects, test_size=0.3, random_state=SEED, stratify=y)
print(f"A3. Train: {len(X_train)} subjects, test: {len(X_test)} subjects ({s_test})")

# Step 2: min-max scaling learned from the training subjects only
mins, maxs = min_max_fit(X_train)
X_train, X_test = min_max_apply(X_train, mins, maxs), min_max_apply(X_test, mins, maxs)

# Step 3 (A4-A6): sklearn kNN
sk = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
print(f"\nA4-A5. sklearn kNN (k={K}): test accuracy {sk.score(X_test, y_test):.3f}, "
      f"train accuracy {sk.score(X_train, y_train):.3f}")
print(f"A6.    sklearn predictions: {sk.predict(X_test).tolist()}")

# Step 4 (A7): your own kNN with fit / predict / score
own = MyKNN(k=K, p=2, sort_algorithm="merge").fit(X_train, y_train)
print(f"\nA7. Own kNN (k={K}): test accuracy {own.score(X_test, y_test):.3f}, "
      f"train accuracy {own.score(X_train, y_train):.3f}")
print(f"    Own predictions:     {own.predict(X_test)}")
print(f"    Actual test labels:  {y_test}")
