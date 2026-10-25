"""
Lab 9 A3 (also project item 1): table of many classifiers on the 16 microstate features,
with several metrics, leave-one-subject-out, with and without s14.

Run:  eeg_env\\Scripts\\python.exe labs\\lab09_a3_classifiers.py
Needs: pip install xgboost catboost
"""

from lab_common import SEED, run_models, save

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Step 1: the classifiers (each entry builds a NEW untrained model; scaling is added by evaluate)
MODELS = {
    "Logistic regression": lambda: LogisticRegression(max_iter=5000, random_state=SEED),
    # SVM gives no probabilities by itself; CalibratedClassifierCV adds them (inside the training fold)
    "SVM (linear)": lambda: CalibratedClassifierCV(SVC(kernel="linear", random_state=SEED), ensemble=False),
    "SVM (RBF)": lambda: CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), ensemble=False),
    "Decision tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED),
    "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.1,
                                     random_state=SEED, verbosity=0),
    "CatBoost": lambda: CatBoostClassifier(iterations=300, depth=3, random_seed=SEED,
                                           verbose=0, allow_writing_files=False),
    "Naive Bayes": lambda: GaussianNB(),
    "kNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=SEED),
}

if __name__ == "__main__":
    # Step 2: run every classifier on both tables
    results = run_models(MODELS)
    save(results, "lab09_a3_classifier_table.csv")

    # Step 3: name the best classifier on the main table (by balanced accuracy, then AUC)
    main = results[results["table"] == "with s14"].sort_values(
        ["balanced_accuracy", "auc"], ascending=False)
    best = main.iloc[0]
    print(f"\nBest on the main table: {best['model']} "
          f"({best['accuracy']:.1%}, 95% CI {best['ci_low']:.0%}-{best['ci_high']:.0%}).")
    print("Caution: picking the best of 11 models flatters it. Treat it as a candidate, not a result,")
    print("until the permutation test (a later step) shows it beats shuffled labels.")
