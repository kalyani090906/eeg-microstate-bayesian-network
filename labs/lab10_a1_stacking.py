"""
Lab 10 A1: stacking classifier. Base models = classifiers from Lab 9 A3; several meta-models
(final_estimator) are compared, plus plain soft voting (just averaging the base models) as a
reference. Everything is scored with the same leave-one-subject-out evaluate().

Run:  eeg_env\\Scripts\\python.exe labs\\lab10_a1_stacking.py
"""

from lab_common import SEED, run_models, save

from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

INNER_FOLDS = 5   # the stack learns from out-of-fold predictions on the training subjects


def base_models():
    # Step 1: the base classifiers (each scales its own inputs)
    return [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("ada", AdaBoostClassifier(n_estimators=100, random_state=SEED)),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("nb", GaussianNB()),
    ]


def stack(final_estimator):
    # Step 2: a stack = base models + a meta-model trained on their out-of-fold probabilities
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    return StackingClassifier(estimators=base_models(), final_estimator=final_estimator,
                              cv=inner, stack_method="predict_proba")


MODELS = {
    "soft voting (reference)": lambda: VotingClassifier(base_models(), voting="soft"),
    "stack + logistic regression": lambda: stack(LogisticRegression(C=0.1, random_state=SEED)),
    "stack + random forest": lambda: stack(RandomForestClassifier(n_estimators=100, random_state=SEED)),
    "stack + SVM": lambda: stack(SVC(kernel="linear", C=0.1, random_state=SEED)),
    "stack + kNN": lambda: stack(KNeighborsClassifier(n_neighbors=5)),
    # Non-negative weights: the meta-model can only ADD UP base models, never flip them
    "stack + ridge, positive weights": lambda: stack(RidgeClassifier(positive=True, random_state=SEED)),
}

if __name__ == "__main__":
    # Step 3: score every stack (base models already scale, so scale=False)
    results = run_models(MODELS, scale=False)
    save(results, "lab10_a1_stacking.csv")
    print("Compare with the single models in lab09_a3_classifier_table.csv.")
    print("Why some stacks score far BELOW 50%: the meta-model learns from out-of-fold predictions on")
    print("~22 subjects. With balanced classes, leaving subjects out tilts the training majority")
    print("against them, so those predictions look anti-correlated with the truth and the meta-model")
    print("learns to FLIP the base models - which is wrong on the new subject. The positive-weights")
    print("meta-model cannot flip, so it behaves like voting. Report this; it is a known small-sample")
    print("effect, not a code error.")
