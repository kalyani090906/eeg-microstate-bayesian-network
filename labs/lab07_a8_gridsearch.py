"""
Lab 7 A8 on the project data: GridSearchCV to tune the decision tree.

Change from the original, and why: the original reported GridSearchCV's best_score_ as the
accuracy. That number is optimistic, because the same subjects chose the settings AND scored
them. Here the grid search runs INSIDE each leave-one-subject-out fold (nested), so the
held-out subject never helps choose. The settings picked on all subjects are printed separately,
as description only.

Run:  eeg_env\\Scripts\\python.exe labs\\lab07_a8_gridsearch.py
"""

from lab_common import FEATURE_LIST, SEED, TABLES, load_feature_names, load_table, run_models, save

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

PARAM_GRID = {
    "max_depth": [1, 2, 3, 4, 5],
    "criterion": ["entropy", "gini"],
    "min_samples_split": [2, 3, 4],
}


def tuned_tree():
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return GridSearchCV(DecisionTreeClassifier(random_state=SEED), PARAM_GRID, cv=inner,
                        scoring="accuracy", n_jobs=-1)


MODELS = {
    "tree, default settings": lambda: DecisionTreeClassifier(random_state=SEED),
    "tree, depth 3 (entropy)": lambda: DecisionTreeClassifier(criterion="entropy", max_depth=3,
                                                              random_state=SEED),
    "tree, GridSearchCV (nested)": tuned_tree,
}

if __name__ == "__main__":
    # Step 1: honest scores (trees do not need scaling)
    results = run_models(MODELS, scale=False)
    save(results, "lab07_a8_gridsearch.csv")

    # Step 2: description only - what the search picks on all 28 subjects
    df = load_table(TABLES["with s14"])
    features = load_feature_names(FEATURE_LIST)
    search = tuned_tree().fit(df[features], df["label"])
    print(f"\nSettings chosen on all 28 subjects: {search.best_params_}")
    print(f"Its inner cross-validation score: {search.best_score_:.3f} "
          "(optimistic - it is the best of 30 settings on the same subjects; do not report it as accuracy)")
