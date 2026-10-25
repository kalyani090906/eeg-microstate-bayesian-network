"""
Lab 9 A2: RandomizedSearchCV to tune the Perceptron and the MLP.

The search runs INSIDE each leave-one-subject-out fold, on the 27 training subjects only
(nested cross-validation), so the held-out subject never influences the chosen settings.
Then, only to show which settings it likes, the search is run once on all subjects
(that last step is description, NOT a performance number).

Run:  eeg_env\\Scripts\\python.exe labs\\lab09_a2_tuning.py      (takes a few minutes)
"""

from lab_common import FEATURE_LIST, SEED, TABLES, load_feature_names, load_table, run_models, save

from scipy.stats import loguniform
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_ITER = 15        # settings tried per search
INNER_FOLDS = 5    # inner cross-validation on the training subjects

# Step 1: what to search over
PERCEPTRON_SPACE = {
    "clf__penalty": [None, "l2", "l1", "elasticnet"],
    "clf__alpha": loguniform(1e-5, 1e-1),
    "clf__eta0": loguniform(1e-3, 1.0),
}
MLP_SPACE = {
    "clf__hidden_layer_sizes": [(5,), (10,), (20,), (10, 5)],
    "clf__activation": ["relu", "tanh", "logistic"],
    "clf__alpha": loguniform(1e-4, 10),
    "clf__learning_rate_init": loguniform(1e-4, 1e-1),
}


def make_search(model, space):
    # Step 2: scaler + model in one pipeline, wrapped in a randomized search
    pipe = Pipeline([("scale", StandardScaler()), ("clf", model)])
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    return RandomizedSearchCV(pipe, space, n_iter=N_ITER, cv=inner, scoring="balanced_accuracy",
                              random_state=SEED, n_jobs=-1)


MODELS = {
    "Perceptron (default)": lambda: Pipeline([("scale", StandardScaler()),
                                              ("clf", Perceptron(random_state=SEED))]),
    "Perceptron (tuned)": lambda: make_search(Perceptron(max_iter=1000, random_state=SEED),
                                              PERCEPTRON_SPACE),
    "MLP (default)": lambda: Pipeline([("scale", StandardScaler()),
                                       ("clf", MLPClassifier(max_iter=2000, random_state=SEED))]),
    "MLP (tuned)": lambda: make_search(MLPClassifier(max_iter=2000, random_state=SEED),
                                       MLP_SPACE),
}

if __name__ == "__main__":
    # Step 3: honest scores (search inside every fold); models are full pipelines, so scale=False
    results = run_models(MODELS, scale=False)
    save(results, "lab09_a2_tuning.csv")
    print("Note: the Perceptron gives no probabilities, so its AUC is computed from 0/1 votes.")

    # Step 4: description only - which settings does the search pick on the whole main table?
    df = load_table(TABLES["with s14"])
    features = load_feature_names(FEATURE_LIST)
    for name in ["Perceptron (tuned)", "MLP (tuned)"]:
        search = MODELS[name]().fit(df[features].values, df["label"].values)
        print(f"\n{name} - settings chosen on all 28 subjects (not a performance number):")
        for key, value in search.best_params_.items():
            print(f"  {key.replace('clf__', '')}: {value}")
