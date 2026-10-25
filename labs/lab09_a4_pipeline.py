"""
Lab 9 A4 (also part of project item 5): a Pipeline that runs several processing steps and the
classifier together:  scale -> pick the top-k features (ANOVA F-score) -> classify.

Because the whole pipeline is refitted inside every leave-one-subject-out fold, the feature
choice only ever sees the training subjects (no leakage).

Run:  eeg_env\\Scripts\\python.exe labs\\lab09_a4_pipeline.py
"""

from lab_common import SEED, run_models, save

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 1: the classifier at the end of the pipeline (change after Lab 9 A3 if you like)
def classifier():
    return LogisticRegression(max_iter=5000, random_state=SEED)


def make_pipeline_k(k):
    # Step 2: the three steps, in order
    return Pipeline([
        ("scale", StandardScaler()),                     # put every feature on the same scale
        ("select", SelectKBest(f_classif, k=k)),         # keep the k most group-separating features
        ("clf", classifier()),                           # classify patient vs control
    ])


MODELS = {f"pipeline top-{k}": (lambda k=k: make_pipeline_k(k)) for k in [4, 8, 12, 16]}

if __name__ == "__main__":
    # Step 3: score every pipeline (it already scales, so scale=False)
    results = run_models(MODELS, scale=False)
    save(results, "lab09_a4_pipeline.csv")
    print("top-16 = all features (selection does nothing); compare it with the smaller k.")
