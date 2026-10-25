"""
Lab 8 A12: MLPClassifier on the project data (16 microstate features),
leave-one-subject-out, with and without s14. A few network sizes are compared.

Run:  eeg_env\\Scripts\\python.exe labs\\lab08_a12_mlp.py
"""

from lab_common import SEED, run_models, save

from sklearn.neural_network import MLPClassifier

# Step 1: a few hidden-layer sizes (scaling is added by evaluate, inside each training fold)
MODELS = {
    f"MLP {size}": (lambda size=size: MLPClassifier(hidden_layer_sizes=size, max_iter=2000,
                                                    random_state=SEED))
    for size in [(5,), (10,), (20,), (10, 5)]
}

if __name__ == "__main__":
    # Step 2: run them on both tables and save
    results = run_models(MODELS)
    save(results, "lab08_a12_mlp.csv")
