import os
import pickle
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
model_path = os.path.join(OUTPUT_DIR, "shared_microstate_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Cluster centers shape:", model.cluster_centers_.shape)
print("GEV (goodness of fit):", model.GEV_)

# Plot the 4 microstate topographic maps (this is the classic
# "4 colorful brain-map" figure you'll have seen in EEG papers)
fig = model.plot()
plt.savefig(os.path.join(OUTPUT_DIR, "microstate_maps.png"), dpi=150)
plt.show()
print("Saved topographic maps to outputs/microstate_maps.png")
