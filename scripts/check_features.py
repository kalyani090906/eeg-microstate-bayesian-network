import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
csv_path = os.path.join(OUTPUT_DIR, "eeg_features.csv")

df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumn names:")
print(list(df.columns))

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "None found")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nBasic stats (mean/std/min/max) per feature:")
print(df.describe().T[["mean", "std", "min", "max"]])

print("\nAny infinite values?")
import numpy as np
numeric_df = df.select_dtypes(include=[np.number])
inf_check = np.isinf(numeric_df).sum()
print(inf_check[inf_check > 0] if inf_check.sum() > 0 else "None found")