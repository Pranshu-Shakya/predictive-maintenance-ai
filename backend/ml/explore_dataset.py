import pandas as pd
from pathlib import Path

# Resolve dataset path relative to this script (backend/data/machine_failure.csv)
data_path = Path(__file__).resolve().parent.parent / "data" / "machine_failure.csv"
if not data_path.exists():
	raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py to create it.")

df = pd.read_csv(data_path)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nFailure distribution:")
print(df["Failure"].value_counts())

print("\nFailure type distribution:")
print(df["Failure_Type"].value_counts())