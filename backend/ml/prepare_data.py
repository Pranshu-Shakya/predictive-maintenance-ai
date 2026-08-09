import pandas as pd

from sklearn.model_selection import train_test_split


# ==========================================
# 1. Load dataset
# ==========================================

from pathlib import Path

# Resolve dataset path relative to this script (backend/data/machine_failure.csv)
data_path = Path(__file__).resolve().parent.parent / "data" / "machine_failure.csv"
if not data_path.exists():
	raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py to create it.")

df = pd.read_csv(data_path)


# ==========================================
# 2. Define features and target
# ==========================================

FEATURES = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Operating_Hours",
    "Flow_Rate"
]

TARGET = "Failure"


X = df[FEATURES]
y = df[TARGET]


# ==========================================
# 3. Train-test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Display results
# ==========================================

print("Dataset shape:")
print(df.shape)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))

print("\nFeatures:")
print(FEATURES)