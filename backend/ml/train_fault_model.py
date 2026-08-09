import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ==========================================
# Load dataset
# ==========================================

from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "machine_failure.csv"
if not data_path.exists():
	raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py to create it.")

df = pd.read_csv(data_path)



# ==========================================
# Features
# ==========================================

FEATURES = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Operating_Hours",
    "Flow_Rate",
]

TARGET = "Fault_Type"


X = df[FEATURES]
y = df[TARGET]


# ==========================================
# Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ==========================================
# Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)


# ==========================================
# Train
# ==========================================

model.fit(
    X_train,
    y_train,
)


# ==========================================
# Predict
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print("=" * 60)
print("FAULT CLASSIFICATION MODEL")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)


# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_,
})

importance = importance.sort_values(
    by="Importance",
    ascending=False,
)

print("\nFeature Importance:")

print(importance)


# ==========================================
# Save model
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
    },
    "fault_model.pkl",
)

print("\nFault model saved successfully:")
print("ml/fault_model.pkl")