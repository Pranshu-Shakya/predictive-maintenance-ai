import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================
# 1. Load dataset
# ==========================================

from pathlib import Path

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
# 4. Create Random Forest model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 5. Train model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 6. Make predictions
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. Evaluate model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ==========================================
# 8. Classification report
# ==========================================

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Failure"],
    zero_division=0
))


# ==========================================
# 9. Confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 10. Feature importance
# ==========================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# ==========================================
# 11. Save model
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    "model.pkl"
)

print("\nModel saved successfully as model.pkl")