import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Load dataset
# df = pd.read_csv("../data/machine_failure.csv")
from pathlib import Path
data_path = Path(__file__).resolve().parent.parent / "data" / "machine_failure.csv"
if not data_path.exists():
	raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py to create it.")

df = pd.read_csv(data_path)

FEATURES = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Operating_Hours",
    "Flow_Rate"
]

X = df[FEATURES]
y = df["Failure"]


# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Load model
saved = joblib.load("model.pkl")
model = saved["model"]


# Predictions
y_pred = model.predict(X_test)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)


print("Confusion Matrix:")
print(cm)


# Visualization
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()