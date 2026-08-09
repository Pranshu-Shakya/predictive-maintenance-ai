import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load dataset
data_path = Path(__file__).resolve().parent.parent / "data" / "machine_failure.csv"
if not data_path.exists():
	raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py to create it.")

df = pd.read_csv(data_path)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Number of duplicates:", df.duplicated().sum())


print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

print("\n" + "=" * 60)
print("FAILURE DISTRIBUTION")
print("=" * 60)

print(df["Failure"].value_counts())

print("\nFailure percentage:")
print(df["Failure"].value_counts(normalize=True) * 100)


plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Failure (0 = Normal, 1 = Failure)")
plt.ylabel("Number of Machines")

plt.show()


print("\n" + "=" * 60)
print("FAILURE TYPE DISTRIBUTION")
print("=" * 60)

print(df["Failure_Type"].value_counts())


plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    y="Failure_Type"
)

plt.title("Failure Type Distribution")
plt.xlabel("Number of Machines")
plt.ylabel("Failure Type")

plt.show()


numeric_columns = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Operating_Hours",
    "Flow_Rate",
    "Failure"
]

correlation = df[numeric_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation)


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()


features = [
    "Temperature",
    "Vibration",
    "Pressure",
    "RPM",
    "Operating_Hours",
    "Flow_Rate"
]

for feature in features:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=df,
        x="Failure",
        y=feature
    )

    plt.title(f"{feature} vs Failure")
    plt.xlabel("Failure (0 = Normal, 1 = Failure)")
    plt.ylabel(feature)

    plt.show()



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




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)