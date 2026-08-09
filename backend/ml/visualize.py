import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "machine_failure.csv"
if not data_path.exists():
	raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py to create it.")

df = pd.read_csv(data_path)

# Failure distribution
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Failure")

plt.title("Machine Failure Distribution")
plt.xlabel("Failure")
plt.ylabel("Number of Machines")

plt.show()


# Vibration vs Temperature
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Vibration",
    y="Temperature",
    hue="Failure"
)

plt.title("Temperature vs Vibration")
plt.show()