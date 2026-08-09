import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================
# Reproducibility
# ==========================================

np.random.seed(42)


# ==========================================
# Dataset configuration
# ==========================================

N = 5000


# Target distribution
# ------------------------------------------
# Normal              -> 50%
# Bearing Failure     -> 20%
# Cavitation          -> 15%
# Shaft Misalignment  -> 15%

fault_types = np.random.choice(
    [
        "Normal",
        "Bearing Failure",
        "Cavitation",
        "Shaft Misalignment",
    ],
    size=N,
    p=[
        0.50,
        0.20,
        0.15,
        0.15,
    ],
)


# ==========================================
# Generate sensor data
# ==========================================

temperature = np.zeros(N)

vibration = np.zeros(N)

pressure = np.zeros(N)

rpm = np.zeros(N)

operating_hours = np.zeros(N)

flow_rate = np.zeros(N)


# ==========================================
# Generate values according to fault type
# ==========================================

for i, fault in enumerate(fault_types):

    # --------------------------------------
    # NORMAL
    # --------------------------------------

    if fault == "Normal":

        temperature[i] = np.random.normal(
            65,
            5
        )

        vibration[i] = np.random.normal(
            2.5,
            0.6
        )

        pressure[i] = np.random.normal(
            8.5,
            0.5
        )

        rpm[i] = np.random.normal(
            1450,
            40
        )

        operating_hours[i] = np.random.uniform(
            500,
            7000
        )

        flow_rate[i] = np.random.normal(
            120,
            8
        )


    # --------------------------------------
    # BEARING FAILURE
    # --------------------------------------

    elif fault == "Bearing Failure":

        temperature[i] = np.random.normal(
            88,
            6
        )

        vibration[i] = np.random.normal(
            7.5,
            1.0
        )

        pressure[i] = np.random.normal(
            8,
            0.7
        )

        rpm[i] = np.random.normal(
            1450,
            60
        )

        operating_hours[i] = np.random.uniform(
            6000,
            10000
        )

        flow_rate[i] = np.random.normal(
            105,
            10
        )


    # --------------------------------------
    # CAVITATION
    # --------------------------------------

    elif fault == "Cavitation":

        temperature[i] = np.random.normal(
            75,
            5
        )

        vibration[i] = np.random.normal(
            6,
            1.0
        )

        pressure[i] = np.random.normal(
            5.8,
            0.5
        )

        rpm[i] = np.random.normal(
            1450,
            50
        )

        operating_hours[i] = np.random.uniform(
            3000,
            9000
        )

        flow_rate[i] = np.random.normal(
            78,
            8
        )


    # --------------------------------------
    # SHAFT MISALIGNMENT
    # --------------------------------------

    elif fault == "Shaft Misalignment":

        temperature[i] = np.random.normal(
            78,
            6
        )

        vibration[i] = np.random.normal(
            6.5,
            1.0
        )

        pressure[i] = np.random.normal(
            8,
            0.7
        )

        # Abnormal RPM
        if np.random.rand() < 0.5:

            rpm[i] = np.random.normal(
                1180,
                50
            )

        else:

            rpm[i] = np.random.normal(
                1720,
                50
            )

        operating_hours[i] = np.random.uniform(
            4000,
            9500
        )

        flow_rate[i] = np.random.normal(
            105,
            10
        )


# ==========================================
# Clip values to realistic ranges
# ==========================================

temperature = np.clip(
    temperature,
    45,
    110
)

vibration = np.clip(
    vibration,
    0.5,
    10
)

pressure = np.clip(
    pressure,
    4.5,
    11
)

rpm = np.clip(
    rpm,
    1100,
    1800
)

operating_hours = np.clip(
    operating_hours,
    500,
    10000
)

flow_rate = np.clip(
    flow_rate,
    50,
    150
)


# ==========================================
# Binary failure target
# ==========================================

failure = (
    fault_types != "Normal"
).astype(int)


# ==========================================
# Create DataFrame
# ==========================================

df = pd.DataFrame({

    "Temperature": temperature,

    "Vibration": vibration,

    "Pressure": pressure,

    "RPM": rpm,

    "Operating_Hours": operating_hours,

    "Flow_Rate": flow_rate,

    "Failure": failure,

    "Fault_Type": fault_types,
})


# ==========================================
# Shuffle dataset
# ==========================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# Save dataset
# ==========================================

output_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "machine_failure.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    output_path,
    index=False
)


# ==========================================
# Print information
# ==========================================

print("Dataset created successfully!")

print(
    f"Number of records: {len(df)}"
)

print("\nFirst 5 rows:")

print(df.head())


print("\nFailure distribution:")

print(
    df["Failure"].value_counts()
)


print("\nFault type distribution:")

print(
    df["Fault_Type"].value_counts()
)


print("\nFault type percentages:")

print(
    df["Fault_Type"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)