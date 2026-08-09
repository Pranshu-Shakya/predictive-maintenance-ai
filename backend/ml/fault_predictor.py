import joblib
import pandas as pd
from pathlib import Path


# Load model from possible locations relative to this file
possible_paths = [
    Path(__file__).resolve().parent / "fault_model.pkl",
    Path(__file__).resolve().parent.parent / "fault_model.pkl",
]

for p in possible_paths:
    if p.exists():
        saved_model = joblib.load(str(p))
        break
else:
    raise FileNotFoundError(
        "fault_model.pkl not found. Checked: "
        + ", ".join(str(p) for p in possible_paths)
    )

model = saved_model["model"]

FEATURES = saved_model["features"]


def predict_fault(sensor_data):

    input_data = pd.DataFrame(
        [[
            sensor_data["Temperature"],
            sensor_data["Vibration"],
            sensor_data["Pressure"],
            sensor_data["RPM"],
            sensor_data["Operating_Hours"],
            sensor_data["Flow_Rate"],
        ]],
        columns=FEATURES,
    )

    prediction = model.predict(
        input_data
    )[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    classes = model.classes_

    probability_map = {
        class_name: float(probability)
        for class_name, probability
        in zip(classes, probabilities)
    }

    return prediction, probability_map