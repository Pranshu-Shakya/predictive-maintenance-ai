from fault_predictor import predict_fault


# ==========================================
# Test Sensor Data
# ==========================================

sensor_data = {
    "Temperature": 87,
    "Vibration": 7.2,
    "Pressure": 7.4,
    "RPM": 1450,
    "Operating_Hours": 8200,
    "Flow_Rate": 98,
}


# ==========================================
# Predict Fault
# ==========================================

fault, probabilities = predict_fault(
    sensor_data
)


# ==========================================
# Display Results
# ==========================================

print("=" * 60)

print("PREDICTED FAULT:")
print(fault)

print("\nFAULT PROBABILITIES:")

for fault_name, probability in probabilities.items():

    print(
        f"{fault_name}: "
        f"{probability:.2%}"
    )