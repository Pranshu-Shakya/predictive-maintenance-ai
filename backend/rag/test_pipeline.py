from generate_response import generate_troubleshooting_response


sensor_data = {
    "Temperature": 87,
    "Vibration": 7.2,
    "Pressure": 7.4,
    "RPM": 1450,
    "Operating_Hours": 8200,
    "Flow_Rate": 98
}


predicted_fault = "Bearing Failure"

failure_probability = 0.81


response = generate_troubleshooting_response(
    sensor_data,
    predicted_fault,
    failure_probability
)


print("\n")
print("=" * 70)
print("AI TROUBLESHOOTING REPORT")
print("=" * 70)

print(response)