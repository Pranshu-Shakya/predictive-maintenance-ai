def detect_fault(sensor_data, failure_probability):

    temperature = sensor_data["Temperature"]
    vibration = sensor_data["Vibration"]
    pressure = sensor_data["Pressure"]
    rpm = sensor_data["RPM"]
    flow_rate = sensor_data["Flow_Rate"]

    # Normal condition
    if failure_probability < 0.40:
        return "Normal Operation"

    # Bearing-related condition
    if vibration >= 7 and temperature >= 80:
        return "Bearing Failure"

    # Cavitation condition
    if pressure < 6.5 and flow_rate < 90:
        return "Cavitation"

    # Misalignment condition
    if rpm < 1250 or rpm > 1650:
        return "Shaft Misalignment"

    # General mechanical problem
    return "General Mechanical Failure"