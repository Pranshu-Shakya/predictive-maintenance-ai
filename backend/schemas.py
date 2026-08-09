from pydantic import BaseModel, Field


class SensorData(BaseModel):

    Temperature: float = Field(
        ...,
        description="Machine temperature in Celsius"
    )

    Vibration: float = Field(
        ...,
        description="Machine vibration in mm/s"
    )

    Pressure: float = Field(
        ...,
        description="Pump pressure in bar"
    )

    RPM: float = Field(
        ...,
        description="Rotational speed"
    )

    Operating_Hours: float = Field(
        ...,
        description="Total machine operating hours"
    )

    Flow_Rate: float = Field(
        ...,
        description="Flow rate in L/min"
    )