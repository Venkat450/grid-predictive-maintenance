from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    air_temperature: float = Field(..., gt=0)
    process_temperature: float = Field(..., gt=0)
    rotational_speed: float = Field(..., gt=0)
    torque: float = Field(..., gt=0)
    tool_wear: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    failure_probability: float
    risk_level: str
