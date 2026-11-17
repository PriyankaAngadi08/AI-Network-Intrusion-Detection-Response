
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IncidentCreate(BaseModel):
    source_ip: str = Field(min_length=1, max_length=45)
    destination_ip: str = Field(min_length=1, max_length=45)
    attack_type: str = Field(min_length=1, max_length=100)
    severity: str = "MEDIUM"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str = ""


class IncidentResponse(IncidentCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    created_at: datetime
