from typing import Optional
from pydantic import BaseModel, Field


class SecurityAlert(BaseModel):
    """
    Common format for security alerts coming from
    different intrusion detection datasets/models.
    """

    alert_id: str

    timestamp: Optional[str] = None

    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None

    source_port: Optional[int] = None
    destination_port: Optional[int] = None

    protocol: Optional[str] = None

    attack_type: str

    label: str

    confidence: float = Field(ge=0.0, le=1.0)

    dataset: Optional[str] = None