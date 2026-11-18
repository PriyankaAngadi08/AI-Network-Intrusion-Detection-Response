
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field
from typing import Optional
import requests


class SecurityAlert(BaseModel):
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


def create_security_alert(
    detection_result: dict,
    flow_metadata: Optional[dict] = None,
    dataset: Optional[str] = None
) -> Optional[SecurityAlert]:

    if not isinstance(detection_result, dict):
        raise TypeError("Detection result must be a dictionary.")

    if detection_result.get("status") != "Attack":
        return None

    attack_type = detection_result.get("attack_type")

    if not attack_type:
        raise ValueError("Attack type is required for an attack alert.")

    metadata = flow_metadata or {}

    alert = SecurityAlert(
        alert_id=f"ALERT-{uuid4().hex[:12].upper()}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_ip=metadata.get("source_ip"),
        destination_ip=metadata.get("destination_ip"),
        source_port=metadata.get("source_port"),
        destination_port=metadata.get("destination_port"),
        protocol=metadata.get("protocol"),
        attack_type=str(attack_type),
        label=str(detection_result.get("prediction", attack_type)),
        confidence=float(detection_result["confidence"]),
        dataset=dataset
    )

    return alert


def send_security_alert(
    alert: SecurityAlert,
    api_url: str,
    timeout: float = 10.0
) -> dict:

    if not isinstance(alert, SecurityAlert):
        raise TypeError("alert must be a SecurityAlert object.")

    if not api_url or not isinstance(api_url, str):
        raise ValueError("A valid API URL is required.")

    try:
        response = requests.post(
            api_url,
            json=alert.model_dump(mode="json"),
            timeout=timeout
        )

        response.raise_for_status()

        try:
            response_data = response.json()
        except ValueError:
            response_data = {"message": response.text}

        return {
            "success": True,
            "status_code": response.status_code,
            "response": response_data
        }

    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "error": str(error)
        }