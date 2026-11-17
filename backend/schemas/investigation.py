
from typing import Any, Literal

from pydantic import BaseModel, Field


class InvestigationCreate(BaseModel):
    incident_id: int
    report_data: dict[str, Any] = Field(default_factory=dict)
    processing_status: Literal[
        "PENDING", "PROCESSING", "COMPLETED", "FAILED"
    ] = "PENDING"


class InvestigationResponse(BaseModel):
    id: int
    incident_id: int
    report_data: dict[str, Any]
    processing_status: str
    workflow_status: str
    created_at: Any
    updated_at: Any

    model_config = {"from_attributes": True}


class WorkflowUpdate(BaseModel):
    workflow_status: Literal[
        "NEW", "IN_REVIEW", "REVIEWED", "RESOLVED"
    ]


class InvestigationReportUpdate(BaseModel):
    report_data: dict[str, Any]
    processing_status: Literal["PROCESSING", "COMPLETED", "FAILED"]
