from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.auth import (
    verify_admin,
    create_access_token,
    get_current_admin,
)
from backend.database import Base, engine, get_db
from backend.models.incident import Incident
from backend.models.response_action import ResponseAction
from backend.models.investigation import Investigation
from backend.models.audit_log import AuditLog
from backend.schemas.incident import IncidentCreate, IncidentResponse
from backend.schemas.investigation import (
    InvestigationCreate,
    InvestigationResponse,
    WorkflowUpdate,
)
from backend.app.services.investigation_engine import investigate_incident
from response.engine import evaluate_incident


app = FastAPI(
    title="AI-NIDS Response System",
    description="Security Operations, Threat Investigation, and Response Backend",
    version="1.4.0",
)


# -------------------- CORS --------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# -------------------- REQUEST SCHEMAS --------------------

class ApprovalRequest(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    notes: str = ""


class InvestigationReportUpdate(BaseModel):
    report_data: dict[str, Any]
    processing_status: Literal[
        "PROCESSING",
        "COMPLETED",
        "FAILED",
    ]


# Register models before creating missing tables.
Base.metadata.create_all(bind=engine)


# -------------------- AUTHENTICATION --------------------

@app.post("/auth/login", tags=["Authentication"])
def admin_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    if not verify_admin(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(form_data.username)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800,
    }


# -------------------- HEALTH CHECKS --------------------

@app.get("/")
def root():
    return {"message": "AI-NIDS Response System Backend Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/database-health")
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {"database": "connected"}

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed",
        )


# -------------------- INCIDENT MANAGEMENT --------------------

@app.post(
    "/incidents",
    response_model=IncidentResponse,
    status_code=201,
    tags=["Incidents"],
)
def create_incident(
    incident_data: IncidentCreate,
    db: Session = Depends(get_db),
):
    incident = Incident(**incident_data.model_dump())

    try:
        db.add(incident)
        db.commit()
        db.refresh(incident)
        return incident

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save incident",
        )


@app.get(
    "/incidents",
    response_model=list[IncidentResponse],
    tags=["Incidents"],
)
def get_incidents(
    db: Session = Depends(get_db),
):
    return db.scalars(
        select(Incident).order_by(Incident.id.desc())
    ).all()


@app.get(
    "/incidents/{incident_id}",
    response_model=IncidentResponse,
    tags=["Incidents"],
)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
):
    incident = db.get(Incident, incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return incident


@app.post(
    "/incidents/{incident_id}/evaluate",
    tags=["Incidents"],
)
def evaluate_incident_api(
    incident_id: int,
    db: Session = Depends(get_db),
):
    incident = db.get(Incident, incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    # Recommendation only. No security action is executed.
    return evaluate_incident(incident)


# -------------------- RESPONSE ACTIONS --------------------

@app.post(
    "/incidents/{incident_id}/response-actions",
    status_code=201,
    tags=["Response Actions"],
)
def create_response_action(
    incident_id: int,
    db: Session = Depends(get_db),
):
    incident = db.get(Incident, incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    recommendation = evaluate_incident(incident)

    action = ResponseAction(
        incident_id=incident.id,
        recommended_action=recommendation["recommended_action"],
        priority=recommendation["priority"],
        approval_status="PENDING",
        execution_status="NOT_EXECUTED",
        notes="Recommendation saved; no security action executed.",
    )

    try:
        db.add(action)
        db.commit()
        db.refresh(action)

        return {
            "message": "Response recommendation saved",
            "response_action_id": action.id,
            "incident_id": action.incident_id,
            "recommended_action": action.recommended_action,
            "priority": action.priority,
            "approval_status": action.approval_status,
            "execution_status": action.execution_status,
            "created_at": action.created_at,
        }

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save response recommendation",
        )


@app.get(
    "/response-actions",
    tags=["Response Actions"],
)
def get_response_actions(
    db: Session = Depends(get_db),
):
    actions = db.scalars(
        select(ResponseAction).order_by(ResponseAction.id.desc())
    ).all()

    return [
        {
            "id": action.id,
            "incident_id": action.incident_id,
            "recommended_action": action.recommended_action,
            "priority": action.priority,
            "approval_status": action.approval_status,
            "execution_status": action.execution_status,
            "notes": action.notes,
            "created_at": action.created_at,
        }
        for action in actions
    ]


@app.post(
    "/response-actions/{action_id}/approval",
    tags=["Response Actions"],
)
def review_response_action(
    action_id: int,
    request: ApprovalRequest,
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    action = db.get(ResponseAction, action_id)

    if action is None:
        raise HTTPException(
            status_code=404,
            detail="Response action not found",
        )

    if action.approval_status != "PENDING":
        raise HTTPException(
            status_code=409,
            detail="This response action has already been reviewed",
        )

    action.approval_status = request.decision
    action.notes = request.notes

    # Record who reviewed the response action and what decision was made.
    audit_entry = AuditLog(
        response_action_id=action.id,
        incident_id=action.incident_id,
        admin_username=current_admin,
        decision=request.decision,
        notes=request.notes,
    )

    try:
        # Save the decision and audit record in one database transaction.
        db.add(audit_entry)
        db.commit()
        db.refresh(action)
        db.refresh(audit_entry)

        return {
            "message": "Response action reviewed successfully",
            "response_action_id": action.id,
            "incident_id": action.incident_id,
            "approval_status": action.approval_status,
            "execution_status": action.execution_status,
            "reviewed_by": current_admin,
            "notes": action.notes,
            "audit_log_id": audit_entry.id,
            "reviewed_at": audit_entry.reviewed_at,
        }

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not update approval status and audit log",
        )


# -------------------- AUDIT LOGGING --------------------

@app.get(
    "/audit-logs",
    tags=["Audit Logs"],
)
def get_audit_logs(
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    """Return the administrator review history. Admin authentication required."""

    logs = db.scalars(
        select(AuditLog).order_by(AuditLog.reviewed_at.desc())
    ).all()

    return [
        {
            "id": log.id,
            "response_action_id": log.response_action_id,
            "incident_id": log.incident_id,
            "admin_username": log.admin_username,
            "decision": log.decision,
            "notes": log.notes,
            "reviewed_at": log.reviewed_at,
        }
        for log in logs
    ]


# -------------------- MONITORING --------------------

@app.get(
    "/monitoring/incidents",
    tags=["Monitoring"],
)
def monitor_incidents(
    db: Session = Depends(get_db),
):
    incidents = db.scalars(
        select(Incident).order_by(Incident.id.desc())
    ).all()

    result = []

    for incident in incidents:
        actions = db.scalars(
            select(ResponseAction)
            .where(ResponseAction.incident_id == incident.id)
            .order_by(ResponseAction.id.desc())
        ).all()

        result.append({
            "incident_id": incident.id,
            "source_ip": incident.source_ip,
            "destination_ip": incident.destination_ip,
            "attack_type": incident.attack_type,
            "severity": incident.severity,
            "confidence": incident.confidence,
            "incident_status": incident.status,
            "created_at": incident.created_at,
            "response_actions": [
                {
                    "id": action.id,
                    "recommended_action": action.recommended_action,
                    "priority": action.priority,
                    "approval_status": action.approval_status,
                    "execution_status": action.execution_status,
                }
                for action in actions
            ],
        })

    return result


@app.get(
    "/monitoring/summary",
    tags=["Monitoring"],
)
def monitoring_summary(
    db: Session = Depends(get_db),
):
    incidents = db.scalars(select(Incident)).all()
    actions = db.scalars(select(ResponseAction)).all()
    investigations = db.scalars(select(Investigation)).all()

    return {
        "total_incidents": len(incidents),
        "incidents_by_severity": {
            severity: sum(
                1
                for incident in incidents
                if (incident.severity or "").upper() == severity
            )
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        },
        "total_response_actions": len(actions),
        "response_actions_by_approval_status": {
            approval_status: sum(
                1
                for action in actions
                if (action.approval_status or "").upper()
                == approval_status
            )
            for approval_status in ["PENDING", "APPROVED", "REJECTED"]
        },
        "response_actions_not_executed": sum(
            1
            for action in actions
            if action.execution_status == "NOT_EXECUTED"
        ),
        "total_investigations": len(investigations),
        "completed_investigations": sum(
            1
            for investigation in investigations
            if investigation.processing_status == "COMPLETED"
        ),
        "investigations_awaiting_review": sum(
            1
            for investigation in investigations
            if investigation.workflow_status in ("NEW", "IN_REVIEW")
        ),
    }


# -------------------- SECURITY OVERVIEW API --------------------

@app.get(
    "/api/security/overview",
    tags=["Security Dashboard"],
)
def security_overview(
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    incidents = db.scalars(select(Incident)).all()
    investigations = db.scalars(select(Investigation)).all()

    attack_type_counts = {}

    for incident in incidents:
        attack_type = incident.attack_type or "Unknown"
        attack_type_counts[attack_type] = (
            attack_type_counts.get(attack_type, 0) + 1
        )

    return {
        "total_alerts": len(incidents),
        "completed_investigations": sum(
            1
            for investigation in investigations
            if investigation.processing_status == "COMPLETED"
        ),
        "high_critical_alerts": sum(
            1
            for incident in incidents
            if (incident.severity or "").upper()
            in ("HIGH", "CRITICAL")
        ),
        "total_investigations": len(investigations),
        "awaiting_review": sum(
            1
            for investigation in investigations
            if investigation.workflow_status in ("NEW", "IN_REVIEW")
        ),
        "alerts_by_attack_type": attack_type_counts,
    }


# -------------------- INVESTIGATION MANAGEMENT --------------------

@app.post(
    "/investigations",
    response_model=InvestigationResponse,
    status_code=201,
    tags=["Investigations"],
)
def create_investigation(
    investigation_data: InvestigationCreate,
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    incident = db.get(Incident, investigation_data.incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    existing = db.scalar(
        select(Investigation).where(
            Investigation.incident_id == investigation_data.incident_id
        )
    )

    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="An investigation already exists for this incident",
        )

    investigation = Investigation(
        incident_id=investigation_data.incident_id,
        report_data=investigation_data.report_data,
        processing_status=investigation_data.processing_status,
        workflow_status="NEW",
    )

    try:
        db.add(investigation)
        db.commit()
        db.refresh(investigation)
        return investigation

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save investigation",
        )


@app.get(
    "/investigations",
    response_model=list[InvestigationResponse],
    tags=["Investigations"],
)
def list_investigations(
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    return db.scalars(
        select(Investigation).order_by(Investigation.created_at.desc())
    ).all()


@app.get(
    "/investigations/{investigation_id}",
    response_model=InvestigationResponse,
    tags=["Investigations"],
)
def get_investigation(
    investigation_id: int,
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    investigation = db.get(Investigation, investigation_id)

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    return investigation


# -------------------- RUN INVESTIGATION ENGINE --------------------

@app.post(
    "/investigations/{investigation_id}/run",
    response_model=InvestigationResponse,
    tags=["Investigations"],
)
def run_investigation(
    investigation_id: int,
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    """
    Generate a preliminary rule-based report for an existing investigation
    and save it to PostgreSQL.

    This does not call an AI model or independently verify threat feeds.
    """

    investigation = db.get(Investigation, investigation_id)

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    incident = db.get(Incident, investigation.incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Linked incident not found",
        )

    try:
        # Mark the report as being processed.
        investigation.processing_status = "PROCESSING"
        db.commit()

        # Generate a report using the rule-based engine.
        report = investigate_incident(incident)

        # Store the generated report.
        investigation.report_data = report
        investigation.processing_status = "COMPLETED"

        db.commit()
        db.refresh(investigation)

        return investigation

    except Exception:
        db.rollback()

        # Try to record that report generation failed.
        try:
            failed_investigation = db.get(
                Investigation,
                investigation_id,
            )

            if failed_investigation is not None:
                failed_investigation.processing_status = "FAILED"
                db.commit()
        except Exception:
            db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Could not generate or save investigation report",
        )


# -------------------- MANUAL REPORT UPDATE --------------------

@app.patch(
    "/investigations/{investigation_id}/report",
    response_model=InvestigationResponse,
    tags=["Investigations"],
)
def update_investigation_report(
    investigation_id: int,
    update: InvestigationReportUpdate,
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    investigation = db.get(Investigation, investigation_id)

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    if update.processing_status == "COMPLETED" and not update.report_data:
        raise HTTPException(
            status_code=422,
            detail="Cannot mark an investigation completed with an empty report",
        )

    investigation.report_data = update.report_data
    investigation.processing_status = update.processing_status

    try:
        db.commit()
        db.refresh(investigation)
        return investigation

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not update investigation report",
        )


# -------------------- INVESTIGATION WORKFLOW --------------------

@app.patch(
    "/investigations/{investigation_id}/workflow",
    response_model=InvestigationResponse,
    tags=["Investigations"],
)
def update_investigation_workflow(
    investigation_id: int,
    update: WorkflowUpdate,
    db: Session = Depends(get_db),
    current_admin: str = Depends(get_current_admin),
):
    investigation = db.get(Investigation, investigation_id)

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    investigation.workflow_status = update.workflow_status

    try:
        db.commit()
        db.refresh(investigation)
        return investigation

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not update investigation workflow",
        )
