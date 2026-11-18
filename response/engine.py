
from datetime import datetime, timezone


def evaluate_incident(incident):
    """
    Recommend a response based on incident severity.
    This function does not execute real security actions.
    """

    severity = incident.severity.upper()

    if severity == "CRITICAL":
        action = "ISOLATION_REQUIRES_ADMIN_APPROVAL"
        priority = "IMMEDIATE"

    elif severity == "HIGH":
        action = "INVESTIGATE_AND_REQUEST_ADMIN_REVIEW"
        priority = "HIGH"

    elif severity == "MEDIUM":
        action = "INVESTIGATE_AND_MONITOR"
        priority = "NORMAL"

    else:
        action = "LOG_AND_MONITOR"
        priority = "LOW"

    return {
        "incident_id": incident.id,
        "severity": severity,
        "recommended_action": action,
        "priority": priority,
        "requires_admin_approval": severity in ["HIGH", "CRITICAL"],
        "executed": False,
        "message": "Recommendation only. No security action was executed.",
        "evaluated_at": datetime.now(timezone.utc).isoformat()
    }
