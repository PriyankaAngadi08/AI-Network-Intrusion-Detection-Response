
from datetime import datetime, timezone


def generate_report(investigation: dict) -> dict:
    """Validate and format an investigation result for the backend."""

    required_fields = ["investigation_id", "alert_id", "status"]

    for field in required_fields:
        if not investigation.get(field):
            raise ValueError(f"Missing required field: {field}")

    ai_analysis = investigation.get("ai_analysis", {})

    return {
        "investigation_id": investigation["investigation_id"],
        "alert_id": investigation["alert_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": investigation["status"],
        "attack_type": investigation.get("attack_type"),
        "label": investigation.get("label"),
        "detection_confidence": investigation.get("detection_confidence"),
        "dataset": investigation.get("dataset"),
        "severity": investigation.get(
            "severity", ai_analysis.get("severity", "undetermined")
        ),
        "summary": investigation.get(
            "summary", "No summary available."
        ),
        "iocs": investigation.get("iocs", []),
        "mitre_technique": investigation.get("mitre_technique", {}),
        "evidence": investigation.get("retrieved_evidence", []),
        "uncertainties": investigation.get(
            "uncertainties", ai_analysis.get("uncertainties", [])
        ),
        "recommendations": investigation.get(
            "recommendations", ai_analysis.get("recommendations", [])
        ),
        "sources": investigation.get(
            "sources", ai_analysis.get("sources", [])
        ),
    }
