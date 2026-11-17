
from datetime import datetime, timezone


def investigate_incident(incident, threat_intel=None):
    """
    Generate a preliminary, rule-based investigation report.
    This is not an AI model or independent threat verification.
    """

    severity = str(
        getattr(incident, "severity", "UNKNOWN")
    ).upper()

    attack_type = str(
        getattr(incident, "attack_type", "UNKNOWN")
    )

    source_ip = getattr(incident, "source_ip", None)
    destination_ip = getattr(incident, "destination_ip", None)
    confidence = getattr(incident, "confidence", None)

    threat_intel = threat_intel or {}

    observed_facts = [
        f"Incident ID: {incident.id}",
        f"Recorded severity: {severity}",
        f"Recorded attack type: {attack_type}",
    ]

    if source_ip:
        observed_facts.append(f"Source IP recorded: {source_ip}")

    if destination_ip:
        observed_facts.append(
            f"Destination IP recorded: {destination_ip}"
        )

    possible_explanations = []

    if "SCAN" in attack_type.upper():
        possible_explanations.append(
            "The recorded attack type indicates scanning activity. "
            "Further evidence is needed to determine intent."
        )
    else:
        possible_explanations.append(
            "The incident classification may indicate suspicious "
            "activity, but additional evidence is needed to establish "
            "what occurred."
        )

    recommendations = [
        "Review the incident and associated network logs.",
        "Verify available indicators using trusted threat-intelligence sources.",
        "Require authorized human review before taking response actions.",
    ]

    report = {
        "summary": (
            f"Incident {incident.id} was recorded with severity "
            f"{severity} and attack type {attack_type}. "
            "This is a preliminary, rule-based report, not a completed "
            "AI investigation."
        ),
        "iocs": threat_intel.get("iocs", []),
        "mitre_technique": [],
        "evidence": [
            {
                "source": "incident_record",
                "detail": {
                    "incident_id": incident.id,
                    "severity": severity,
                    "attack_type": attack_type,
                    "source_ip": source_ip,
                    "destination_ip": destination_ip,
                    "confidence": confidence,
                },
            }
        ],
        "observed_facts": observed_facts,
        "possible_explanations": possible_explanations,
        "uncertainties": [
            "No independent threat-feed verification was performed.",
            "No AI model analysis was performed.",
            "MITRE ATT&CK mapping has not yet been verified.",
        ],
        "recommendations": recommendations,
        "analysis_type": "RULE_BASED_PRELIMINARY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return report
