
from investigation.reports.report_generator import generate_report


sample_investigation = {
    "investigation_id": "INV-TEST-001",
    "alert_id": "TEST-001",
    "status": "completed",
    "attack_type": "PortScan",
    "label": "ATTACK",
    "detection_confidence": 0.97,
    "dataset": "CIC-IDS2017",
    "severity": "medium",
    "summary": "Possible port scanning activity detected.",
    "iocs": [
        {
            "type": "IP",
            "value": "192.168.1.25",
            "role": "source",
            "status": "internal_or_non_global",
        }
    ],
    "mitre_technique": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "tactic": "Discovery",
        "mapping_status": "candidate",
    },
    "retrieved_evidence": [
        {
            "source": "network_scanning.md",
            "text": "Port scanning can identify open ports and services.",
            "similarity": 0.82,
        }
    ],
    "uncertainties": [
        "The alert alone does not prove a compromise."
    ],
    "recommendations": [
        "Review the source host's network activity."
    ],
    "sources": ["network_scanning.md"],
    "ai_analysis": {},
}


# Generate the final investigation report
report = generate_report(sample_investigation)


# Check required report fields
required_fields = [
    "investigation_id",
    "alert_id",
    "created_at",
    "status",
    "severity",
    "summary",
    "iocs",
    "mitre_technique",
    "evidence",
    "uncertainties",
    "recommendations",
    "sources",
]

for field in required_fields:
    assert field in report, f"Missing field: {field}"


# Check important values
assert report["investigation_id"] == "INV-TEST-001"
assert report["alert_id"] == "TEST-001"
assert report["status"] == "completed"
assert report["severity"] == "medium"
assert report["iocs"] == sample_investigation["iocs"]
assert report["mitre_technique"] == sample_investigation["mitre_technique"]
assert report["evidence"] == sample_investigation["retrieved_evidence"]

print("Integration contract test passed!")
print("Report fields:", list(report.keys()))
print("Investigation ID:", report["investigation_id"])
print("Alert ID:", report["alert_id"])
print("Severity:", report["severity"])
