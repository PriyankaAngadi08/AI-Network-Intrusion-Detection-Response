
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
    "summary": "A possible port scan was detected.",
    "iocs": [],
    "mitre_technique": {
        "technique_id": "T1046",
        "mapping_status": "candidate",
    },
    "retrieved_evidence": [],
    "uncertainties": ["No full packet capture was provided."],
    "recommendations": ["Review the source host's network activity."],
    "sources": ["network_scanning.md"],
}

report = generate_report(sample_investigation)

print(report)

assert report["investigation_id"] == "INV-TEST-001"
assert report["alert_id"] == "TEST-001"
assert report["severity"] == "medium"
assert "created_at" in report

print("Report generator test passed!")
