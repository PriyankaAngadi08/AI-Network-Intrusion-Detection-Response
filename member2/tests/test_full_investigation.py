
from investigation.alert_schema import SecurityAlert
from investigation.investigator import investigate_alert


alert = SecurityAlert(
    alert_id="FULL-TEST-001",
    timestamp="2026-10-28T10:00:00",
    source_ip="203.0.113.10",
    destination_ip="192.168.1.10",
    source_port=4500,
    destination_port=80,
    protocol="TCP",
    attack_type="PortScan",
    label="ATTACK",
    confidence=0.97,
    dataset="TEST"
)

print("Starting full investigation...")

try:
    report = investigate_alert(alert)

    print("\nInvestigation completed!")
    print("Investigation ID:", report["investigation_id"])
    print("Alert ID:", report["alert_id"])
    print("Attack Type:", report["attack_type"])
    print("Severity:", report["severity"])
    print("Summary:", report["summary"])
    print("Recommendations:", report["recommendations"])

    assert report["investigation_id"] == "INV-FULL-TEST-001"
    assert report["alert_id"] == "FULL-TEST-001"
    assert report["status"] == "completed"

    print("\nFull investigation test passed!")

except Exception as error:
    print("\nInvestigation failed.")
    print("Error:", error)
    raise
