
from investigation.alert_schema import SecurityAlert
from investigation.investigator import investigate_alert

alert = SecurityAlert(
    alert_id="TEST-002",
    source_ip="192.168.1.25",
    destination_ip="192.168.1.10",
    source_port=4500,
    destination_port=80,
    protocol="TCP",
    attack_type="PortScan",
    label="ATTACK",
    confidence=0.97,
    dataset="CIC-IDS2017",
)

result = investigate_alert(alert)

print(result)

assert result["alert_id"] == "TEST-002"
assert len(result["iocs"]) == 2
assert result["mitre_technique"]["technique_id"] == "T1046"

print("Investigation pipeline test passed!")
