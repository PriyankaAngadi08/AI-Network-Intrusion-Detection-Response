
from investigation.alert_schema import SecurityAlert
from threat_intelligence.ioc_extractor import extract_iocs


alert = SecurityAlert(
    alert_id="TEST-001",
    source_ip="192.168.1.25",
    destination_ip="192.168.1.10",
    attack_type="PortScan",
    label="ATTACK",
    confidence=0.97,
    dataset="CIC-IDS2017",
)

result = extract_iocs(alert)

print(result)

assert len(result) == 2
assert result[0]["value"] == "192.168.1.25"
assert result[1]["value"] == "192.168.1.10"

print("IOC extraction test passed!")
