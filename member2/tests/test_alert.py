from investigation.alert_schema import SecurityAlert


alert = SecurityAlert(
    alert_id="TEST-001",
    timestamp="2026-09-11T19:30:00",
    source_ip="192.168.1.25",
    destination_ip="192.168.1.10",
    source_port=4500,
    destination_port=80,
    protocol="TCP",
    attack_type="PortScan",
    label="ATTACK",
    confidence=0.97,
    dataset="TEST"
)

print(alert)