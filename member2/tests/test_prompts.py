
from investigation.llm.prompts import build_investigation_prompt

prompt = build_investigation_prompt(
    alert={
        "alert_id": "TEST-003",
        "attack_type": "PortScan",
        "label": "ATTACK",
        "confidence": 0.97,
    },
    iocs=[
        {
            "type": "IP",
            "value": "192.168.1.25",
            "role": "source",
            "status": "internal_or_non_global",
        }
    ],
    mitre_technique={
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "mapping_status": "candidate",
    },
    evidence=[
        {
            "source": "network_scanning.md",
            "text": "Port scanning identifies hosts and open ports.",
        }
    ],
)

print(prompt[:500])

assert "PortScan" in prompt
assert "network_scanning.md" in prompt
assert "Do not invent threat-intelligence findings" in prompt

print("Prompt builder test passed!")
