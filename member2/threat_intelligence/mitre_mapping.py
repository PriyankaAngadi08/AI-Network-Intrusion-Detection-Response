
# Basic MITRE ATT&CK mapping for common network alerts.
# These mappings are candidates, not proof that a technique occurred.

MITRE_MAPPING = {
    "PortScan": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "tactic": "Discovery",
    },
    "DoS": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact",
    },
    "DDoS": {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact",
    },
    "Brute Force": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access",
    },
}


def map_attack(attack_type: str) -> dict:
    """Return a candidate MITRE ATT&CK mapping for an alert."""
    normalized_type = attack_type.strip().lower()

    for name, details in MITRE_MAPPING.items():
        if name.lower() == normalized_type:
            return {
                **details,
                "matched_attack_type": attack_type,
                "mapping_status": "candidate",
            }

    return {
        "technique_id": None,
        "technique_name": None,
        "tactic": None,
        "matched_attack_type": attack_type,
        "mapping_status": "unmapped",
        "reason": "No mapping is configured for this attack type",
    }
