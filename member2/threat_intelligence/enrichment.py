
from ipaddress import ip_address

# Simulated data for development and testing only.
# These are NOT real threat-intelligence findings.
SAMPLE_INTELLIGENCE = {
    "203.0.113.10": {
        "status": "suspicious",
        "source": "LOCAL_TEST_FEED",
        "reason": "Simulated suspicious indicator for testing",
    },
    "198.51.100.20": {
        "status": "malicious",
        "source": "LOCAL_TEST_FEED",
        "reason": "Simulated malicious indicator for testing",
    },
}


def enrich_ioc(ioc: dict) -> dict:
    """Enrich an IP indicator using a local sample feed."""
    value = ioc.get("value", "")

    try:
        address = ip_address(value)
    except ValueError:
        return {
            **ioc,
            "status": "invalid",
            "source": None,
            "reason": "Invalid IP address",
        }

    normalized_ip = str(address)

    # Check the simulated feed first so test entries match correctly.
    if normalized_ip in SAMPLE_INTELLIGENCE:
        return {
            **ioc,
            **SAMPLE_INTELLIGENCE[normalized_ip],
            "value": normalized_ip,
        }

    # Private, loopback, and other non-global IPs need internal context.
    if not address.is_global:
        return {
            **ioc,
            "value": normalized_ip,
            "status": "internal_or_non_global",
            "source": "IP_ADDRESS_CLASSIFICATION",
            "reason": "Not a globally routable public IP",
        }

    return {
        **ioc,
        "value": normalized_ip,
        "status": "unknown",
        "source": "LOCAL_TEST_FEED",
        "reason": "No matching entry in the sample feed",
    }


def enrich_iocs(iocs: list[dict]) -> list[dict]:
    """Enrich multiple indicators."""
    return [enrich_ioc(ioc) for ioc in iocs]
