
from ipaddress import ip_address
from investigation.alert_schema import SecurityAlert


def extract_iocs(alert: SecurityAlert) -> list[dict]:
    """Extract available IP indicators from a security alert."""

    iocs = []

    indicators = [
        ("source", alert.source_ip),
        ("destination", alert.destination_ip),
    ]

    for role, value in indicators:
        if not value:
            continue

        try:
            normalized_ip = str(ip_address(value))
        except ValueError:
            continue

        ioc = {
            "type": "IP",
            "value": normalized_ip,
            "role": role,
        }

        if ioc not in iocs:
            iocs.append(ioc)

    return iocs
