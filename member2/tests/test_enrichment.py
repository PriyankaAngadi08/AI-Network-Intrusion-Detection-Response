
from threat_intelligence.enrichment import enrich_iocs


sample_iocs = [
    {
        "type": "IP",
        "value": "203.0.113.10",
        "role": "source",
    },
    {
        "type": "IP",
        "value": "192.168.1.25",
        "role": "destination",
    },
    {
        "type": "IP",
        "value": "8.8.8.8",
        "role": "source",
    },
]

results = enrich_iocs(sample_iocs)

for result in results:
    print(result)

assert results[0]["status"] == "suspicious"
assert results[1]["status"] == "internal_or_non_global"
assert results[2]["status"] == "unknown"

print("Threat intelligence enrichment test passed!")
