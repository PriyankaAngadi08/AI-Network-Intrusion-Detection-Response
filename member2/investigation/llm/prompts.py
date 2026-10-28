
import json


def build_investigation_prompt(
    alert: dict,
    iocs: list[dict],
    mitre_technique: dict,
    evidence: list[dict],
) -> str:
    """Build an evidence-grounded prompt for investigating a security alert."""

    context = {
        "alert": alert,
        "iocs": iocs,
        "mitre_technique": mitre_technique,
        "retrieved_evidence": [
            {
                "source": item.get("source"),
                "text": item.get("text"),
            }
            for item in evidence
        ],
    }

    context_json = json.dumps(context, indent=2, ensure_ascii=False)

    return f"""
You are a cybersecurity investigation assistant.

Investigate the alert using only the supplied information and evidence.

Rules:
1. Do not invent threat-intelligence findings.
2. Do not claim an IP is malicious just because it is suspicious or unknown.
3. Treat MITRE ATT&CK mappings as candidates unless evidence supports them.
4. Clearly state when evidence is insufficient.
5. Separate observed facts from possible explanations.
6. Recommend defensive actions; do not execute them.
7. Refer to the supplied document sources when explaining your conclusions.

Return a JSON object with these keys:
- summary
- severity
- observed_facts
- possible_explanation
- evidence
- uncertainties
- recommendations
- sources

Severity must be one of: low, medium, high, critical, or undetermined.

INPUT DATA:
{context_json}
""".strip()
